import os
import torch
import torchaudio
import librosa
import random
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import json # Import json for saving class map
import sys # Import sys for path manipulation

# --- CONFIGURATION ---
BUCKET_NAME = "voice-model-trainer-b6814.firebasestorage.app"
DOWNLOAD_DIR = "user_training_data"
SAMPLE_RATE = 16000
DURATION = 3  # seconds
NUM_CLASSES = 0 # Will be set dynamically
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VGGish-specific Mel Spectrogram Parameters (from vggish_params.py) ---
# These parameters are crucial for generating Mel features compatible with VGGish
VGGISH_SAMPLE_RATE = 16000
VGGISH_STFT_WINDOW_LENGTH_SECONDS = 0.025 # 25ms
VGGISH_STFT_HOP_LENGTH_SECONDS = 0.010    # 10ms
VGGISH_MEL_BINS = 64
VGGISH_MEL_MIN_HZ = 125
VGGISH_MEL_MAX_HZ = 7500
VGGISH_EXAMPLE_WINDOW_SECONDS = 0.96 # 960ms
VGGISH_EXAMPLE_HOP_SECONDS = 0.96    # 960ms (no overlap for examples)

# Calculated parameters
VGGISH_STFT_WINDOW_LENGTH_SAMPLES = int(round(VGGISH_STFT_WINDOW_LENGTH_SECONDS * VGGISH_SAMPLE_RATE))
VGGISH_STFT_HOP_LENGTH_SAMPLES = int(round(VGGISH_STFT_HOP_LENGTH_SECONDS * VGGISH_SAMPLE_RATE))
VGGISH_EXAMPLE_WINDOW_SAMPLES = int(round(VGGISH_EXAMPLE_WINDOW_SECONDS * VGGISH_SAMPLE_RATE))
VGGISH_EXAMPLE_HOP_SAMPLES = int(round(VGGISH_EXAMPLE_HOP_SECONDS * VGGISH_SAMPLE_RATE))

# --- AUDIO AUGMENTATION ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# --- INITIALIZE FIREBASE ---
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': BUCKET_NAME
    })
    bucket = storage.bucket()
except ValueError:
    print("Firebase app already initialized or firebase_key.json not found. Skipping initialization.")
except FileNotFoundError:
    print("firebase_key.json not found. Ensure it's in the root directory for local testing.")

def download_training_data():
    try:
        import shutil
        if os.path.exists(DOWNLOAD_DIR):
            print(f"Removing existing {DOWNLOAD_DIR} directory...")
            shutil.rmtree(DOWNLOAD_DIR)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except ImportError:
        print("Shutil not available, skipping directory cleanup. Ensure directory is empty.")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if 'bucket' not in globals() or bucket is None:
        print("Firebase bucket not initialized. Skipping data download.")
        return

    print(f"Downloading data from Firebase bucket '{BUCKET_NAME}'...")
    blobs = bucket.list_blobs(prefix="user_training_data/")
    downloaded_count = 0
    for blob in blobs:
        if blob.name.endswith(".wav"):
            local_path_parts = blob.name.split("/")[1:]
            if not local_path_parts:
                continue
            local_path = os.path.join(DOWNLOAD_DIR, *local_path_parts)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded {blob.name}")
            downloaded_count += 1
    if downloaded_count == 0:
        print(f"⚠️ No .wav files found in '{DOWNLOAD_DIR}' prefix in Firebase Storage.")

# --- Custom VGGish-style waveform to examples function ---
def custom_waveform_to_examples(data, sample_rate):
    """
    Converts audio waveform into a sequence of log-Mel spectrogram examples,
    compatible with VGGish input requirements.
    """
    if data.ndim == 1:
        # Single audio waveform, reshape to (1, num_samples) for consistent batching
        data = np.expand_dims(data, 0)
    
    data = data.astype(np.float32)

    all_examples = []
    for waveform in data:
        # 1. Compute STFT magnitude spectrogram
        n_fft = VGGISH_STFT_WINDOW_LENGTH_SAMPLES
        hop_length = VGGISH_STFT_HOP_LENGTH_SAMPLES

        # Pad waveform if it's too short to create even one frame
        if len(waveform) < n_fft:
            waveform = np.pad(waveform, (0, n_fft - len(waveform)), mode='constant')

        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=VGGISH_MEL_BINS,
            fmin=VGGISH_MEL_MIN_HZ,
            fmax=VGGISH_MEL_MAX_HZ,
            power=1.0
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0, top_db=None)
        log_mel_spectrogram = log_mel_spectrogram.T # Transpose to (num_frames, num_mel_bins)

        # 2. Frame into 0.96-second examples
        frames_per_example = int(round(VGGISH_EXAMPLE_WINDOW_SECONDS / VGGISH_STFT_HOP_LENGTH_SECONDS))
        hop_frames = int(round(VGGISH_EXAMPLE_HOP_SECONDS / VGGISH_STFT_HOP_LENGTH_SECONDS))

        if frames_per_example == 0:
            frames_per_example = 1

        if log_mel_spectrogram.shape[0] < frames_per_example:
            pad_needed = frames_per_example - log_mel_spectrogram.shape[0]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, pad_needed), (0, 0)), mode='constant')
            print(f"DEBUG: Padded log_mel_spectrogram to ensure enough frames for examples. New shape: {log_mel_spectrogram.shape}")

        framed_examples = librosa.util.frame(
            log_mel_spectrogram,
            frame_length=frames_per_example,
            hop_length=hop_frames,
            axis=0
        )
        
        all_examples.append(framed_examples)

    if not all_examples:
        return np.empty((0, frames_per_example, VGGISH_MEL_BINS), dtype=np.float32)

    return np.concatenate(all_examples, axis=0)

# --- NEW: Custom VGGish Model Architecture (copied from torchvggish/vggish.py) ---
class CustomVGGish(nn.Module):
    def __init__(self):
        super(CustomVGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096), # Adjusted based on 96x64 input and pooling
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True) # VGGish output is ReLU'd
        )

    def forward(self, x):
        # Input x is (batch_size, num_frames, num_mel_bins) e.g., (N, 96, 64)
        # VGGish expects (batch_size, 1, num_frames, num_mel_bins) for Conv2d
        x = x.unsqueeze(1) # Add channel dimension: (N, 1, 96, 64)
        
        x = self.features(x)
        
        # Flatten the output of convolutional layers
        # The output of the last MaxPool2d (kernel_size=2, stride=2) on 96x64 input:
        # 96 / 2 / 2 / 2 / 2 = 6
        # 64 / 2 / 2 / 2 / 2 = 4
        # So, the feature map size is 6x4 (or 4x6 depending on which dimension is which).
        # Let's verify this carefully.
        # Input: (1, 96, 64)
        # Pool1 (2x2): (1, 48, 32)
        # Pool2 (2x2): (1, 24, 16)
        # Pool3 (2x2): (1, 12, 8)
        # Pool4 (2x2): (1, 6, 4)
        # Output channels from last conv: 512
        # So, the flattened size should be 512 * 6 * 4 = 12288
        
        x = x.permute(0, 2, 3, 1).contiguous() # (N, 6, 4, 512)
        x = x.view(x.size(0), -1) # Flatten to (N, 512 * 6 * 4) = (N, 12288)

        x = self.embeddings(x)
        return x # Output shape: (N, 128)

# --- END Custom VGGish Model Architecture ---


# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        target_length_samples = SAMPLE_RATE * DURATION
        wav = np.zeros(target_length_samples, dtype=np.float32) 
        
        try:
            read_wav, sr = sf.read(path)
            
            if read_wav is None or read_wav.size == 0:
                print(f"⚠️ Warning: Audio file {path} is empty after sf.read. Using zero-padded array.")
            else:
                resampled_wav = librosa.resample(read_wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                
                if len(resampled_wav) < target_length_samples:
                    wav = np.pad(resampled_wav, (0, target_length_samples - len(resampled_wav)), 'constant')
                elif len(resampled_wav) > target_length_samples:
                    wav = resampled_wav[:target_length_samples]
                else:
                    wav = resampled_wav

        except Exception as e:
            print(f"❌ Error during audio read/resample for {path}: {e}. Using zero-padded array.")
        
        if wav.size != target_length_samples:
            print(f"❌ Critical Error: Final wav size mismatch for {path}. Expected {target_length_samples}, got {wav.size}. Forcing to zero array.")
            wav = np.zeros(target_length_samples, dtype=np.float32)

        if self.augment:
            if wav.dtype != np.float32:
                print(f"DEBUG: Converting audio from {wav.dtype} to float32 before augmentation for {path}")
                wav = wav.astype(np.float32)
            wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

        print(f"DEBUG: __getitem__ output wav shape: {wav.shape}")
        print(f"DEBUG: __getitem__ output wav dtype: {wav.dtype}")
        if wav.size > 0:
            print(f"DEBUG: __getitem__ output wav min/max/mean: {wav.min():.4f}/{wav.max():.4f}/{wav.mean():.4f}")
            print(f"DEBUG: __getitem__ output wav contains NaN: {np.isnan(wav).any()}")
            print(f"DEBUG: __getitem__ output wav contains Inf: {np.isinf(wav).any()}")
        else:
            print("DEBUG: __getitem__ output wav is empty.")

        return torch.tensor(wav).float(), label


# --- MODEL (Transfer Learning with VGGish) ---
class VGGishFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggish_base = CustomVGGish() # Use our custom VGGish model
        self.vggish_base.eval() # Set to eval mode

        # Load pre-trained weights from the original VGGish model
        try:
            # Load the original VGGish model to get its state_dict
            original_vggish_model = torch.hub.load('harritaylor/pytorch-vggish', 'vggish', pretrained=True)
            
            # Load the state_dict into our custom model
            self.vggish_base.load_state_dict(original_vggish_model.state_dict())
            print("DEBUG: Pre-trained VGGish weights successfully loaded into CustomVGGish.")
            
            # Freeze parameters of the base VGGish model
            for param in self.vggish_base.parameters():
                param.requires_grad = False
            print("DEBUG: CustomVGGish parameters frozen.")

        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load pre-trained VGGish weights or configure CustomVGGish in __init__: {type(e).__name__}: {e}")
            # If loading fails, the model will proceed with randomly initialized weights for VGGish_base,
            # but it will print an error, and training might be less effective.

    def forward(self, x):
        # x is [batch_size, num_samples] (raw audio)
        if x.numel() == 0:
            return torch.zeros(x.shape[0], 128).to(x.device)

        numpy_wav = x.cpu().contiguous().numpy().astype(np.float32)
        
        print(f"DEBUG: Input to VGGishFeatureExtractor.forward: x.shape={x.shape}, x.dtype={x.dtype}")
        print(f"DEBUG: numpy_wav shape before custom_waveform_to_examples: {numpy_wav.shape}")
        
        try:
            examples_batch_np = custom_waveform_to_examples(numpy_wav, SAMPLE_RATE)
            
            print(f"DEBUG: examples_batch_np shape after custom_waveform_to_examples: {examples_batch_np.shape}")
            print(f"DEBUG: examples_batch_np dtype after custom_waveform_to_examples: {examples_batch_np.dtype}")
            if examples_batch_np.size > 0:
                print(f"DEBUG: examples_batch_np min/max/mean: {examples_batch_np.min():.4f}/{examples_batch_np.max():.4f}/{examples_batch_np.mean():.4f}")
                print(f"DEBUG: examples_batch_np contains NaN: {np.isnan(examples_batch_np).any()}")
                print(f"DEBUG: examples_batch_np contains Inf: {np.isinf(examples_batch_np).any()}")
            else:
                print("DEBUG: examples_batch_np is empty.")

        except Exception as e:
            print(f"❌ ERROR: Exception in custom_waveform_to_examples: {type(e).__name__}: {e}")
            print("Returning zero-filled examples_batch (NumPy) to prevent crash.")
            examples_batch_np = np.zeros((x.shape[0], 96, 64), dtype=np.float32)

        examples_batch = torch.from_numpy(examples_batch_np).to(x.device)

        # --- NEW: Call forward on our CustomVGGish model ---
        try:
            # Our CustomVGGish model expects (N, 96, 64) and outputs (N, 128)
            embeddings = self.vggish_base(examples_batch) # Pass through our custom VGGish
            print(f"DEBUG: Embeddings shape after self.vggish_base.forward: {embeddings.shape}")

            # Calculate num_segments_per_audio based on the actual output of embeddings
            if x.shape[0] > 0:
                num_segments_per_audio = embeddings.shape[0] // x.shape[0]
            else:
                num_segments_per_audio = 1

            if num_segments_per_audio == 0:
                num_segments_per_audio = 1
            
            print(f"DEBUG: Dynamically calculated num_segments_per_audio: {num_segments_per_audio}")
            print(f"DEBUG: Expected total elements for reshape (batch_size * num_segments_per_audio * 128): {x.shape[0] * num_segments_per_audio * 128}")
            print(f"DEBUG: Actual embeddings.numel(): {embeddings.numel()}")

            # Reshape embeddings to (batch_size, num_segments_per_audio, 128)
            try:
                reshaped_embeddings = embeddings.view(x.shape[0], num_segments_per_audio, 128)
                print(f"DEBUG: Reshaped embeddings shape: {reshaped_embeddings.shape}")
            except RuntimeError as e:
                print(f"❌ CRITICAL ERROR: RuntimeError during reshape of embeddings: {type(e).__name__}: {e}")
                print(f"   Attempting to reshape embeddings.shape={embeddings.shape} to ({x.shape[0]}, {num_segments_per_audio}, 128)")
                return torch.zeros(x.shape[0], 128).to(x.device)

            # Average pool the embeddings across the time segments (dim=1)
            pooled_embeddings = torch.mean(reshaped_embeddings, dim=1) # Output shape: [batch_size, 128]
            print(f"DEBUG: Pooled embeddings shape: {pooled_embeddings.shape}")

        except Exception as e:
            print(f"❌ ERROR: Exception during self.vggish_base.forward or subsequent pooling: {type(e).__name__}: {e}")
            print("Returning zero-filled embeddings to prevent crash.")
            pooled_embeddings = torch.zeros(x.shape[0], 128).to(x.device)

        return pooled_embeddings

class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = VGGishFeatureExtractor()

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        print(f"DEBUG: Features shape before classifier: {features.shape}")
        output = self.classifier(features)
        return output

# --- MAIN TRAINING ---
def main():
    download_training_data()

    all_audio_paths = []
    all_audio_labels = []
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in files:
            if f.endswith(".wav"):
                all_audio_paths.append(os.path.join(root, f))
                all_audio_labels.append(os.path.basename(root))

    if not all_audio_paths:
        print("No audio files found for training. Please ensure 'user_training_data' in Firebase Storage contains .wav files in subdirectories (e.g., user_training_data/speaker1/audio.wav).")
        print("Training cannot proceed without data.")
        return

    valid_audio_paths = []
    valid_audio_labels = []
    min_vggish_input_samples = int(VGGISH_STFT_WINDOW_LENGTH_SECONDS * SAMPLE_RATE)
    print(f"DEBUG: Using VGGish params for pre-validation - STFT_WINDOW_LENGTH_SECONDS={VGGISH_STFT_WINDOW_LENGTH_SECONDS}, Calculated min_vggish_input_samples={min_vggish_input_samples}")

    print("Pre-validating audio files...")
    for i, path in enumerate(all_audio_paths):
        try:
            wav_check, sr_check = sf.read(path)
            if wav_check is None or wav_check.size == 0:
                print(f"Skipping empty audio file: {path}")
                continue
            
            resampled_wav_check = librosa.resample(wav_check, orig_sr=sr_check, target_sr=SAMPLE_RATE)
            
            if len(resampled_wav_check) < min_vggish_input_samples:
                print(f"Skipping audio file too short for VGGish framing ({len(resampled_wav_check)} samples, min needed {min_vggish_input_samples}): {path}")
                continue

            valid_audio_paths.append(path)
            valid_audio_labels.append(all_audio_labels[i])

        except Exception as e:
            print(f"Skipping problematic audio file (error during pre-validation read/resample): {path} - {e}")
            continue

    if not valid_audio_paths:
        print("No valid audio files found after pre-validation. Training cannot proceed.")
        return

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(valid_audio_labels)
    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)
    print(f"Found {len(valid_audio_paths)} valid audio files across {NUM_CLASSES} classes.")

    with open("class_to_label.json", "w") as f:
        json.dump({str(i): label for i, label in enumerate(label_encoder.classes_)}, f)
    print("Class map saved to class_to_label.json")

    label_counts = np.bincount(encoded_labels)
    can_stratify = all(count >= 2 for count in label_counts) and len(valid_audio_paths) >= 2

    if not can_stratify:
        print("Not enough samples per class for stratified split or total samples < 2. Using non-stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(
            valid_audio_paths, encoded_labels, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            valid_audio_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )

    train_dataset = VoiceDataset(X_train, y_train, augment=True)
    val_dataset = VoiceDataset(X_val, y_val, augment=False)

    def custom_collate_fn(batch):
        filtered_batch = [item for item in batch if item[1] != -1]
        
        if not filtered_batch:
            return torch.empty(0, SAMPLE_RATE * DURATION).float(), torch.empty(0, dtype=torch.long)
        
        audios = torch.stack([item[0] for item in filtered_batch])
        labels = torch.tensor([item[1] for item in filtered_batch])
        return audios, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    model = TransferLearningCNN(NUM_CLASSES).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Using device: {DEVICE}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if x.numel() == 0:
                print(f"  Skipping empty batch {batch_idx+1} in training (no valid samples).")
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Train Loss: {loss.item():.4f}")


        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_idx_val, (x, y) in enumerate(val_loader):
                if x.numel() == 0:
                    print(f"  Skipping empty batch {batch_idx_val+1} in validation (no valid samples).")
                    continue

                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "best_voice_model.pt")
            print("✅ Model saved (best)")

    print("🎉 Training complete")

if __name__ == "__main__":
    import shutil
    main()
