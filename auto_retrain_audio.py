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

# --- CRITICAL: Ensure VGGish related modules are loaded globally and early ---
# This block ensures torchvggish is available before any other code uses it.
vggish_input = None
vggish_params = None

try:
    # First, ensure the repository is cloned and cached by torch.hub
    # This call is blocking and will download the repo if not present.
    print("Ensuring VGGish repository is cloned to torch.hub cache...")
    # Use a dummy variable for the model as we just need the side effect of cloning
    _ = torch.hub.load('harritaylor/pytorch-vggish', 'vggish', verbose=False)
    print("VGGish repository successfully cloned/found in cache.")

    vggish_repo_path = os.path.join(torch.hub.get_dir(), 'harritaylor_pytorch-vggish_master')
    torchvggish_path = os.path.join(vggish_repo_path, 'torchvggish')

    if torchvggish_path not in sys.path:
        sys.path.insert(0, torchvggish_path)
        print(f"Added {torchvggish_path} to sys.path for VGGish modules.")

    # Now, import the modules directly since the path is set
    from torchvggish import vggish_input as imported_vggish_input
    from torchvggish import vggish_params as imported_vggish_params
    vggish_input = imported_vggish_input
    vggish_params = imported_vggish_params
    print("Successfully imported vggish_input and vggish_params globally.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to setup VGGish module imports globally. Training will likely fail. Error: {e}")
    # vggish_input and vggish_params remain None, which will trigger errors later if used.

# --- END CRITICAL GLOBAL IMPORT BLOCK ---


# --- CONFIGURATION ---
BUCKET_NAME = "voice-model-trainer-b6814.firebasestorage.app"
DOWNLOAD_DIR = "user_training_data"
SAMPLE_RATE = 16000
DURATION = 3  # seconds
NUM_CLASSES = 0 # Will be set dynamically
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AUDIO AUGMENTATION ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

# --- INITIALIZE FIREBASE ---
# Note: In a real GitHub Actions environment, 'firebase_key.json' is created by the workflow.
# For local testing, ensure firebase_key.json is present or mock firebase_admin.
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': BUCKET_NAME
    })
    bucket = storage.bucket()
except ValueError:
    print("Firebase app already initialized or firebase_key.json not found. Skipping initialization.")
    # This block handles cases where the app might be initialized elsewhere or key is missing for local dev.
    # In GitHub Actions, the key will always be created.
except FileNotFoundError:
    print("firebase_key.json not found. Ensure it's in the root directory for local testing.")
    # For GitHub Actions, this file is created by the workflow.

def download_training_data():
    # Only import shutil if needed, to avoid potential import errors if not present
    try:
        import shutil
        if os.path.exists(DOWNLOAD_DIR):
            print(f"Removing existing {DOWNLOAD_DIR} directory...")
            shutil.rmtree(DOWNLOAD_DIR)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except ImportError:
        print("Shutil not available, skipping directory cleanup. Ensure directory is empty.")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Mock bucket for local testing if Firebase isn't fully set up
    if 'bucket' not in globals() or bucket is None:
        print("Firebase bucket not initialized. Skipping data download.")
        return

    print(f"Downloading data from Firebase bucket '{BUCKET_NAME}'...")
    blobs = bucket.list_blobs(prefix="user_training_data/")
    downloaded_count = 0
    for blob in blobs:
        if blob.name.endswith(".wav"):
            # Construct local path, skipping the 'user_training_data/' prefix
            local_path_parts = blob.name.split("/")[1:]
            if not local_path_parts: # Handle case where blob name is just "user_training_data/"
                continue
            local_path = os.path.join(DOWNLOAD_DIR, *local_path_parts)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded {blob.name}")
            downloaded_count += 1
    if downloaded_count == 0:
        print(f"⚠️ No .wav files found in '{DOWNLOAD_DIR}' prefix in Firebase Storage.")

# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment
        # No Mel Spectrogram transform here, VGGish handles it internally from raw audio

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        target_length_samples = SAMPLE_RATE * DURATION
        # Initialize wav with zeros of the target length. This ensures a valid array even if read fails.
        wav = np.zeros(target_length_samples, dtype=np.float32) 
        
        try:
            read_wav, sr = sf.read(path)
            
            # If read_wav is empty or invalid, proceed with the zero-initialized 'wav'
            if read_wav is None or read_wav.size == 0:
                print(f"⚠️ Warning: Audio file {path} is empty after sf.read. Using zero-padded array.")
            else:
                # Resample. If resampling fails, catch the exception.
                resampled_wav = librosa.resample(read_wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                
                # Apply padding or truncation
                if len(resampled_wav) < target_length_samples:
                    wav = np.pad(resampled_wav, (0, target_length_samples - len(resampled_wav)), 'constant')
                elif len(resampled_wav) > target_length_samples:
                    wav = resampled_wav[:target_length_samples]
                else:
                    wav = resampled_wav # Length is already correct

        except Exception as e:
            print(f"❌ Error during audio read/resample for {path}: {e}. Using zero-padded array.")
            # 'wav' remains the zero-initialized array
        
        # Final check to ensure 'wav' is of the correct size before augmentation/tensor conversion
        # This should always be true due to initialization and padding, but as a safeguard.
        if wav.size != target_length_samples:
            print(f"❌ Critical Error: Final wav size mismatch for {path}. Expected {target_length_samples}, got {wav.size}. Forcing to zero array.")
            wav = np.zeros(target_length_samples, dtype=np.float32)

        # Apply augmentation to the (potentially zero-padded) waveform
        if self.augment:
            # Check dtype before augmentation if it's float64
            if wav.dtype == np.float64:
                print(f"DEBUG: Converting audio from float64 to float32 before augmentation for {path}")
                wav = wav.astype(np.float32)
            wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

        # Debugging prints for wav after all processing in __getitem__
        print(f"DEBUG: __getitem__ output wav shape: {wav.shape}")
        print(f"DEBUG: __getitem__ output wav dtype: {wav.dtype}")
        if wav.size > 0:
            print(f"DEBUG: __getitem__ output wav min/max/mean: {wav.min():.4f}/{wav.max():.4f}/{wav.mean():.4f}")
            print(f"DEBUG: __getitem__ output wav contains NaN: {np.isnan(wav).any()}")
            print(f"DEBUG: __getitem__ output wav contains Inf: {np.isinf(wav).any()}")
        else:
            print("DEBUG: __getitem__ output wav is empty.")

        # Convert to tensor. Label is always valid as it comes from pre-filtered list.
        return torch.tensor(wav).float(), label


# --- MODEL (Transfer Learning with VGGish) ---
class VGGishFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGGish model from PyTorch Hub
        # This will download the repository to the cache if not present
        # Note: torch.hub.load was already called globally at the top to ensure path setup.
        self.vggish = torch.hub.load('harritaylor/pytorch-vggish', 'vggish')

        # Set VGGish to evaluation mode (important for consistent feature extraction)
        self.vggish.eval()

        # Freeze VGGish parameters to use it as a fixed feature extractor
        for param in self.vggish.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x is [batch_size, num_samples] (raw audio)
        # We now explicitly preprocess the waveform to VGGish examples (Mel Spectrograms)
        # This bypasses the problematic _preprocess method in vggish.py
        # vggish_input.waveform_to_examples expects a 1D numpy array or a 2D tensor [batch_size, samples]
        # and returns [batch_size, num_frames, num_mels]

        # Ensure vggish_input is not None before using it
        if vggish_input is None: # Use the globally imported vggish_input
            raise RuntimeError("vggish_input module was not successfully loaded globally. Check script startup.")

        # Move tensor to CPU for numpy conversion, then back to device
        # Ensure x is contiguous before converting to numpy
        # Handle potential empty input from collate_fn if all samples in a batch were problematic
        if x.numel() == 0: # Check if the tensor is empty
            # Return a dummy tensor of the expected output shape if input is empty
            # This prevents crashing but means this batch won't contribute to training
            return torch.zeros(x.shape[0], 128).to(x.device) # Assuming x.shape[0] is batch size

        # Explicitly cast to float32 and ensure 1D or 2D for waveform_to_examples
        # If batch size is 1, flatten to 1D array as waveform_to_examples can take (N,) or (B, N)
        if x.dim() == 2 and x.shape[0] == 1:
            numpy_wav = x.cpu().contiguous().numpy().flatten().astype(np.float32)
        else:
            numpy_wav = x.cpu().contiguous().numpy().astype(np.float32)
        
        # Debugging prints for numpy_wav and VGGish parameters
        print(f"DEBUG: Input to VGGishFeatureExtractor.forward: x.shape={x.shape}, x.dtype={x.dtype}")
        print(f"DEBUG: numpy_wav shape before waveform_to_examples: {numpy_wav.shape}")
        print(f"DEBUG: numpy_wav dtype before waveform_to_examples: {numpy_wav.dtype}")
        if numpy_wav.size > 0:
            print(f"DEBUG: numpy_wav min/max/mean: {numpy_wav.min():.4f}/{numpy_wav.max():.4f}/{numpy_wav.mean():.4f}")
            print(f"DEBUG: numpy_wav contains NaN: {np.isnan(numpy_wav).any()}")
            print(f"DEBUG: numpy_wav contains Inf: {np.isinf(numpy_wav).any()}")
        else:
            print("DEBUG: numpy_wav is empty.")
        
        if vggish_params is not None:
            print(f"DEBUG: VGGish Params - STFT_WINDOW_LENGTH_SECONDS: {vggish_params.STFT_WINDOW_LENGTH_SECONDS}")
            print(f"DEBUG: VGGish Params - STFT_HOP_LENGTH_SECONDS: {vggish_params.STFT_HOP_LENGTH_SECONDS}")
            print(f"DEBUG: VGGish Params - SAMPLE_RATE: {vggish_params.SAMPLE_RATE}")
        else:
            print("DEBUG: vggish_params module not loaded, cannot print internal VGGish parameters.")

        # --- NEW: Aggressive error handling for waveform_to_examples and type conversion ---
        try:
            temp_examples_batch = vggish_input.waveform_to_examples(numpy_wav, SAMPLE_RATE)
            
            # Explicitly convert to numpy array if it's a Tensor, detaching first
            if isinstance(temp_examples_batch, torch.Tensor):
                print("DEBUG: vggish_input.waveform_to_examples returned a Tensor. Detaching and converting to NumPy array.")
                examples_batch_np = temp_examples_batch.detach().cpu().numpy() # ADDED .detach()
            else:
                examples_batch_np = temp_examples_batch # Already a NumPy array

            # --- NEW: Debugging prints for examples_batch_np ---
            print(f"DEBUG: examples_batch_np shape after conversion: {examples_batch_np.shape}")
            print(f"DEBUG: examples_batch_np dtype after conversion: {examples_batch_np.dtype}")
            if examples_batch_np.size > 0:
                print(f"DEBUG: examples_batch_np min/max/mean: {examples_batch_np.min():.4f}/{examples_batch_np.max():.4f}/{examples_batch_np.mean():.4f}")
                print(f"DEBUG: examples_batch_np contains NaN: {np.isnan(examples_batch_np).any()}")
                print(f"DEBUG: examples_batch_np contains Inf: {np.isinf(examples_batch_np).any()}")
            else:
                print("DEBUG: examples_batch_np is empty.")
            # --- END NEW ---

        except Exception as e: # Catch any unexpected errors, including ValueError
            print(f"❌ ERROR: Exception in vggish_input.waveform_to_examples: {e}")
            print("Returning zero-filled examples_batch (NumPy) to prevent crash.")
            # Fallback for examples_batch_np if any error occurs
            num_frames_fallback = int((SAMPLE_RATE * DURATION - vggish_params.STFT_WINDOW_LENGTH_SECONDS * SAMPLE_RATE) / (vggish_params.STFT_HOP_LENGTH_SECONDS * SAMPLE_RATE)) + 1
            if num_frames_fallback <= 0:
                num_frames_fallback = 1
            examples_batch_np = np.zeros((x.shape[0], num_frames_fallback, 64), dtype=np.float32)

        # --- END NEW ---

        examples_batch = torch.from_numpy(examples_batch_np).to(x.device) # Use the potentially converted/fallback numpy array

        # --- NEW: Aggressive error handling for vggish.forward ---
        try:
            embeddings = self.vggish.forward(examples_batch) # Pass the preprocessed examples
        except Exception as e:
            print(f"❌ ERROR: Exception during self.vggish.forward: {e}")
            print("Returning zero-filled embeddings to prevent crash.")
            embeddings = torch.zeros(examples_batch.shape[0], 128).to(x.device) # Fallback

        # --- END NEW ---

        # Average pool the embeddings across the time segments
        # This results in a single 128-dim embedding per audio clip
        pooled_embeddings = torch.mean(embeddings, dim=1) # Output shape: [batch_size, 128]
        return pooled_embeddings

class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use the VGGish model as our feature extractor backbone
        self.feature_extractor = VGGishFeatureExtractor()

        # Define a new classification head that takes VGGish embeddings (128 features)
        # and outputs probabilities for our specific number of classes.
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), # First linear layer
            nn.ReLU(),          # Activation function
            nn.Dropout(0.5),    # Dropout for regularization
            nn.Linear(64, num_classes) # Final linear layer to output class scores
        )

    def forward(self, x):
        # Pass the raw audio through the VGGish feature extractor
        features = self.feature_extractor(x) # Output: [batch_size, 128]

        # Pass the extracted features through the new classification head
        output = self.classifier(features)
        return output

# --- MAIN TRAINING ---
def main():
    # The global import block at the very top handles VGGish module loading.
    # No need for _ensure_vggish_modules_loaded() here anymore.

    download_training_data()

    # Load file paths and labels
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

    # --- Pre-filter problematic audio files here ---
    valid_audio_paths = []
    valid_audio_labels = []
    target_length_samples = SAMPLE_RATE * DURATION

    # Use the globally imported vggish_params for min_vggish_input_samples
    min_vggish_input_samples = SAMPLE_RATE * 0.1 # Default fallback

    if vggish_params is not None:
        min_vggish_input_samples = int(vggish_params.STFT_WINDOW_LENGTH_SECONDS * SAMPLE_RATE)
        print(f"DEBUG: Using VGGish params for pre-validation - STFT_WINDOW_LENGTH_SECONDS={vggish_params.STFT_WINDOW_LENGTH_SECONDS}, Calculated min_vggish_input_samples={min_vggish_input_samples}")
    else:
        # Fallback if vggish_params could not be loaded at startup
        print("Warning: vggish_params not loaded globally. Using default min_vggish_input_samples for pre-validation.")


    print("Pre-validating audio files...")
    for i, path in enumerate(all_audio_paths):
        try:
            # Attempt to read and resample to ensure it's a valid audio file
            wav_check, sr_check = sf.read(path)
            if wav_check is None or wav_check.size == 0: # Check for None or empty array
                print(f"Skipping empty audio file: {path}")
                continue
            
            resampled_wav_check = librosa.resample(wav_check, orig_sr=sr_check, target_sr=SAMPLE_RATE)
            
            # Check if the resampled audio is still too short for VGGish's internal framing
            if len(resampled_wav_check) < min_vggish_input_samples:
                print(f"Skipping audio file too short for VGGish framing ({len(resampled_wav_check)} samples, min needed {min_vggish_input_samples}): {path}")
                continue

            # If it passes these checks, it's considered valid for inclusion
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

    # Save class map
    with open("class_to_label.json", "w") as f:
        json.dump({str(i): label for i, label in enumerate(label_encoder.classes_)}, f)
    print("Class map saved to class_to_label.json")

    # Stratified split
    # Ensure there are enough samples for splitting (at least 2 per class for stratification)
    # Check if any class has less than 2 samples for stratification
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

    # Custom collate_fn (simplified as __getitem__ always returns valid data)
    def custom_collate_fn(batch):
        # Filter out any samples that were returned as dummy data (label -1) from VoiceDataset
        # This is a safeguard if __getitem__ failed to process a file even after pre-validation.
        filtered_batch = [item for item in batch if item[1] != -1]
        
        if not filtered_batch:
            # Return empty tensors of correct shape if batch is empty, to avoid DataLoader crash
            return torch.empty(0, SAMPLE_RATE * DURATION).float(), torch.empty(0, dtype=torch.long)
        
        audios = torch.stack([item[0] for item in filtered_batch])
        labels = torch.tensor([item[1] for item in filtered_batch])
        return audios, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    # Initialize model with the new TransferLearningCNN
    model = TransferLearningCNN(NUM_CLASSES).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Using device: {DEVICE}")

    loss_fn = nn.CrossEntropyLoss()
    # Use a slightly smaller learning rate for fine-tuning, or if only training the head
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Adjusted learning rate

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Skip empty batches from custom_collate_fn
            if x.numel() == 0: # Check if tensor is empty
                print(f"  Skipping empty batch {batch_idx+1} in training (no valid samples).")
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0: # Print loss every 10 batches
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Train Loss: {loss.item():.4f}")


        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_idx_val, (x, y) in enumerate(val_loader): # Added batch_idx_val for clarity
                # Skip empty batches from custom_collate_fn
                if x.numel() == 0: # Check if tensor is empty
                    print(f"  Skipping empty batch {batch_idx_val+1} in validation (no valid samples).")
                    continue

                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        # Ensure total is not zero to avoid division by zero
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "best_voice_model.pt")
            print("✅ Model saved (best)")

    print("🎉 Training complete")

if __name__ == "__main__":
    # Ensure shutil is imported if main is called directly for local testing
    import shutil
    main()
