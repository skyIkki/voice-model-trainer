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

# --- NEW: Import sys and torch.hub, but remove direct vggish_input import here ---
import sys
import torch.hub

# Placeholder for the vggish_input module, will be loaded dynamically
_vggish_input_module = None
# --- END NEW ---

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
        wav, sr = sf.read(path)
        # Resample to VGGish expected sample rate (16kHz)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        # --- NEW: Robust length fixing and validation ---
        target_length_samples = SAMPLE_RATE * DURATION
        if len(wav) < target_length_samples:
            # Pad with zeros if too short
            wav = np.pad(wav, (0, target_length_samples - len(wav)), 'constant')
        elif len(wav) > target_length_samples:
            # Truncate if too long
            wav = wav[:target_length_samples]

        # Critical check: Ensure the waveform is not empty or invalid after processing
        if wav.size == 0 or np.all(wav == 0):
            # Handle cases where audio might be problematic (e.g., completely silent or too short to process)
            # For training, you might want to skip this sample or replace it with a valid one.
            # For now, we'll return a zero tensor and log a warning.
            print(f"⚠️ Warning: Processed audio for {path} is empty or all zeros. Skipping or handling gracefully.")
            # Return a zero tensor of expected shape and a dummy label to avoid crashing the DataLoader
            return torch.zeros(target_length_samples).float(), -1 # -1 can indicate invalid label, handle in training loop
        # --- END NEW ---

        if self.augment:
            wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

        wav = torch.tensor(wav).float()
        # VGGish expects [batch_size, samples] for its forward pass
        # So, we return [samples] here, and DataLoader will batch it correctly.
        return wav, label

# --- MODEL (Transfer Learning with VGGish) ---
class VGGishFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGGish model from PyTorch Hub
        # This will download the repository to the cache if not present
        self.vggish = torch.hub.load('harritaylor/pytorch-vggish', 'vggish')

        # --- Dynamically add the torchvggish path to sys.path and import vggish_input ---
        global _vggish_input_module # Access the global placeholder
        if _vggish_input_module is None: # Only import once
            # Get the path where torch.hub.load cloned the repo
            vggish_repo_path = os.path.join(torch.hub.get_dir(), 'harritaylor_pytorch-vggish_master')
            torchvggish_path = os.path.join(vggish_repo_path, 'torchvggish')
            if torchvggish_path not in sys.path:
                sys.path.insert(0, torchvggish_path) # Add to path
            try:
                # Import with alias and assign to global placeholder
                from torchvggish import vggish_input as imported_vggish_input
                _vggish_input_module = imported_vggish_input
            except ImportError as e:
                raise ImportError(f"Failed to import vggish_input from {torchvggish_path}. Error: {e}")
        # --- END NEW ---

        # Set VGGish to evaluation mode (important for consistent feature extraction)
        self.vggish.eval()

        # Freeze VGGish parameters to use it as a fixed feature extractor
        for param in self.vggish.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x is [batch_size, num_samples] (raw audio)
        # We now explicitly preprocess the waveform to VGGish examples (Mel Spectrograms)
        # This bypasses the problematic _preprocess method in vggish.py
        # _vggish_input_module.waveform_to_examples expects a 1D numpy array or a 2D tensor [batch_size, samples]
        # and returns [batch_size, num_frames, num_mels]

        # Ensure _vggish_input_module is not None before using it
        if _vggish_input_module is None:
            raise RuntimeError("vggish_input_module was not successfully loaded. Check VGGishFeatureExtractor __init__.")

        # Move tensor to CPU for numpy conversion, then back to device
        # Ensure x is contiguous before converting to numpy
        examples_batch = _vggish_input_module.waveform_to_examples(x.cpu().contiguous().numpy(), SAMPLE_RATE) # ADDED .contiguous()
        examples_batch = torch.from_numpy(examples_batch).to(x.device) # Move back to device

        # Pass the preprocessed examples through VGGish's feature extraction layers
        # The vggish model's forward method can take these examples directly
        embeddings = self.vggish.forward(examples_batch) # Pass the preprocessed examples

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
    download_training_data()

    # Load file paths and labels
    audio_paths = []
    audio_labels = []
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in files:
            if f.endswith(".wav"):
                audio_paths.append(os.path.join(root, f))
                audio_labels.append(os.path.basename(root))

    if not audio_paths:
        print("No audio files found for training. Please ensure 'user_training_data' in Firebase Storage contains .wav files in subdirectories (e.g., user_training_data/speaker1/audio.wav).")
        print("Training cannot proceed without data.")
        return

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(audio_labels)
    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)
    print(f"Found {len(audio_paths)} audio files across {NUM_CLASSES} classes.")

    # Save class map
    # Using 'json' module directly for saving
    with open("class_to_label.json", "w") as f:
        json.dump({str(i): label for i, label in enumerate(label_encoder.classes_)}, f)
    print("Class map saved to class_to_label.json")

    # Stratified split
    # Ensure there are enough samples for splitting (at least 2 per class for stratification)
    if len(audio_paths) < 2 or (NUM_CLASSES > 1 and min(np.bincount(encoded_labels)) < 2):
        print("Not enough samples per class for stratified split. Using non-stratified split or skipping validation.")
        # Fallback to non-stratified if stratification is impossible
        X_train, X_val, y_train, y_val = train_test_split(
            audio_paths, encoded_labels, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            audio_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )

    # --- NEW: Filter out potentially problematic samples from the dataset ---
    # This requires a custom collate_fn for DataLoader if you want to remove them
    # For simplicity, let's just make sure the dataset creation handles it.
    # If a sample returns -1 as label due to invalid audio, we need to filter it out.
    # A cleaner approach would be to filter X_train, X_val, y_train, y_val directly.
    # For now, we'll rely on the warning and hope the model can handle a few problematic samples
    # or that the fix_length and pad ensures valid input.
    # The primary fix is to ensure `fix_length` and padding are correct.
    # --- END NEW ---

    train_dataset = VoiceDataset(X_train, y_train, augment=True)
    val_dataset = VoiceDataset(X_val, y_val, augment=False)

    # --- NEW: Custom collate_fn to filter out problematic samples if any were marked with label -1 ---
    def custom_collate_fn(batch):
        filtered_batch = [item for item in batch if item[1] != -1] # Filter out samples with dummy label -1
        if not filtered_batch:
            # If all samples in a batch are invalid, return empty tensors or handle as error
            # For now, return empty lists, DataLoader will likely raise an error if batch is empty
            return [], []
        
        # Default collate behavior for the filtered batch
        audios = torch.stack([item[0] for item in filtered_batch])
        labels = torch.tensor([item[1] for item in filtered_batch])
        return audios, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    # --- END NEW ---


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
            # --- NEW: Skip empty batches from custom_collate_fn ---
            if len(x) == 0:
                print(f"  Skipping empty batch {batch_idx+1} in training.")
                continue
            # --- END NEW ---

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
            for x, y in val_loader:
                # --- NEW: Skip empty batches from custom_collate_fn ---
                if len(x) == 0:
                    print(f"  Skipping empty batch {batch_idx+1} in validation.")
                    continue
                # --- END NEW ---

                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
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
