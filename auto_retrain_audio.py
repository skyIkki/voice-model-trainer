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

# --- CONFIGURATION ---
BUCKET_NAME = "voice-model-trainer-b6814.firebasestorage.app"
DOWNLOAD_DIR = "user_training_data"
SAMPLE_RATE = 16000
DURATION = 3  # seconds
NUM_CLASSES = 0
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

def download_training_data():
    # Only import shutil if needed, to avoid potential import errors if not present
    try:
        import shutil
        if os.path.exists(DOWNLOAD_DIR):
            shutil.rmtree(DOWNLOAD_DIR)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except ImportError:
        print("Shutil not available, skipping directory cleanup. Ensure directory is empty.")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Mock bucket for local testing if Firebase isn't fully set up
    if 'bucket' not in globals() or bucket is None:
        print("Firebase bucket not initialized. Skipping data download.")
        return

    blobs = bucket.list_blobs(prefix="user_training_data/")
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

# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment
        # Initialize Mel Spectrogram transform
        # n_mels: number of mel bands, often 64 or 128
        # n_fft: FFT window size, often 1024 or 2048
        # hop_length: number of samples between successive frames
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048, # Increased FFT size for better frequency resolution
            hop_length=512,
            n_mels=128 # More mel bands for richer features
        )
        self.amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        wav, sr = sf.read(path)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav = librosa.util.fix_length(wav, size=SAMPLE_RATE * DURATION)

        if self.augment:
            wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

        wav = torch.tensor(wav).float()
        # torchaudio transforms expect [channels, samples]
        wav = wav.unsqueeze(0) # Add channel dimension: [1, samples]

        # Apply Mel Spectrogram transform
        mel_spec = self.mel_spectrogram_transform(wav)
        # Convert to dB scale, which is more perceptually relevant
        mel_spec_db = self.amplitude_to_db_transform(mel_spec)

        return mel_spec_db, label # Return the 2D Mel Spectrogram

# --- MODEL ---
# This CNN will now process 2D Mel Spectrograms, so it needs 2D convolutional layers.
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # Input: [batch_size, 1, n_mels, time_frames]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # Add Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduce spatial dimensions

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Pool to 1x1 across spatial dimensions
            nn.Flatten(),
            nn.Dropout(0.5), # Add Dropout for regularization
            nn.Linear(256, num_classes) # Output features from AdaptiveAvgPool2d are 256
        )

    def forward(self, x):
        return self.model(x)

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
        return

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(audio_labels)
    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)

    # Save class map
    with open("class_to_label.json", "w") as f:
        import json
        json.dump({str(i): label for i, label in enumerate(label_encoder.classes_)}, f)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        audio_paths, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )

    train_dataset = VoiceDataset(X_train, y_train, augment=True)
    val_dataset = VoiceDataset(X_val, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model with the new ImprovedCNN
    model = ImprovedCNN(NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
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
