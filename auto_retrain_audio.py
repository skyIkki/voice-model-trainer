import os
import torch
import torchaudio
import logging
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    USE_AUGMENT = True
except ImportError:
    USE_AUGMENT = False

# --- CONFIGURATION ---
DATA_DIR = "voice_training_data"
MODEL_PATH = "models/audio_model.pt"
SAMPLE_RATE = 16000
NUM_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 0.001

# --- LOGGER ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- AUDIO AUGMENTATION ---
if USE_AUGMENT:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3)
    ])
else:
    augment = None

# --- DATASET ---
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.label_map = {}
        for idx, label in enumerate(sorted(os.listdir(root_dir))):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                self.label_map[label] = idx
                for file in os.listdir(label_dir):
                    if file.endswith(".wav"):
                        self.samples.append(os.path.join(label_dir, file))
                        self.labels.append(idx)
        logging.info(f"📂 Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.samples[idx])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        if augment:
            waveform = torch.tensor(
                augment(samples=waveform.numpy(), sample_rate=SAMPLE_RATE)
            )

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=64
        )(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        return mel_spec, self.labels[idx]

# --- MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# --- TRAINING ---
def train_and_save():
    dataset = AudioDataset(DATA_DIR)
    if len(dataset) == 0:
        logging.warning("❌ No audio samples found.")
        return

    num_classes = len(set(dataset.labels))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = SimpleCNN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()

        acc = 100.0 * correct / len(train_data)
        logging.info(f"📦 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss:.4f} - Acc: {acc:.2f}%")

        # Validate
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            logging.info(f"✅ Validation Loss: {val_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"💾 Model saved to {MODEL_PATH}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    train_and_save()
