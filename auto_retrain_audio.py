#!/usr/bin/env python3

import os
import torch
import torchaudio
import shutil
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn.functional as F
from torch import nn, optim
import firebase_admin
from firebase_admin import credentials, storage
import json

# Constants
FIREBASE_BUCKET = 'your-project-id.appspot.com'
FIREBASE_CRED_PATH = 'firebase_key.json'
LOCAL_DATA_DIR = 'voice_training_data'
MODEL_SAVE_PATH = 'best_voice_model.pt'

# ---- Firebase Init ----
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_BUCKET
    })
bucket = storage.bucket()

# ---- Download Data from Firebase ----
def download_user_data():
    if os.path.exists(LOCAL_DATA_DIR):
        shutil.rmtree(LOCAL_DATA_DIR)
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    blobs = bucket.list_blobs(prefix='user_training_data/')
    for blob in blobs:
        if blob.name.endswith('.wav'):
            local_path = os.path.join(LOCAL_DATA_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded {blob.name} to {local_path}")

# ---- Dataset Definition ----
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.samples = []
        self.labels = {}
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                label = fname.split('_')[0]
                if label not in self.labels:
                    self.labels[label] = len(self.labels)
                self.samples.append((os.path.join(folder, fname), self.labels[label]))

        with open("class_to_label.json", "w") as f:
            json.dump(self.labels, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        waveform, sr = torchaudio.load(filepath)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        return waveform.mean(dim=0), label  # mono, shape: (time,)

# ---- Model ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 1960, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        return self.net(x)

# ---- Train ----
def train_model():
    dataset = AudioDataset(LOCAL_DATA_DIR)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleCNN(num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    for epoch in range(10):
        model.train()
        total_loss = 0
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Saved best model")

# ---- Main ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("📥 Downloading training data from Firebase...")
    download_user_data()
    
    print("🎙️ Starting training process...")
    train_model()
    
    print("✅ Training complete.")
