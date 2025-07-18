import os
import json
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import firebase_admin
from firebase_admin import credentials, storage
from pathlib import Path
import random
import shutil

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIREBASE_CRED_JSON = "firebase_key.json"
FIREBASE_BUCKET_NAME = "your-firebase-project.appspot.com"  # 🔁 Replace with yours
STORAGE_FOLDER = "user_training_data"
LOCAL_DATA_DIR = "voice_training_data"
MODEL_SAVE_PATH = "best_voice_model.pt"
LABEL_MAP_PATH = "class_to_label.json"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# --- SETUP FIREBASE ---
if not os.path.exists(FIREBASE_CRED_JSON):
    with open(FIREBASE_CRED_JSON, "w") as f:
        f.write(os.environ["FIREBASE_SERVICE_ACCOUNT_KEY"])

cred = credentials.Certificate(FIREBASE_CRED_JSON)
firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})
bucket = storage.bucket()

# --- DOWNLOAD AUDIO DATA FROM FIREBASE STORAGE ---
def download_data():
    if os.path.exists(LOCAL_DATA_DIR):
        shutil.rmtree(LOCAL_DATA_DIR)
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    blobs = bucket.list_blobs(prefix=STORAGE_FOLDER + "/")
    for blob in blobs:
        if blob.name.endswith(".wav"):
            local_path = os.path.join(LOCAL_DATA_DIR, *blob.name.split("/")[1:])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded: {blob.name}")

# --- AUDIO DATASET CLASS ---
class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.label_to_class = {}

        for idx, label_dir in enumerate(sorted(os.listdir(root_dir))):
            label_path = self.root_dir / label_dir
            if not label_path.is_dir(): continue
            self.label_to_class[idx] = label_dir
            for audio_file in label_path.glob("*.wav"):
                self.samples.append((audio_file, idx))

        with open(LABEL_MAP_PATH, "w") as f:
            json.dump(self.label_to_class, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# --- SIMPLE MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- TRAINING FUNCTION ---
def train_model(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.mean(dim=1, keepdim=True)  # mono
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# --- MAIN ---
if __name__ == "__main__":
    print("⬇️ Downloading audio data...")
    download_data()

    print("🗃️ Preparing dataset...")
    transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
    dataset = VoiceDataset(LOCAL_DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🧠 Training model...")
    model = SimpleCNN(num_classes=len(dataset.label_to_class)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, loader, criterion, optimizer)

    print(f"💾 Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("✅ Done!")
