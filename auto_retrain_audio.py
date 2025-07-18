import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials, storage

# --- CONFIG ---
AUDIO_DIR = "data/audio"  # Replace with your path
MODEL_PATH = "model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
UPLOAD_TO_FIREBASE = True  # Set to False to skip Firebase upload

# --- FIREBASE INIT ---
if UPLOAD_TO_FIREBASE:
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'your-firebase-app.appspot.com'
        })
    bucket = storage.bucket()

# --- AUDIO AUGMENTATION ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# --- CUSTOM DATASET ---
class AudioDataset(Dataset):
    def __init__(self, file_list, labels, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0).numpy()

        if self.augment:
            waveform = augment(waveform, sample_rate=SAMPLE_RATE)

        waveform = torch.tensor(waveform, dtype=torch.float).unsqueeze(0)  # [1, T]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label

# --- SIMPLE MODEL ---
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 7998, 64)  # depends on input size
        self.fc2 = nn.Linear(64, 10)  # adjust output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- LOAD DATA FILES ---
def load_audio_files(audio_dir):
    files, labels = [], []
    for folder in os.listdir(audio_dir):
        folder_path = os.path.join(audio_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    files.append(os.path.join(folder_path, file))
                    labels.append(int(folder))  # folder name as label
    return files, labels

def train():
    print("[INFO] Loading dataset...")
    files, labels = load_audio_files(AUDIO_DIR)
    x_train, x_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, random_state=42)

    train_dataset = AudioDataset(x_train, y_train, augment=True)
    test_dataset = AudioDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AudioClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    if UPLOAD_TO_FIREBASE:
        blob = bucket.blob(f"models/{os.path.basename(MODEL_PATH)}")
        blob.upload_from_filename(MODEL_PATH)
        print(f"[INFO] Uploaded model to Firebase: {blob.public_url}")

if __name__ == "__main__":
    train()
