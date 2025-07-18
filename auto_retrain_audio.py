import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import firebase_admin
from firebase_admin import credentials, storage
import soundfile as sf

# --- Configuration ---
FIREBASE_CRED_PATH = "firebase_key.json"
BUCKET_NAME = "your-bucket-name.appspot.com"  # Change to your Firebase bucket
DATA_DIR = "user_training_data"
MODEL_OUTPUT_PATH = "voice_model.pth"

# --- Firebase Setup ---
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
bucket = storage.bucket()

def download_data():
    blobs = bucket.list_blobs(prefix=DATA_DIR + "/")
    os.makedirs(DATA_DIR, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith(".wav"):
            local_path = os.path.join(DATA_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            print(f"Downloaded: {blob.name} → {local_path}")

# --- Audio Augmentation ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
])

# --- Dataset ---
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".wav"):
                        self.filepaths.append(os.path.join(label_path, file))
                        self.labels.append(label)

        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y, sr = librosa.load(path, sr=16000)
        y = augment(samples=y, sample_rate=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return torch.tensor(mfcc).unsqueeze(0).float(), label

# --- Simple Model ---
class VoiceModel(nn.Module):
    def __init__(self, num_classes):
        super(VoiceModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(32 * 19 * 19, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# --- Training ---
def train_and_save():
    download_data()
    dataset = AudioDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_classes = len(set(dataset.labels))
    
    model = VoiceModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_save()
