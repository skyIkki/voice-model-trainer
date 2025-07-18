#!/usr/bin/env python3

import os
import json
import logging
import random
import shutil
import base64
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
import firebase_admin
from firebase_admin import credentials, storage

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example function: download training data from Firebase Storage
def download_training_data():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="training_data/")
    os.makedirs("data", exist_ok=True)
    for blob in blobs:
        file_path = os.path.join("data", os.path.basename(blob.name))
        blob.download_to_filename(file_path)
        print(f"Downloaded: {file_path}")

# Example Dataset class
class AudioDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
        self.transform = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.files[idx])
        mel = self.transform(waveform)
        label = random.randint(0, 1)  # dummy label
        return mel, label

# Example training function
def train_model(model, dataloader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Main function
def main():
    # Initialize Firebase Admin
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_service_account.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'your-firebase-project.appspot.com'
        })

    download_training_data()
    
    dataset = AudioDataset("data")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Example model - simple classifier
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 44, 64),  # adjust dimensions according to your mel spectrogram
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    train_model(model, dataloader)

if __name__ == "__main__":
    main()
