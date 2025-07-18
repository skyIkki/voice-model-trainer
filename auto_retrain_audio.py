import os
import firebase_admin
from firebase_admin import credentials, storage
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.model_selection import train_test_split

# --- CONFIG ---
BUCKET_NAME = 'your-bucket-name.appspot.com'  # Change this to your actual bucket name
LOCAL_DATA_DIR = 'voice_training_data'
FOLDER_IN_FIREBASE = 'user_training_data'

# --- FIREBASE SETUP ---
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": os.getenv('FIREBASE_PROJECT_ID'),
    "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
    "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
    "client_id": os.getenv('FIREBASE_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
})
firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
bucket = storage.bucket()

# --- DOWNLOAD USER AUDIO ---
def download_training_data():
    print("📥 Downloading user training data from Firebase...")
    blobs = bucket.list_blobs(prefix=FOLDER_IN_FIREBASE + '/')
    for blob in blobs:
        if not blob.name.endswith('/'):
            local_path = os.path.join(LOCAL_DATA_DIR, os.path.relpath(blob.name, FOLDER_IN_FIREBASE))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"✅ Downloaded: {blob.name}")
    print("✅ All user audio downloaded.")

# --- AUDIO DATASET CLASS ---
class AudioDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform.mean(dim=0), self.labels[idx]  # Mono

# --- SIMPLE MODEL ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- FEATURE EXTRACTION ---
def extract_features(waveform):
    return torch.mean(waveform, dim=1)

# --- TRAINING FUNCTION ---
def train_model(train_loader, test_loader, input_size, num_classes):
    model = SimpleClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        for data, labels in train_loader:
            data = extract_features(data).view(data.size(0), -1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = extract_features(data).view(data.size(0), -1)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {acc:.2f}%")
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), "best_voice_model.pt")
            print("💾 Best model saved.")

# --- MAIN ---
if __name__ == "__main__":
    download_training_data()

    # Collect all files and labels
    print("🗂️ Preparing dataset...")
    files = []
    labels = []
    label_map = {}
    current_label = 0
    for root, dirs, filenames in os.walk(LOCAL_DATA_DIR):
        for f in filenames:
            if f.endswith('.wav'):
                label_name = os.path.basename(root)
                if label_name not in label_map:
                    label_map[label_name] = current_label
                    current_label += 1
                files.append(os.path.join(root, f))
                labels.append(label_map[label_name])

    with open("class_to_label.json", "w") as f:
        import json
        json.dump(label_map, f)

    # Shuffle and split
    combined = list(zip(files, labels))
    random.shuffle(combined)
    files[:], labels[:] = zip(*combined)
    train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=0.2)

    train_dataset = AudioDataset(train_files, train_labels)
    test_dataset = AudioDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print("🚀 Starting training...")
    dummy_input_size = 1  # Placeholder for input shape; we'll use raw waveform mean
    train_model(train_loader, test_loader, input_size=dummy_input_size, num_classes=len(label_map))
    print("✅ Training complete!")
