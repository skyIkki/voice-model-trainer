#!/usr/bin/env python3
import os, json, logging, random, shutil, base64
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
import firebase_admin
from firebase_admin import credentials, storage

#  CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "best_voice_model.pt"
LABEL_FILE = "class_to_label.json"
USER_DIR = "voice_training_data"
BASE_DIR = "base_training_data"
BUCKET = "flower-identification-c2ef6.firebasestorage.app"
USER_PREFIX = "user_training_data/"
MODEL_PREFIX = ""
BATCH_SIZE, LR, EPOCHS = 32, 0.001, 15
VAL_RATIO = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
def set_seed(s=42): random.seed(s); torch.manual_seed(s); torchaudio.set_audio_backend("sox_io")
set_seed()

def init_firebase():
    key = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    if not key: raise Exception("Missing FIREBASE_SERVICE_ACCOUNT_KEY")
    cred = credentials.Certificate(json.loads(base64.b64decode(key)))
    firebase_admin.initialize_app(cred, {'storageBucket': BUCKET})
    logging.info("✅ Firebase initialized")

def download_user_audio():
    os.makedirs(USER_DIR, exist_ok=True)
    bucket = storage.bucket()
    for blob in bucket.list_blobs(prefix=USER_PREFIX):
        if blob.name.endswith("/"): continue
        rel = blob.name[len(USER_PREFIX):]
        target = os.path.join(USER_DIR, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        blob.download_to_filename(target)
        logging.debug(f"🎧 {blob.name} → {target}")

class AudioDataset(Dataset):
    def __init__(self, root):
        self.samples, self.labels = [], {}
        for speaker in sorted(os.listdir(root)):
            path = os.path.join(root, speaker)
            if os.path.isdir(path):
                idx = len(self.labels)
                self.labels[speaker] = idx
                for f in os.listdir(path):
                    if f.lower().endswith(".wav"):
                        self.samples.append((os.path.join(path, f), idx))
        logging.info(f"Loaded {len(self.samples)} samples from {root}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        wav, sr = torchaudio.load(path)
        wav = wav / wav.abs().max()
        spec = torchaudio.transforms.MelSpectrogram()(wav)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        if spec.shape[-1] < 128:
            spec = torch.nn.functional.pad(spec, (0, 128 - spec.shape[-1]))
        spec = spec[:, :, :128]
        return spec, label

def prepare_data():
    datasets = []
    for d in [BASE_DIR, USER_DIR]:
        if os.path.isdir(d): datasets.append(AudioDataset(d))
    assert datasets, "No data!"
    full = torch.utils.data.ConcatDataset(datasets)
    n = len(full); v = int(n * VAL_RATIO); t = n - v
    return random_split(full, [t, v])

def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)

def train_and_save():
    train_ds, val_ds = prepare_data()
    num_classes = len({y for _, y in train_ds})
    model = build_model(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        # -(Add training loop here)-
    # -(After training)-
    torch.jit.script(model.cpu()).save(MODEL_FILE)
    # Save class->label map
    cls_map = {i: name for name, i in train_ds.dataset.datasets[0].labels.items()}
    with open(LABEL_FILE, "w") as f: json.dump(cls_map, f)
    # Upload model + mapping + version
    bucket = storage.bucket()
    for fname in [MODEL_FILE, LABEL_FILE]:
        blob = bucket.blob(os.path.join(MODEL_PREFIX, fname))
        blob.upload_from_filename(fname)
    logging.info("🏁 Model and mapping uploaded")

if __name__ == "__main__":
    init_firebase()
    download_user_audio()
    train_and_save()
