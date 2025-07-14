#!/usr/bin/env python3
import os, json, logging, random, shutil, base64
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models
import firebase_admin
from firebase_admin import credentials, storage

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "best_voice_model.pt"
LABEL_FILE = "class_to_label.json"
USER_DIR = "voice_training_data"
BASE_DIR = "base_training_data"
USER_PREFIX = "user_training_data/"
MODEL_PREFIX = ""
BATCH_SIZE, LR, EPOCHS = 32, 0.001, 15
VAL_RATIO = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)


set_seed()

# --- FIREBASE INITIALIZATION ---
def init_firebase():
    key = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    json_key = json.loads(base64.b64decode(key))
    cred = credentials.Certificate(json_key)
    firebase_admin.initialize_app(cred, {
        "storageBucket": "voice-model-trainer-b6814.firebasestorage.app"
    })

# --- DOWNLOAD USER AUDIO ---
def download_user_audio():
    os.makedirs(USER_DIR, exist_ok=True)
    bucket = storage.bucket()
    for blob in bucket.list_blobs(prefix=USER_PREFIX):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(USER_PREFIX):]
        target = os.path.join(USER_DIR, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        blob.download_to_filename(target)
        logging.debug(f"🎧 {blob.name} → {target}")

# --- AUDIO DATASET ---
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
        logging.info(f"📂 Loaded {len(self.samples)} samples from {root}")

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


# --- PREPARE DATA ---
def prepare_data():
    datasets = []
    for d in [BASE_DIR, USER_DIR]:
        if os.path.isdir(d):
            datasets.append(AudioDataset(d))
    assert datasets, "No data!"
    full = torch.utils.data.ConcatDataset(datasets)
    n = len(full)
    v = int(n * VAL_RATIO)
    t = n - v
    return random_split(full, [t, v])


# --- MODEL ---
def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)


# --- WRAPPER FOR EXPORT ---
class VoiceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.melspec = torchaudio.transforms.MelSpectrogram(n_fft=400, hop_length=160, n_mels=128)
        self.db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav):  # wav: shape [1, num_samples]
        if wav.dim() == 2 and wav.size(0) == 1:
            spec = self.melspec(wav)
            spec = self.db(spec)
            if spec.shape[-1] < 128:
                spec = torch.nn.functional.pad(spec, (0, 128 - spec.shape[-1]))
            spec = spec[:, :, :128]
            return self.model(spec)
        else:
            raise ValueError("Expected input shape [1, num_samples]")


# --- TRAIN AND SAVE ---
def train_and_save():
    train_ds, val_ds = prepare_data()
    num_classes = len({y for _, y in train_ds})
    model = build_model(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        logging.info(f"📦 Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = crit(out, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        logging.info(f"🔍 Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_weights.pth")
            logging.info("✅ Best model updated")

    # Load best weights
    if os.path.exists("best_weights.pth"):
        model.load_state_dict(torch.load("best_weights.pth"))

    # Export with wrapper for Android
    wrapped = VoiceWrapper(model.cpu())
    torch.jit.script(wrapped).save(MODEL_FILE)
    logging.info(f"🧠 Exported wrapped model as {MODEL_FILE}")

    # Save label map
    cls_map = {i: name for name, i in train_ds.dataset.datasets[0].labels.items()}
    with open(LABEL_FILE, "w") as f:
        json.dump(cls_map, f)
    logging.info(f"📄 Saved label map to {LABEL_FILE}")

    # Upload to Firebase
    bucket = storage.bucket()
    for fname in [MODEL_FILE, LABEL_FILE]:
        blob = bucket.blob(os.path.join(MODEL_PREFIX, fname))
        blob.upload_from_filename(fname)
        logging.info(f"☁️ Uploaded {fname} to Firebase")


# --- MAIN ---
if __name__ == "__main__":
    init_firebase()
    download_user_audio()
    train_and_save()
