# ... (keep all your existing imports unchanged)
# No changes here

# --- CONFIGURATION ---
BUCKET_NAME = "voice-model-trainer-b6814.firebasestorage.app"
DOWNLOAD_DIR = "user_training_data"
SAMPLE_RATE = 16000
DURATION = 3
MIN_REQUIRED_SAMPLES = 15360  # For VGGish: 0.96s * 16000
NUM_CLASSES = 0
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (audio augmentation, firebase setup, and download_training_data)
# No changes needed in those sections

# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        target_length_samples = SAMPLE_RATE * DURATION
        wav = np.zeros(target_length_samples, dtype=np.float32)

        try:
            read_wav, sr = sf.read(path)
            if read_wav.size == 0:
                print(f"⚠️ Empty file: {path}")
            else:
                resampled_wav = librosa.resample(read_wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                if len(resampled_wav) < target_length_samples:
                    wav = np.pad(resampled_wav, (0, target_length_samples - len(resampled_wav)), 'constant')
                else:
                    wav = resampled_wav[:target_length_samples]

            if wav.size < MIN_REQUIRED_SAMPLES:
                print(f"⚠️ Skipping short audio (< 15360 samples): {path}")
                return torch.zeros(0).float(), -1

            if self.augment:
                wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

            return torch.tensor(wav).float(), label

        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
            return torch.zeros(0).float(), -1

# --- MODEL ---
class VGGishFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggish = torch.hub.load('harritaylor/pytorch-vggish', 'vggish')
        global _vggish_input_module
        if _vggish_input_module is None:
            vggish_repo_path = os.path.join(torch.hub.get_dir(), 'harritaylor_pytorch-vggish_master')
            torchvggish_path = os.path.join(vggish_repo_path, 'torchvggish')
            sys.path.insert(0, torchvggish_path)
            from torchvggish import vggish_input as imported_vggish_input
            _vggish_input_module = imported_vggish_input

        self.vggish.eval()
        for param in self.vggish.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.numel() == 0 or x.shape[1] < MIN_REQUIRED_SAMPLES:
            print("⚠️ Skipping VGGish forward due to insufficient input length.")
            return torch.zeros(x.shape[0], 128).to(x.device)

        try:
            x_np = x.cpu().contiguous().numpy()
            examples_batch = _vggish_input_module.waveform_to_examples(x_np, SAMPLE_RATE)
            examples_batch = torch.from_numpy(examples_batch).to(x.device)
            embeddings = self.vggish.forward(examples_batch)
            return torch.mean(embeddings, dim=1)
        except Exception as e:
            print(f"❌ Error in VGGish forward: {e}")
            return torch.zeros(x.shape[0], 128).to(x.device)

class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = VGGishFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# --- MAIN ---
def main():
    download_training_data()

    # Collect files
    all_audio_paths, all_audio_labels = [], []
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in files:
            if f.endswith(".wav"):
                all_audio_paths.append(os.path.join(root, f))
                all_audio_labels.append(os.path.basename(root))

    if not all_audio_paths:
        print("⚠️ No .wav files found.")
        return

    valid_audio_paths, valid_audio_labels = [], []
    target_length_samples = SAMPLE_RATE * DURATION

    print("🔎 Validating audio files...")
    for i, path in enumerate(all_audio_paths):
        try:
            wav, sr = sf.read(path)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(wav) < target_length_samples:
                wav = np.pad(wav, (0, target_length_samples - len(wav)), 'constant')
            else:
                wav = wav[:target_length_samples]
            if wav.size >= MIN_REQUIRED_SAMPLES and not np.all(wav == 0):
                valid_audio_paths.append(path)
                valid_audio_labels.append(all_audio_labels[i])
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not valid_audio_paths:
        print("❌ No valid audio files after validation.")
        return

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(valid_audio_labels)
    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)

    with open("class_to_label.json", "w") as f:
        json.dump({str(i): label for i, label in enumerate(label_encoder.classes_)}, f)

    counts = np.bincount(encoded_labels)
    can_stratify = all(c >= 2 for c in counts)
    if not can_stratify or len(valid_audio_paths) < 2:
        stratify = None
    else:
        stratify = encoded_labels

    X_train, X_val, y_train, y_val = train_test_split(
        valid_audio_paths, encoded_labels, test_size=0.2, stratify=stratify, random_state=42
    )

    train_dataset = VoiceDataset(X_train, y_train, augment=True)
    val_dataset = VoiceDataset(X_val, y_val, augment=False)

    def custom_collate_fn(batch):
        filtered = [b for b in batch if b[1] != -1 and b[0].numel() >= MIN_REQUIRED_SAMPLES]
        if not filtered:
            return torch.empty(0, target_length_samples), torch.empty(0, dtype=torch.long)
        audios = torch.stack([b[0] for b in filtered])
        labels = torch.tensor([b[1] for b in filtered])
        return audios, labels

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    model = TransferLearningCNN(NUM_CLASSES).to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if x.numel() == 0:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                if x.numel() == 0:
                    continue
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} - Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "best_voice_model.pt")
            print("✅ Model saved.")

    print("🎉 Training complete.")

if __name__ == "__main__":
    import shutil
    main()
