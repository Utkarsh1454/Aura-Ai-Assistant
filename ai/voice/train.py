# ─────────────────────────────────────────────────────────────
# Voice Emotion Training Script
# Dataset: RAVDESS / CREMA-D
#
# Filename convention expected (RAVDESS):
#   03-01-XX-...  where XX = emotion code
#   Map these to your 5 labels below.
#
# Alternatively place .wav files in emotion-named sub-folders:
#   data/voice_dataset/
#     happy/  *.wav
#     sad/
#     stress/
#     neutral/
#     fatigue/
# ─────────────────────────────────────────────────────────────
import os
import glob
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from ai.voice.model import VoiceModel
from backend.config import EMOTION_LABELS

# ── Config ───────────────────────────────────────────────────
DATA_DIR   = "data/voice_dataset"
MODEL_OUT  = "ai/voice/model.pth"
SR         = 16_000
N_MFCC     = 40
MAX_LEN    = 200       # max time frames (truncate / pad)
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────
class VoiceDataset(Dataset):
    def __init__(self, root: str):
        self.samples: list[tuple[str, int]] = []
        for label_idx, label in enumerate(EMOTION_LABELS):
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                continue
            for f in glob.glob(os.path.join(folder, "*.wav")):
                self.samples.append((f, label_idx))
        random.shuffle(self.samples)
        print(f"Found {len(self.samples)} audio files across {len(EMOTION_LABELS)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, _ = librosa.load(path, sr=SR, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC).T  # [T, 40]

        # Pad or truncate to MAX_LEN
        if mfcc.shape[0] < MAX_LEN:
            pad = np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC), dtype=np.float32)
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:MAX_LEN]

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label)


def train():
    print(f"Training on: {DEVICE}")
    ds  = VoiceDataset(DATA_DIR)
    val_size = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = VoiceModel(n_mfcc=N_MFCC, num_classes=len(EMOTION_LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for mfcc, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(mfcc)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mfcc.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += mfcc.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(mfcc).argmax(1) == labels).sum().item()
                val_total   += mfcc.size(0)

        train_acc = correct / total * 100
        val_acc   = val_correct / val_total * 100
        print(f"  Loss: {total_loss/total:.4f}  Train: {train_acc:.1f}%  Val: {val_acc:.1f}%")
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✅ Saved best model → {MODEL_OUT}")

    print(f"\nDone. Best val accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    train()
