# ─────────────────────────────────────────────────────────────
# Face Emotion Training Script
# Dataset: FER2013  (or any folder of 48×48 grayscale images)
#
# Expected data/face_dataset/ layout:
#   data/face_dataset/
#     happy/   *.jpg or *.png
#     sad/
#     stress/
#     neutral/
#     fatigue/
# ─────────────────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from ai.face.model import EmotionCNN

# ── Config ───────────────────────────────────────────────────
DATA_DIR     = "data/face_dataset"
MODEL_OUT    = "ai/face/model.pth"
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-3
IMG_SIZE     = 48
VAL_SPLIT    = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Augmented transforms ─────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

val_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def train():
    print(f"Training on: {DEVICE}")

    # Load dataset
    full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {full_ds.classes}")
    print(f"Train samples: {train_size}  |  Val samples: {val_size}")

    model = EmotionCNN(num_classes=len(full_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ───────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = correct / total * 100
        val_acc   = val_correct / val_total * 100
        print(f"  Loss: {train_loss/total:.4f}  Train Acc: {train_acc:.1f}%  Val Acc: {val_acc:.1f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✅ Saved best model → {MODEL_OUT}  (val acc: {val_acc:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    train()
