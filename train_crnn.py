import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# CONFIG
# ------------------------------
DATASET_PATH = r"C:/Projects/Multimodal Emotional Art Generation/data/mfcc_ravdess/"

emotion_labels = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}

MAX_LEN = 240      # MATCHES YOUR NEW MFCC SHAPE
N_MFCC = 40


# ------------------------------
# DATASET
# ------------------------------
class MFCCDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def pad(self, mfcc):
        if mfcc.shape[0] >= MAX_LEN:
            return mfcc[:MAX_LEN]
        pad = np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))
        return np.vstack([mfcc, pad])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mfcc = np.load(self.files[idx]).astype(np.float32)

        # normalize per sample
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

        mfcc = self.pad(mfcc)
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ------------------------------
# CRNN MODEL — STRONG VERSION
# ------------------------------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )

        # For MAX_LEN = 240 → after 3 pools: 240 → 120 → 60 → 30
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=160,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(160 * 2, 8)
        )

    def forward(self, x):
        x = self.cnn(x)          # (B,128,30,5)
        x = x.mean(dim=3)        # (B,128,30)
        x = x.permute(0, 2, 1)   # (B,30,128)

        out, _ = self.lstm(x)    # (B,30,320)
        out = out.mean(dim=1)    # (B,320)

        return self.fc(out)


# ------------------------------
# LOAD DATA
# ------------------------------
files, labels = [], []

for emotion, idx in emotion_labels.items():
    folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.exists(folder):
        continue

    for f in os.listdir(folder):
        if f.endswith(".npy"):
            files.append(os.path.join(folder, f))
            labels.append(idx)

print("Total samples:", len(files))

# split
X_train, X_val, y_train, y_val = train_test_split(
    files, labels, test_size=0.2, random_state=42, stratify=labels
)

train_ds = MFCCDataset(X_train, y_train)
val_ds = MFCCDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


# ------------------------------
# TRAINING SETUP
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0

print("\n Training Started...\n")

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(40):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # validation
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    acc = accuracy_score(targets, preds)

    print(f"Epoch {epoch+1}/40 | Loss: {total_loss:.3f} | Val Acc: {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "crnn_best_model.pth")
        print(f" Saved BEST model (Acc={acc:.3f})")

print("\n Training Completed! Best Accuracy:", best_acc)
