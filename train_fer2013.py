import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import timm

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, val_loader, device, epochs, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch}/{epochs}")

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                pred = torch.argmax(out, dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        acc = accuracy_score(targets, preds)
        print(f"Validation Accuracy: {acc:.4f}")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f" Saved Best Model — Acc={acc:.4f}")

    print("\n Training Completed!")
    print("Best Accuracy:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the whole dataset (7 emotion folders)
    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    # Split (80 train / 20 val)
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=full_dataset.targets
    )

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = EmotionNet(num_classes=len(full_dataset.classes), pretrained=args.pretrained).to(device)

    save_path = "best_fer_efficientnet.pth"

    train(model, train_loader, val_loader, device, args.epochs, save_path)
