import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

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
            nn.Linear(160*2, 8)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=3)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out.mean(dim=1)

        return self.fc(out)