import os
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn

# ---------------------------------
# CONFIG — MUST MATCH TRAINING
# ---------------------------------
MODEL_PATH = r"C:/Projects/Multimodal Emotional Art Generation/models/speech/crnn_best_model.pth"

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

MAX_LEN = 240
N_MFCC = 40

DEVICE = "cpu"

EMOTION_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# ---------------------------------
# MODEL (same as training)
# ---------------------------------
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
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)

# ---------------------------------
# AUDIO → MFCC
# ---------------------------------
def extract_mfcc(y, sr):
    # resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # MFCC shape: (time, 40)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=256
    ).T

    # normalize same as training
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

    # pad/trim
    if mfcc.shape[0] >= MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])

    return mfcc.astype(np.float32)

# ---------------------------------
# RECORD
# ---------------------------------
def record_audio():
    print(f"Recording {RECORD_SECONDS}s... Speak now!")
    audio = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    print("✔ Recorded")
    return audio

# ---------------------------------
# PREDICT
# ---------------------------------
def predict(model, mfcc):
    x = torch.tensor(mfcc).float().unsqueeze(0).unsqueeze(0)  # (1,1,240,40)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    emotion_id = int(np.argmax(probs))
    return EMOTION_MAP[emotion_id], float(probs[emotion_id]), probs

# ---------------------------------
# MAIN LOOP
# ---------------------------------
def main():
    print("Loading model...")
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(" Model loaded.\n")

    print("Real-time Emotion Recognition Ready!")
    print("Press ENTER to record, or type q + ENTER to quit.\n")

    while True:
        cmd = input("Press ENTER to speak, q to quit: ").strip().lower()
        if cmd == "q":
            break

        audio = record_audio()
        mfcc = extract_mfcc(audio, SAMPLE_RATE)

        emotion, conf, probs = predict(model, mfcc)

        # formatted probs
        prob_str = ", ".join([f"{EMOTION_MAP[i]}:{probs[i]:.2f}" for i in range(8)])

        print(f"\n Emotion: {emotion.upper()} (conf {conf:.3f})")
        print("Probabilities:", prob_str, "\n")

if __name__ == "__main__":
    main()
