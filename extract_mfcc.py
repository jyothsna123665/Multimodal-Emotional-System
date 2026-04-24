import os
import librosa
import numpy as np
from tqdm import tqdm

RAW_PATH = r"C:/Projects/Multimodal Emotional Art Generation/data/raw/RAVDESS/"
SAVE_PATH = r"C:/Projects/Multimodal Emotional Art Generation/data/mfcc_ravdess/"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

TARGET_SR = 48000
N_MFCC = 40
MAX_LEN = 240    # <-- CORRECT, prevents truncation

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=None)

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    mfcc = librosa.feature.mfcc(
        y=y, sr=TARGET_SR, 
        n_mfcc=N_MFCC, n_fft=2048, hop_length=512
    ).T

    # Normalize per file
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    # Pad/trim
    if mfcc.shape[0] >= MAX_LEN:
        mfcc = mfcc[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))
        mfcc = np.vstack([mfcc, pad])

    return mfcc.astype(np.float32)

os.makedirs(SAVE_PATH, exist_ok=True)
for emo in emotion_map.values():
    os.makedirs(os.path.join(SAVE_PATH, emo), exist_ok=True)

print("Scanning dataset...")

for root, _, files in os.walk(RAW_PATH):
    for f in tqdm(files):
        if f.endswith(".wav"):
            emo_id = f.split("-")[2]
            if emo_id not in emotion_map:
                continue

            mfcc = extract_mfcc(os.path.join(root, f))
            save_folder = os.path.join(SAVE_PATH, emotion_map[emo_id])
            np.save(os.path.join(save_folder, f.replace(".wav", ".npy")), mfcc)

print("Done.")
