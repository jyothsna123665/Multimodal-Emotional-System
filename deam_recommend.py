import pandas as pd
import os
import random
import pygame

# ---------------- BASE PATH ----------------
BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# ---------------- LOAD DATA ----------------
file1 = os.path.join(
    BASE,
    "data/raw/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
)

file2 = os.path.join(
    BASE,
    "data/raw/DEAM/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"
)

# Load both parts
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge
df = pd.concat([df1, df2], ignore_index=True)

# ✅ FIX: remove unwanted spaces in column names
df.columns = df.columns.str.strip()

# ✅ Rename columns safely
df = df.rename(columns={
    "song_id": "id",
    "valence_mean": "valence",
    "arousal_mean": "arousal"
})

print("✅ DEAM Loaded:", df.shape)
print("📊 Columns:", df.columns.tolist())


# ---------------- EMOTION MAP ----------------
emotion_map = {
    "happy":      (0.8, 0.7),
    "sad":        (0.2, 0.3),
    "angry":      (0.2, 0.8),
    "fearful":    (0.3, 0.7),
    "neutral":    (0.5, 0.5),
    "surprised":  (0.7, 0.8),
    "disgust":    (0.2, 0.6)
}


# ---------------- RECOMMEND FUNCTION ----------------
def recommend_song(emotion):
    if emotion not in emotion_map:
        print("⚠ Unknown emotion:", emotion)
        return None

    target_v, target_a = emotion_map[emotion]

    # Calculate distance
    df["distance"] = ((df["valence"] - target_v) ** 2 +
                      (df["arousal"] - target_a) ** 2) ** 0.5

    # Get top closest songs
    top = df.nsmallest(5, "distance")

    if len(top) == 0:
        print("❌ No matching songs found")
        return None

    song_id = int(random.choice(top["id"].values))

    return song_id


# ---------------- PLAY FUNCTION ----------------
def play_song(song_id):
    import time

    path = os.path.join(
        BASE,
        "data/raw/DEAM/DEAM_audio/MEMD_audio",
        f"{song_id}.mp3"
    )

    if not os.path.exists(path):
        print("❌ Song not found:", path)
        return

    try:
        print(f"🎵 Loading: {path}")

        pygame.mixer.init(frequency=44100, size=-16, channels=2)

        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(1.0)   # 🔊 MAX VOLUME
        pygame.mixer.music.play()

        print(f"🎵 Playing Song ID: {song_id}")

        # keep alive
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    except Exception as e:
        print("❌ Error playing song:", e)