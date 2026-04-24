import torch
import numpy as np
import sounddevice as sd
import librosa
import whisper
import cv2
import time
import threading
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deepface import DeepFace
from models.speech.model import CRNN
from models.music.deam_recommend import recommend_song, play_song

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 10
MAX_LEN = 240
N_MFCC = 40
CONF_THRESHOLD = 40   # 🔥 threshold for neutral decision

# ---------------- GO EMOTIONS ----------------
GO_EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

# ---------------- MAP TEXT EMOTIONS ----------------
def map_emotion(e):
    mapping = {
        "joy":"happy",
        "sadness":"sad",
        "anger":"angry",
        "fear":"fearful",
        "surprise":"surprised",
        "neutral":"neutral"
    }
    return mapping.get(e.lower(), e.lower())

# ---------------- LOAD MODELS ----------------
print("🔊 Loading models...")

speech_model = CRNN()
speech_model.load_state_dict(torch.load("crnn_best_model.pth", map_location="cpu"))
speech_model.eval()

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
text_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

whisper_model = whisper.load_model("base")

fusion_model = pickle.load(open("fusion_model.pkl", "rb"))

print("✅ All models loaded\n")

# ---------------- GLOBAL ----------------
audio_data = None
face_frame = None

# ---------------- AUDIO ----------------
def record_audio():
    global audio_data
    print("🎤 Recording audio...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio_data = audio.flatten()
    print("✅ Audio done")

# ---------------- FACE ----------------
def capture_face():
    global face_frame
    print("📷 Capturing face...")
    cap = cv2.VideoCapture(0)

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_frame = frame.copy()
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        if time.time() - start > DURATION:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Face done")

# ---------------- MFCC ----------------
def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.vstack([mfcc, np.zeros((MAX_LEN - mfcc.shape[0], N_MFCC))])
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc.astype(np.float32)

# ---------------- SPEECH ----------------
def speech_predict(y):
    x = torch.tensor(extract_mfcc(y), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    probs = torch.softmax(speech_model(x), dim=1).detach().numpy()[0]

    emotions = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
    idx = np.argmax(probs)

    return emotions[idx], probs[idx] * 100

# ---------------- TEXT ----------------
def text_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    probs = torch.softmax(text_model(**inputs).logits, dim=1).detach().numpy()[0]
    idx = np.argmax(probs)

    emotion = GO_EMOTIONS[idx]
    emotion = map_emotion(emotion)

    return emotion, probs[idx] * 100

# ---------------- FACE ----------------
def face_predict(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0]

    emotion = result["dominant_emotion"]
    confidence = result["emotion"][emotion]

    return emotion.lower(), confidence

# ---------------- MAIN ----------------
if __name__ == "__main__":

    input("Press ENTER to start...")

    # Run in parallel
    t1 = threading.Thread(target=record_audio)
    t2 = threading.Thread(target=capture_face)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\n🧠 Processing...\n")

    # ---------------- SPEECH ----------------
    s_em, s_conf = speech_predict(audio_data)

    # ---------------- TEXT ----------------
    text = whisper_model.transcribe(audio_data)["text"]
    t_em, t_conf = text_predict(text)

    # ---------------- FACE ----------------
    f_em, f_conf = face_predict(face_frame)

    # ---------------- SMART FUSION ----------------
    emotions = [s_em, t_em, f_em]
    confidences = [s_conf, t_conf, f_conf]

    # If all weak → neutral
    if all(c < CONF_THRESHOLD for c in confidences):
        final_emotion = "neutral"
        final_conf = max(confidences)

    else:
        X = np.array([confidences])
        fusion_pred = fusion_model.predict(X)[0]

        final_emotion = emotions[fusion_pred]
        final_conf = confidences[fusion_pred]

    # ---------------- OUTPUT ----------------
    print("\n================ FINAL RESULT ================")
    print(f"📝 Text            : {text}")
    print(f"📝 Text Emotion    : {t_em} ({t_conf:.2f}%)")
    print(f"🔊 Speech Emotion  : {s_em} ({s_conf:.2f}%)")
    print(f"🙂 Face Emotion    : {f_em} ({f_conf:.2f}%)")
    print(f"🎯 Final Emotion   : {final_emotion.upper()} ({final_conf:.2f}%)")

    song_id = recommend_song(final_emotion)
    if song_id:
        play_song(song_id)