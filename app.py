import streamlit as st
import numpy as np
import torch
import cv2
import sounddevice as sd
import librosa
import os

# ---------------- PAGE ----------------
st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("🎭 Multimodal Emotion Detection System")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    from models.speech.model import CRNN
    from models.music.deam_recommend import recommend_song

    model = CRNN()
    model.load_state_dict(torch.load("crnn_best_model.pth", map_location="cpu"))
    model.eval()

    return model, recommend_song

speech_model, recommend_song = load_models()

# ---------------- STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = None

# ---------------- BUTTONS ----------------
st.subheader("Choose Mode 👇")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    if st.button("📝 Text"):
        st.session_state.mode = "text"

with col2:
    if st.button("📷 Image"):
        st.session_state.mode = "image"

with col3:
    if st.button("🎤 Speech"):
        st.session_state.mode = "speech"

with col4:
    if st.button("🎭 Multimodal"):
        st.session_state.mode = "multi"


# ---------------- COMMON FUNCTIONS ----------------

def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40).T

    if mfcc.shape[0] >= 240:
        mfcc = mfcc[:240]
    else:
        pad = np.zeros((240 - mfcc.shape[0], 40))
        mfcc = np.vstack([mfcc, pad])

    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    return mfcc


def speech_predict():
    st.info("🎤 Recording (3 sec)...")
    audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
    sd.wait()

    mfcc = extract_mfcc(audio.flatten())
    x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(speech_model(x), dim=1).numpy()[0]

    labels = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
    idx = np.argmax(probs)

    return labels[idx], probs[idx]


def text_predict(text):
    if not text.strip():
        return "neutral", 0.99

    text = text.lower()

    if "happy" in text:
        return "happy", 0.9
    elif "sad" in text:
        return "sad", 0.9
    elif "angry" in text:
        return "angry", 0.9
    else:
        return "neutral", 0.8


def face_predict():
    st.info("📷 Capturing face...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "No face", 0

    return "neutral", 0.9


def fuse(t, s, f):
    emotions = [t, s, f]

    if all(e == "neutral" for e in emotions):
        return "neutral"

    return max(set(emotions), key=emotions.count)


# ---------------- MUSIC PLAYER ----------------
def play_music(song_id):
    path = f"data/raw/DEAM/DEAM_audio/MEMD_audio/{song_id}.mp3"

    if os.path.exists(path):
        st.success(f"🎵 Song ID: {song_id}")

        audio_file = open(path, "rb")
        audio_bytes = audio_file.read()

        # 🎧 Streamlit player (with controls)
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.error("Song not found")


# ---------------- MODES ----------------

# -------- TEXT --------
if st.session_state.mode == "text":
    st.subheader("📝 Text Emotion")

    text = st.text_input("Enter text")

    if st.button("Analyze Text"):
        em, conf = text_predict(text)
        st.success(f"Emotion: {em} ({conf:.2f})")


# -------- IMAGE --------
elif st.session_state.mode == "image":
    st.subheader("📷 Image Emotion")

    if st.button("Capture Face"):
        em, conf = face_predict()
        st.success(f"Emotion: {em} ({conf:.2f})")


# -------- SPEECH --------
elif st.session_state.mode == "speech":
    st.subheader("🎤 Speech Emotion")

    if st.button("Record & Analyze"):
        em, conf = speech_predict()
        st.success(f"Emotion: {em} ({conf:.2f})")


# -------- MULTIMODAL --------
elif st.session_state.mode == "multi":
    st.subheader("🎭 Multimodal Emotion")

    text = st.text_input("Enter text (optional)")

    if st.button("Run Full Detection"):

        t_em, _ = text_predict(text)
        s_em, _ = speech_predict()
        f_em, _ = face_predict()

        final = fuse(t_em, s_em, f_em)

        st.subheader("📊 Results")
        st.write(f"📝 Text: {t_em}")
        st.write(f"🎤 Speech: {s_em}")
        st.write(f"📷 Face: {f_em}")

        st.success(f"🎯 Final Emotion: {final}")

        # 🎵 Play music
        song_id = recommend_song(final)
        if song_id:
            play_music(song_id)