import cv2
from deepface import DeepFace

def load_face_emotion_model():
    print("DeepFace: Using built-in emotion analyzer…")
    return None


def predict_face_emotion(model, frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Minimal stable DeepFace call (works on all versions)
        result = DeepFace.analyze(
            img_path=rgb,
            actions=['emotion'],
            enforce_detection=False
        )

        # Some versions return list, some dict → handle both
        if isinstance(result, list):
            return result[0]["dominant_emotion"]
        else:
            return result["dominant_emotion"]

    except Exception as e:
        print("Emotion prediction error:", e)
        return None
