import cv2
from deepface import DeepFace

# ----------------------------------------
# NO MODEL LOADING NEEDED
# ----------------------------------------
def load_face_emotion_model():
    print("DeepFace does not require manual model loading.")
    return None  # dummy, to keep function structure


# ----------------------------------------
# PREDICT EMOTION FROM A FRAME
# ----------------------------------------
def predict_face_emotion(_, frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            img_path=rgb,
            actions=['emotion'],
            enforce_detection=False
        )

        if result is None:
            return "No Face Detected"

        # DeepFace returns list
        if isinstance(result, list):
            result = result[0]

        if "dominant_emotion" not in result:
            return "No Face Detected"

        return result["dominant_emotion"]

    except Exception:
        return "No Face Detected"
