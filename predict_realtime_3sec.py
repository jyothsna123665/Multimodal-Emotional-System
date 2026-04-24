import cv2
import time
from models.image.face_model_3sec import load_face_emotion_model, predict_face_emotion

def run_realtime_face_emotion():
    print("Opening webcam for 3 seconds...")
    model = load_face_emotion_model()  # returns None (DeepFace loads internally)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    start = time.time()
    frame_captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        frame_captured = frame.copy()

        cv2.imshow("Face Emotion (3 sec)", frame)
        cv2.waitKey(1)

        if time.time() - start >= 3:
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_captured is None:
        print("❌ No valid frame for prediction")
        return

    print("🧠 Predicting emotion...")
    emotion = predict_face_emotion(model, frame_captured)

    if emotion == "No Face Detected":
        print("\n⚠️ No Face Detected")
    else:
        print("\n🎯 Detected Emotion:", emotion)


if __name__ == "__main__":
    run_realtime_face_emotion()
