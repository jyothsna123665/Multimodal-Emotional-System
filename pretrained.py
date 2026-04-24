import cv2
from deepface import DeepFace

def main():
    cap = cv2.VideoCapture(0)

    print(" DeepFace Facial Emotion Detector Running...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            emotion = result[0]['dominant_emotion']

            cv2.putText(
                frame,
                f"Emotion: {emotion.upper()}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Error:", e)

        cv2.imshow("DeepFace Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
