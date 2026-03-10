import cv2
import numpy as np
import tensorflow as tf
from detectors.face_detector import FaceDetector
from utils.preprocessing import preprocess_face

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "models/emotion_cnn_fer2013.h5"

def run_emotion_detection(source="webcam"):
    model = tf.keras.models.load_model(MODEL_PATH)
    detector = FaceDetector()
    cap = cv2.VideoCapture(0) if source == "webcam" else cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for bbox in faces:
            face_img = preprocess_face(frame, bbox)
            prediction = model.predict(np.expand_dims(face_img, axis=0))[0]
            emotion = EMOTION_CLASSES[np.argmax(prediction)]

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()