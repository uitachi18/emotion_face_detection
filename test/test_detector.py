import cv2
from detectors.face_detector import FaceDetector
from utils.preprocessing import preprocess_face

img = cv2.imread("sample.jpg")
fd = FaceDetector()
faces = fd.detect_faces(img)

for bbox in faces:
    face_img = preprocess_face(img, bbox)
    cv2.imshow("Processed Face", face_img)
    cv2.waitKey(0)