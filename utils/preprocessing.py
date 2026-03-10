import cv2

def preprocess_face(image, bbox, target_size=(48, 48)):
    x, y, w, h = bbox
    face = image[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, target_size)
    face = face / 255.0
    return face.reshape(target_size[0], target_size[1], 1)