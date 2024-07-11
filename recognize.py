import cv2
import os
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to extract faces from an image
def extract_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

# Function to train the recognizer on known faces
def train_recognizer(data_folder_path):
    faces = []
    labels = []

    for i, person_folder in enumerate(os.listdir(data_folder_path)):
        person_folder_path = os.path.join(data_folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            for filename in os.listdir(person_folder_path):
                if filename.endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_folder_path, filename)
                    img = cv2.imread(img_path)
                    extracted_faces = extract_faces(img)
                    if extracted_faces:
                        faces.append(cv2.cvtColor(extracted_faces[0], cv2.COLOR_BGR2GRAY))
                        labels.append(i) 

    # Use LBPHFaceRecognizer (or another algorithm)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer

# Train the recognizer
recognizer = train_recognizer("known_faces")

# Function to recognize faces in a live video stream
def recognize_faces(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            label_id, confidence = recognizer.predict(roi_gray)

            if confidence < 100:
                label_text = f"Person {label_id}"
            else:
                label_text = "Unknown"

            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Facial Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start recognizing faces
recognize_faces()