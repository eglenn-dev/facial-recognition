import cv2  # Import OpenCV library for computer vision tasks
import os   # Import os module for interacting with the operating system
import numpy as np  # Import NumPy library for numerical operations

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to extract faces from an image
def extract_faces(image):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Return a list of cropped face images
    return [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

# Function to train the face recognizer
def train_recognizer(data_folder_path):
    # Initialize empty lists to store face images and corresponding labels
    faces = []
    labels = []

    # Iterate over each person's folder in the data folder
    for i, person_folder in enumerate(os.listdir(data_folder_path)):
        person_folder_path = os.path.join(data_folder_path, person_folder)
        # Check if the item is a directory
        if os.path.isdir(person_folder_path):
            # Iterate over each image file in the person's folder
            for filename in os.listdir(person_folder_path):
                # Check if the file is an image (JPEG or PNG)
                if filename.endswith(('.jpg', '.png')):
                    # Construct the full path to the image
                    img_path = os.path.join(person_folder_path, filename)
                    # Read the image using OpenCV
                    img = cv2.imread(img_path)
                    # Extract faces from the image
                    extracted_faces = extract_faces(img)
                    # If at least one face is detected
                    if extracted_faces:
                        # Convert the first extracted face to grayscale and append it to the faces list
                        faces.append(cv2.cvtColor(extracted_faces[0], cv2.COLOR_BGR2GRAY))
                        # Append the corresponding label (person index) to the labels list
                        labels.append(i) 

    # Create an LBPH face recognizer object
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Train the recognizer using the collected face images and labels
    recognizer.train(faces, np.array(labels))
    # Return the trained recognizer
    return recognizer

# Train the face recognizer using images from the "known_faces" folder
recognizer = train_recognizer("known_faces")

# Function to recognize faces in a video stream
def recognize_faces(video_source=0):
    # Open the default camera (or specified video source)
    cap = cv2.VideoCapture(video_source)
    # Initialize the name variable
    name = ""

    # Continuously capture frames from the video stream
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame using the Haar Cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # For each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face on the original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Extract the region of interest (ROI) corresponding to the face in grayscale
            roi_gray = gray[y:y+h, x:x+w]

            # Recognize the face using the trained recognizer
            label_id, confidence = recognizer.predict(roi_gray)

            # Map the label ID to a person's name
            if label_id == 0:
                name = "Andree"
            elif label_id == 1:
                name = "Ethan"
            else: 
                name = "Unspecified"
            # Determine the label text based on confidence
            if confidence < 100:
                label_text = f"Person {name}"
            else:
                label_text = "Unknown"

            # Display the label text (person's name or "Unknown") above the bounding box
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with face recognition results
        cv2.imshow('Facial Recognition', frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Execute the recognize_faces() function if the script is run directly
if __name__ == "__main__":
    recognize_faces()