import cv2
import dlib
import numpy as np

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shapepred.bz2")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face
        landmarks = predictor(gray, face)
        
        # Convert the landmark (x, y) coordinates to a NumPy array
        landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])

        # Example: draw glasses overlay
        # Define the points for the left and right eyes
        left_eye_points = landmarks_array[36:42]
        right_eye_points = landmarks_array[42:48]

        # Compute the center of each eye
        left_eye_center = left_eye_points.mean(axis=0).astype("int")
        right_eye_center = right_eye_points.mean(axis=0).astype("int")

        # Draw glasses using ellipses
        cv2.line(frame, tuple(mouth_center - np.array([20, 0])), tuple(mouth_center + np.array([20, 0])), (0, 0, 0), 5)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
