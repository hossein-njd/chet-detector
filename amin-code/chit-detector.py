import cv2
import dlib
import numpy as np

# Load the pre-trained shape predictor model
PREDICTOR_PATH = '/home/amir-agho/programing/chet-detector/amin-code/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Function to detect head movement in 4 directions: left, right, up, and down
def detect_head_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        # Extract the coordinates of the eyes and nose
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        nose = landmarks.part(30)

        # Calculate horizontal distance (left/right)
        eye_distance_x = right_eye.x - left_eye.x
        nose_offset_x = nose.x - (left_eye.x + right_eye.x) / 2

        # Calculate vertical distance (up/down)
        eye_average_y = (left_eye.y + right_eye.y) / 2
        nose_offset_y = nose.y - eye_average_y

        # Initialize movement direction
        movement = "Head facing forward"

        # Detect horizontal movement (left/right)
        if abs(nose_offset_x) > eye_distance_x * 0.2:  # Adjust sensitivity for left/right movement
            if nose_offset_x > 0:
                movement = "Head turned left"
            else:
                movement = "Head turned right"

        # Detect vertical movement (up/down) with a neutral zone
        elif abs(nose_offset_y) > eye_distance_x * 0.55:  # Increased sensitivity threshold for up/down
            if nose_offset_y > 0:
                movement = "Head tilted down"
            else:
                movement = "Head tilted up"

        # Display the movement on the frame
        cv2.putText(frame, movement, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Start capturing video from the laptop camera
#
cap = cv2.VideoCapture(0)




while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_head_movement(frame)
    cv2.imshow("Head Movement Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
