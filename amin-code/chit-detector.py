import cv2
import dlib
import numpy as np
import requests

# Load the pre-trained shape predictor model
PREDICTOR_PATH = '/home/amir-agho/programing/chet-detector/amin-code/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

API_URL = "http://127.0.0.1:5000"  # آدرس API که می‌خواهید به آن درخواست ارسال کنید

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to send head movement data to API
def send_head_movement_to_api(direction):
    data = {'direction': direction}
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print(f"Successfully sent {direction} to API")
        else:
            print(f"Failed to send {direction} to API, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to API: {e}")

# Function to detect eye direction with increased sensitivity
def detect_eye_direction(landmarks, eye_indices):
    left_point = landmarks.part(eye_indices[0])
    right_point = landmarks.part(eye_indices[3])
    top_mid = midpoint(landmarks.part(eye_indices[1]), landmarks.part(eye_indices[2]))
    bottom_mid = midpoint(landmarks.part(eye_indices[4]), landmarks.part(eye_indices[5]))

    # Eye width and height
    eye_width = right_point.x - left_point.x
    eye_height = bottom_mid[1] - top_mid[1]

    # Eye center
    eye_center_x = (left_point.x + right_point.x) / 2
    eye_center_y = (top_mid[1] + bottom_mid[1]) / 2

    # Determine direction with finer sensitivity
    direction = "Looking forward"

    # Increased sensitivity: smaller threshold for horizontal and vertical detection
    horizontal_threshold = eye_width * 0.99  # Smaller threshold for left-right detection
    vertical_threshold = eye_height * 0.99    # Smaller threshold for up-down detection

    if abs(eye_center_x - left_point.x) > horizontal_threshold:  # Check horizontal
        if eye_center_x < (left_point.x + right_point.x) / 2:
            direction = "Looking left"
        else:
            direction = "Looking right"
    elif abs(eye_center_y - top_mid[1]) > vertical_threshold:  # Check vertical
        if eye_center_y < (top_mid[1] + bottom_mid[1]) / 2:
            direction = "Looking up"
        else:
            direction = "Looking down"

    return direction

# Function to detect head movement in 4 directions and eye directions
def detect_head_and_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extract eye direction (left and right eyes) with higher sensitivity
        left_eye_direction = detect_eye_direction(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_direction = detect_eye_direction(landmarks, [42, 43, 44, 45, 46, 47])

        # Display eye directions on the frame
        cv2.putText(frame, f"Left eye: {left_eye_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Right eye: {right_eye_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Calculate horizontal distance (left/right) for head movement
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        nose = landmarks.part(30)

        eye_distance_x = right_eye.x - left_eye.x
        nose_offset_x = nose.x - (left_eye.x + right_eye.x) / 2

        # Calculate vertical distance (up/down) for head movement
        eye_average_y = (left_eye.y + right_eye.y) / 2
        nose_offset_y = nose.y - eye_average_y

        # Initialize head movement direction
        head_movement = "Head facing forward"

        # Detect horizontal head movement (left/right)
        if abs(nose_offset_x) > eye_distance_x * 0.2:  # Adjust sensitivity for left/right movement
            if nose_offset_x > 0:
                head_movement = "Head turned left"
                send_head_movement_to_api("Head turned left")  # ارسال به API
            else:
                head_movement = "Head turned right"
                send_head_movement_to_api("Head turned right")  # ارسال به API

        # Detect vertical head movement (up/down) with a neutral zone
        elif abs(nose_offset_y) > eye_distance_x * 0.55:  # Adjust sensitivity for up/down movement
            if nose_offset_y > 0:
                head_movement = "Head tilted down"
                send_head_movement_to_api("Head tilted down")  # ارسال به API
            else:
                head_movement = "Head tilted up"
                send_head_movement_to_api("Head tilted up")  # ارسال به API

        # Display the head movement on the frame
        cv2.putText(frame, head_movement, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Start capturing video from the laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_head_and_eye_movement(frame)
    cv2.imshow("Head and Eye Movement Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()