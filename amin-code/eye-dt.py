import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# نقاط چپ و راست چشم‌ها از مدل FaceMesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream or file")

def preprocess_eye(eye_img):
    """پیش‌پردازش تصویر چشم با استفاده از فیلترها برای پیدا کردن مردمک"""
    gray = cv.cvtColor(eye_img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv.threshold(gray, 40, 255, cv.THRESH_BINARY_INV)
    return threshold

def get_eye_center(eye_points, faces):
    """تابعی برای محاسبه مرکز چشم"""
    eye_region = np.array([[faces[0][p][0], faces[0][p][1]] for p in eye_points])
    (ex, ey, ew, eh) = cv.boundingRect(eye_region)
    eye_roi = frame[ey:ey+eh, ex:ex+ew]
    eye_roi_processed = preprocess_eye(eye_roi)
    
    # پیدا کردن کانتورهای مردمک
    contours, _ = cv.findContours(eye_roi_processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    
    if contours:
        (ix, iy, iw, ih) = cv.boundingRect(contours[0])
        ix_cntr, iy_centr = ix + int(iw/2) + ex, iy + int(ih/2) + ey
        return (ix_cntr, iy_centr)
    return None

# ذخیره موقعیت مردمک‌ها برای فیلتر میانگین
previous_eye_center_x = None
previous_eye_center_y = None
threshold_movement = 10  # آستانه حرکت در پیکسل

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_img, bbox = detector.findFaces(frame)
        face_img, faces = meshdetector.findFaceMesh(frame)
        if bbox and faces:
            # مرکز چشم چپ
            left_eye_center = get_eye_center(LEFT_EYE, faces)
            # مرکز چشم راست
            right_eye_center = get_eye_center(RIGHT_EYE, faces)
            
            if left_eye_center and right_eye_center:
                # نمایش مرکز هر دو چشم
                cv.circle(frame, left_eye_center, 5, (0, 0, 255), -1)
                cv.circle(frame, right_eye_center, 5, (0, 0, 255), -1)

                # محاسبه حرکت چشم‌ها برای تعیین جهت نگاه
                left_eye_x, left_eye_y = left_eye_center
                right_eye_x, right_eye_y = right_eye_center

                eye_center_x = (left_eye_x + right_eye_x) // 2
                eye_center_y = (left_eye_y + right_eye_y) // 2

                # فیلتر حرکت‌های کوچک چشم
                if previous_eye_center_x is not None:
                    movement_x = abs(eye_center_x - previous_eye_center_x)
                    movement_y = abs(eye_center_y - previous_eye_center_y)

                    # اگر حرکت بیشتر از آستانه بود، جهت جدید محاسبه شود
                    if movement_x > threshold_movement or movement_y > threshold_movement:
                        previous_eye_center_x = eye_center_x
                        previous_eye_center_y = eye_center_y

                        frame_center_x = frame.shape[1] // 2
                        offset_x = abs(left_eye_x - right_eye_x)

                        if eye_center_x > frame_center_x + offset_x:
                            text = "right"
                        elif eye_center_x < frame_center_x - offset_x:
                            text = "left"
                        else:
                            text = "center"

                        cv.putText(frame, text, (100, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                else:
                    previous_eye_center_x = eye_center_x
                    previous_eye_center_y = eye_center_y

        cv.imshow('Eye Gaze Detection', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
