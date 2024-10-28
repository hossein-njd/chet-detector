import cv2
from ultralytics import YOLO
import requests

# آدرس API
API_URL = 'https://sngh4sn7-3000.euw.devtunnels.ms/api/direction'  # آدرس API 
# تابع ارسال هشدار به API
def alert_api(object_name):
    payload = {'object': object_name}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print(f"Alert sent for {object_name}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert for {object_name}: {e}")

# بارگذاری مدل YOLO نسخه 8
model = YOLO('yolov8n.pt')  # مدل YOLOv8، مطمئن شوید فایل مدل در مسیر درست است

# لیست کلاس‌های مورد نظر
target_classes = ["cell phone", "paper"]

# ویدئو
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # انجام تشخیص با مدل YOLO
    results = model(frame)

    # بررسی و ارسال هشدار در صورت شناسایی اشیاء هدف
    for result in results:
        boxes = result.boxes  # جعبه‌های تشخیص داده‌شده
        for box in boxes:
            label = result.names[int(box.cls)]  # نام کلاس
            confidence = box.conf[0]  # اعتماد

            if confidence > 0.5 and label in target_classes:  # آستانه اعتماد و کلاس‌های هدف
                # ارسال هشدار به API
                alert_api(label)

                # رسم مستطیل و متن برای شیء شناسایی‌شده
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # مختصات جعبه
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # نمایش فریم
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
