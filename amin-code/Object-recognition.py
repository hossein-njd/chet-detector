import cv2
import numpy as np
import requests

# بارگذاری مدل YOLO
net = cv2.dnn.readNet('yolov11.weights', 'yolov11.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# کلاس‌های مورد نظر
classes = ["phone", "notebook"]  # به لیست کلاس‌های خود اضافه کنید

# آدرس API
API_URL = "http://your_api_endpoint.com/alert"

def alert_api(object_name):
    payload = {'object': object_name}
    response = requests.post(API_URL, json=payload)
    print(f"Alert sent for {object_name}: {response.status_code}")

# ویدئو
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # پیش‌پردازش تصویر
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # تجزیه و تحلیل خروجی‌ها
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # آستانه اعتماد
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # مستطیل دور شی
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # اگر گوشی یا جزوه شناسایی شد، به API خبر بده
            if label in classes:
                alert_api(label)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
