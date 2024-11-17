##  این برنامه درست شده تا فیلمی که از سمت فرانت به صورت زنده دارد فرستاده میشود
#  برسی کند و بفهمد که در تصویر فقط یک نفر باشد و در تصور وایت برد و گوشی و کاغذ نباشد
from fastapi import FastAPI, WebSocket
import cv2
from ultralytics import YOLO
import asyncio

app = FastAPI()
model = YOLO("yolov8n.pt")  # از مدل پایه YOLOv8 استفاده می‌کنیم

# تابع پردازش ویدیو
async def process_frame(frame):
    results = model(frame)
    
    # دریافت لیست برچسب‌ها از نتایج
    labels = [model.names[int(box[5])] for box in results[0].boxes.data]
    
    # بررسی وجود بیش از یک نفر
    people_count = labels.count('person')
    alert_people = people_count > 1

    # بررسی وجود اشیاء خاص
    alert_phone = 'cell phone' in labels
    alert_whiteboard = 'whiteboard' in labels
    alert_paper = 'paper' in labels

    return alert_people, alert_phone, alert_whiteboard, alert_paper

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        data = await websocket.receive_bytes()  # دریافت فریم ویدیویی از فرانت‌اند
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)  # تبدیل بایت‌ها به تصویر OpenCV

        # پردازش فریم و استخراج اطلاعات
        alert_people, alert_phone, alert_whiteboard, alert_paper = await process_frame(frame)

        # ارسال پیام‌ها به فرانت‌اند بر اساس شرایط مورد نظر
        if alert_people:
            await websocket.send_text("هشدار: بیش از یک نفر در تصویر وجود دارد")
        if alert_phone:
            await websocket.send_text("هشدار: گوشی در تصویر شناسایی شد")
        if alert_whiteboard:
            await websocket.send_text("هشدار: وایت‌برد در تصویر شناسایی شد")
        if alert_paper:
            await websocket.send_text("هشدار: کاغذ پر شده در تصویر شناسایی شد")

        await asyncio.sleep(0.1)  # تنظیم فاصله زمانی بین پردازش‌ها

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# این برنامه هنوز تست نشده