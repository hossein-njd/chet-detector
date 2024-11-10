##  در اینجا برنامه‌ای بر اساس ویژگی‌هایی که توضیح داده‌اید، نوشته‌ام. این برنامه از WebSocket برای دریافت داده‌ها و مقایسه اثر انگشت‌ها استفاده می‌کند.
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, WebSocket
from scipy.spatial.distance import cosine

app = FastAPI()

# فرض کنید این اثر انگشت چهره ذخیره‌شده است.
# در واقع، این باید از یک پایگاه داده یا فایل‌های ذخیره‌شده خوانده شود.
stored_face_fingerprint = np.random.rand(128)  # اثر انگشت چهره ذخیره‌شده (برای مثال)

# تابع برای مقایسه اثر انگشت چهره
def compare_face_fingerprints(received_face_fingerprint):
    similarity = 1 - cosine(stored_face_fingerprint, received_face_fingerprint)
    return similarity

# WebSocket برای دریافت داده‌ها از فرانت‌اند
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # دریافت داده‌ها از فرانت‌اند (اثر انگشت چهره به صورت باینری)
        data = await websocket.receive_text()  # دریافت داده‌ها از فرانت‌اند

        # فرض کنید داده‌ها به فرمت JSON ارسال می‌شوند
        try:
            import json
            data = json.loads(data)
            received_face_fingerprint = np.array(data['face_fingerprint'])
        except Exception as e:
            print(f"Error in receiving data: {e}")
            continue

        # مقایسه اثر انگشت چهره
        face_similarity = compare_face_fingerprints(received_face_fingerprint)
        if face_similarity < 0.9:
            await websocket.send_text("هشدار: اثر انگشت چهره تطابق کمتری دارد.")

        await websocket.send_text("بررسی انجام شد.")

## این برنامه هنوز تست نشده