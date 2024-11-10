##  در اینجا برنامه‌ای بر اساس ویژگی‌هایی که توضیح داده‌اید، نوشته‌ام. این برنامه از WebSocket برای دریافت داده‌ها و مقایسه اثر انگشت‌ها استفاده می‌کند.

import cv2
import face_recognition
import numpy as np
import librosa
from fastapi import FastAPI, WebSocket
from scipy.spatial.distance import cosine

app = FastAPI()

# فرض کنید اینها اثر انگشت‌های ذخیره‌شده چهره و صوتی هستند.
# در واقع، این‌ها باید از یک پایگاه داده یا فایل‌های ذخیره‌شده خوانده شوند.
stored_face_fingerprint = np.random.rand(128)  # اثر انگشت چهره ذخیره‌شده (برای مثال)
stored_audio_fingerprint = np.random.rand(13)  # اثر انگشت صوتی ذخیره‌شده (برای مثال)

# تابع برای مقایسه اثر انگشت‌های چهره
def compare_face_fingerprints(received_face_fingerprint):
    similarity = 1 - cosine(stored_face_fingerprint, received_face_fingerprint)
    return similarity

# تابع برای مقایسه اثر انگشت‌های صوتی
def compare_audio_fingerprints(received_audio_fingerprint):
    if received_audio_fingerprint is None:
        return None
    similarity = 1 - cosine(stored_audio_fingerprint, received_audio_fingerprint)
    return similarity

# WebSocket برای دریافت داده‌ها از فرانت‌اند
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # دریافت داده‌ها از فرانت‌اند (اثر انگشت چهره و صوتی به صورت باینری)
        data = await websocket.receive_text()  # دریافت داده‌ها از فرانت‌اند

        # فرض کنید داده‌ها به فرمت JSON ارسال می‌شوند
        try:
            import json
            data = json.loads(data)
            received_face_fingerprint = np.array(data['face_fingerprint'])
            received_audio_fingerprint = np.array(data.get('audio_fingerprint', None))  # None اگر صوتی نباشد
        except Exception as e:
            print(f"Error in receiving data: {e}")
            continue

        # مقایسه اثر انگشت چهره
        face_similarity = compare_face_fingerprints(received_face_fingerprint)
        if face_similarity < 0.9:
            await websocket.send_text("هشدار: اثر انگشت چهره تطابق کمتری دارد.")

        # مقایسه اثر انگشت صوتی
        if received_audio_fingerprint is None:
            await websocket.send_text("هشدار: اثر انگشت صوتی خالی است.")
        else:
            audio_similarity = compare_audio_fingerprints(received_audio_fingerprint)
            if audio_similarity < 0.7:
                await websocket.send_text("هشدار: اثر انگشت صوتی تطابق کمتری دارد.")

        # مکث کوچک برای جلوگیری از ارسال داده‌های مکرر در هر دور
        await websocket.send_text("بررسی انجام شد.")
## این برنامه هنوز تست نشده