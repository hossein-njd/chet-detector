## این برنامه برای تشخیص و مشخص کردن این که صدای فرد هست 
import numpy as np
from fastapi import FastAPI, WebSocket
from scipy.spatial.distance import cosine
import librosa

app = FastAPI()

# فرض کنید این اثر انگشت صوتی ذخیره‌شده است.
# در واقع، این باید از یک پایگاه داده یا فایل‌های ذخیره‌شده خوانده شود.
stored_audio_fingerprint = np.random.rand(13)  # اثر انگشت صوتی ذخیره‌شده (برای مثال)

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
        # دریافت داده‌ها از فرانت‌اند (اثر انگشت صوتی به صورت باینری)
        data = await websocket.receive_text()  # دریافت داده‌ها از فرانت‌اند

        # فرض کنید داده‌ها به فرمت JSON ارسال می‌شوند
        try:
            import json
            data = json.loads(data)
            received_audio_fingerprint = np.array(data.get('audio_fingerprint', None))  # None اگر صوتی نباشد
        except Exception as e:
            print(f"Error in receiving data: {e}")
            continue

        # مقایسه اثر انگشت صوتی
        if received_audio_fingerprint is None:
            await websocket.send_text("هشدار: اثر انگشت صوتی خالی است.")
        else:
            audio_similarity = compare_audio_fingerprints(received_audio_fingerprint)
            if audio_similarity < 0.7:
                await websocket.send_text("هشدار: اثر انگشت صوتی تطابق کمتری دارد.")
            else:
                await websocket.send_text("اثر انگشت صوتی تطابق خوبی دارد.")

        await websocket.send_text("بررسی انجام شد.")
