## این برنامه برای تشخیص و مشخص کردن این که صدای فرد هست این برنامه باید ورودی رو از گوشی بگیرد  
import numpy as np
from fastapi import FastAPI, WebSocket
from scipy.spatial.distance import cosine
import librosa
import io

app = FastAPI()

# فرض کنید این اثر انگشت صوتی ذخیره‌شده است.
# در واقع، این باید از یک پایگاه داده یا فایل‌های ذخیره‌شده خوانده شود.
stored_audio_fingerprint = np.random.rand(13)  # اثر انگشت صوتی ذخیره‌شده (برای مثال)

# تابع برای استخراج اثر انگشت صوتی از فایل صوتی
def extract_audio_fingerprint(audio_data):
    # بارگذاری فایل صوتی از داده‌های باینری
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    # استخراج ویژگی‌های MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # میانگین ویژگی‌ها برای ایجاد اثر انگشت صوتی
    fingerprint = np.mean(mfccs, axis=1)
    return fingerprint

# تابع برای مقایسه اثر انگشت‌های صوتی
def compare_audio_fingerprints(received_audio_fingerprint, file_audio_fingerprint):
    similarity = 1 - cosine(received_audio_fingerprint, file_audio_fingerprint)
    return similarity

# WebSocket برای دریافت داده‌ها از فرانت‌اند
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # دریافت داده‌ها از فرانت‌اند (اثر انگشت صوتی به صورت باینری و فایل صوتی)
        data = await websocket.receive_text()  # دریافت داده‌ها از فرانت‌اند

        try:
            import json
            data = json.loads(data)
            received_audio_fingerprint = np.array(data.get('audio_fingerprint', None))  # اثر انگشت صوتی ارسال شده
            audio_file_data = data.get('audio_file', None)  # داده صوتی خام (به صورت base64 یا array)
        except Exception as e:
            print(f"Error in receiving data: {e}")
            continue

        # اگر داده صوتی و اثر انگشت موجود باشد
        if received_audio_fingerprint is not None and audio_file_data is not None:
            # استخراج اثر انگشت صوتی از فایل صوتی
            file_audio_fingerprint = extract_audio_fingerprint(audio_file_data)
            
            # مقایسه اثر انگشت‌ها
            audio_similarity = compare_audio_fingerprints(received_audio_fingerprint, file_audio_fingerprint)
            
            if audio_similarity < 0.8:
                await websocket.send_text("این صدای فرد دیگری است.")  # تطابق پایین
            else:
                await websocket.send_text("این صدای فرد شبیه است.")  # تطابق بالا
        else:
            await websocket.send_text("داده‌ها ناقص هستند. لطفاً اثر انگشت و فایل صوتی را ارسال کنید.")
        
        await websocket.send_text("بررسی انجام شد.")
