## ## این برنامه برای تشخیص و مشخص کردن این که تصویر فرد هست که ورودی را از سیستم روبرو فرد میگیرد 
#
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, WebSocket
from scipy.spatial.distance import cosine
import io

app = FastAPI()

# فرض کنید این اثر انگشت چهره ذخیره‌شده است.
# در واقع، این باید از یک پایگاه داده یا فایل‌های ذخیره‌شده خوانده شود.
stored_face_fingerprint = np.random.rand(128)  # اثر انگشت چهره ذخیره‌شده (برای مثال)

# تابع برای استخراج اثر انگشت چهره از تصویر
def extract_face_fingerprint(image_data):
    # بارگذاری تصویر از داده‌های باینری
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # شناسایی چهره‌ها در تصویر
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    if len(face_encodings) > 0:
        # فرض می‌کنیم اولین چهره استخراج‌شده را استفاده کنیم
        return face_encodings[0]
    else:
        return None

# تابع برای مقایسه اثر انگشت چهره
def compare_face_fingerprints(received_face_fingerprint, extracted_face_fingerprint):
    similarity = 1 - cosine(received_face_fingerprint, extracted_face_fingerprint)
    return similarity

# WebSocket برای دریافت داده‌ها از فرانت‌اند
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # دریافت داده‌ها از فرانت‌اند (اثر انگشت چهره به صورت باینری و فایل تصویری)
        data = await websocket.receive_text()  # دریافت داده‌ها از فرانت‌اند

        # فرض کنید داده‌ها به فرمت JSON ارسال می‌شوند
        try:
            import json
            data = json.loads(data)
            received_face_fingerprint = np.array(data['face_fingerprint'])  # اثر انگشت چهره ارسال‌شده
            image_data = data['image_data']  # داده تصویری خام (base64 یا array)
        except Exception as e:
            print(f"Error in receiving data: {e}")
            continue

        # اگر داده تصویری و اثر انگشت موجود باشد
        if received_face_fingerprint is not None and image_data is not None:
            # استخراج اثر انگشت چهره از تصویر
            extracted_face_fingerprint = extract_face_fingerprint(image_data)
            
            if extracted_face_fingerprint is not None:
                # مقایسه اثر انگشت‌ها
                face_similarity = compare_face_fingerprints(received_face_fingerprint, extracted_face_fingerprint)
                
                if face_similarity < 0.8:
                    await websocket.send_text("این چهره مربوط به فرد دیگری است.")  # تطابق پایین
                else:
                    await websocket.send_text("این چهره متعلق به فرد است.")  # تطابق بالا
            else:
                await websocket.send_text("چهره‌ای در تصویر شناسایی نشد.")
        else:
            await websocket.send_text("داده‌ها ناقص هستند. لطفاً اثر انگشت و تصویر را ارسال کنید.")
        
        await websocket.send_text("بررسی انجام شد.")