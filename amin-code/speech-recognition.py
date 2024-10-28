import time
import speech_recognition as sr
import librosa
import numpy as np
from sklearn.svm import SVC
import requests
import pickle

# مرحله ۱: دریافت صدای ورودی
def get_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    return audio.get_wav_data()

# مرحله ۲: استخراج ویژگی‌های صوتی
def extract_features(audio_data):
    y, sr = librosa.load(audio_data, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# مرحله ۳: آموزش مدل
def train_model(known_audio_files):
    features = [extract_features(file) for file in known_audio_files]
    labels = [0] * len(known_audio_files)  # 0 به عنوان برچسب صاحب اصلی
    model = SVC()
    model.fit(features, labels)
    with open("voice_model.pkl", "wb") as f:
        pickle.dump(model, f)

# مرحله ۴: شناسایی صدا
def identify_voice(audio_data):
    with open("voice_model.pkl", "rb") as f:
        model = pickle.load(f)
    features = extract_features(audio_data)
    return model.predict([features])[0]

# مرحله ۵: ارسال پیام به API
def alert_api():
    requests.post("YOUR_API_ENDPOINT", json={"message": "Unauthorized voice detected"})

# کنترل طول مدت صدای غیرمجاز
def monitor_unauthorized_voice():
    start_time = None  # شروع زمان برای صدای غیرمجاز

    while True:
        audio_data = get_audio()
        if identify_voice(audio_data) != 0:
            # اگر صدای غیرمجاز شناسایی شد
            if start_time is None:
                start_time = time.time()  # شروع شمارش زمان

            # اگر مدت صدای غیرمجاز بیش از 2 ثانیه باشد
            elif time.time() - start_time >= 2:
                alert_api()  # ارسال هشدار به API
                start_time = None  # بازنشانی تایمر
        else:
            # اگر صدای مجاز شناسایی شد، تایمر را بازنشانی می‌کنیم
            start_time = None

# اجرای برنامه
monitor_unauthorized_voice()
