from flask import Flask, request, jsonify
import cv2
import os
import face_recognition
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # پوشه برای ذخیره فایل‌های ویدیو و صوت
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. تابع برای پردازش ویدیو و استخراج اثر انگشت چهره
def extract_face_encodings(video_path):
    cap = cv2.VideoCapture(video_path)
    face_encodings_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # تشخیص چهره‌ها در فریم
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_encodings:
            face_encodings_list.append(face_encodings[0])  # ذخیره اولین چهره شناسایی‌شده

    cap.release()

    if face_encodings_list:
        face_fingerprint = np.mean(face_encodings_list, axis=0)  # میانگین اثر انگشت چهره‌ها
        return face_fingerprint.tolist()
    return None

# 2. تابع برای استخراج اثر انگشت صوتی
def extract_audio_fingerprint(audio_path):
    try:
        # بارگذاری فایل صوتی
        y, sr = librosa.load(audio_path, sr=None)  # sr=None برای حفظ نرخ نمونه‌برداری اصلی
        if len(y) == 0:
            return None, "Audio is empty"

        # بررسی کیفیت صوت (مثلاً سطح انرژی)
        energy = np.sum(y**2) / len(y)
        if energy < 1e-4:  # آستانه‌ای برای شناسایی کیفیت پایین
            return None, "Audio quality is too low"

        # استخراج ویژگی‌های MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_fingerprint = np.mean(mfccs, axis=1)  # میانگین MFCC
        return audio_fingerprint.tolist(), None
    except Exception as e:
        return None, str(e)

# 3. مسیر API برای دریافت ویدیو و پردازش
@app.route('/process_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['file']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # ذخیره فایل ویدیو
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # پردازش ویدیو
    face_fingerprint = extract_face_encodings(video_path)

    # پاک کردن فایل ویدیو
    os.remove(video_path)

    if not face_fingerprint:
        return jsonify({'error': 'face_fingerprint not found'}), 400

    # بازگشت نتایج به فرانت‌اند
    return jsonify({
        'face_fingerprint': face_fingerprint
    })

# 4. مسیر API برای پردازش فایل صوتی
@app.route('/process_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['file']
    if audio.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # ذخیره فایل صوتی
    audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(audio_path)

    # پردازش فایل صوتی
    audio_fingerprint, error = extract_audio_fingerprint(audio_path)

    # پاک کردن فایل صوتی
    os.remove(audio_path)

    if error:
        return jsonify({'error': error}), 400

    # بازگشت اثر انگشت صوتی به فرانت‌اند
    return jsonify({
        'audio_fingerprint': audio_fingerprint
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
