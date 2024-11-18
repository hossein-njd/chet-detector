###این برنامه برای تشخیص فرد و درست کردن یک امضا صوتی و تصویری از فرد هست
### این برنامه از قسمت از سه قسمت 1 رودی 2 امضا تصوری 3 امضا صوتی تقسیم شده
from flask import Flask, request, jsonify
import cv2
import os
import face_recognition
import librosa
import numpy as np
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # پوشه برای ذخیره فایل‌ها
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. تابع برای استخراج صدا از ویدیو با استفاده از ffmpeg
def extract_audio_from_video(video_path, audio_path):
    try:
        command = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '2', audio_path
        ]
        subprocess.run(command, check=True)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio using ffmpeg: {e}")
        return None

# 2. تابع برای پردازش ویدیو و استخراج اثر انگشت چهره
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

# 3. تابع برای استخراج اثر انگشت صوتی
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

# 4. مسیر API برای دریافت ویدیو و پردازش (استخراج صدا و تصویر)
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

    # استخراج صدا از ویدیو
    audio_path = os.path.join(UPLOAD_FOLDER, 'extracted_audio.wav')
    audio_file = extract_audio_from_video(video_path, audio_path)

    if audio_file is None:
        os.remove(video_path)
        return jsonify({'error': 'Failed to extract audio from video'}), 400

    # پردازش اثر انگشت صوتی
    audio_fingerprint, error = extract_audio_fingerprint(audio_path)
    if error:
        os.remove(video_path)
        os.remove(audio_path)
        return jsonify({'error': error}), 400

    # پردازش اثر انگشت تصویری
    face_fingerprint = extract_face_encodings(video_path)
    if not face_fingerprint:
        os.remove(video_path)
        os.remove(audio_path)
        return jsonify({'error': 'No face fingerprint found in the video'}), 400

    # پاک کردن فایل‌های موقتی
    os.remove(video_path)
    os.remove(audio_path)

    # بازگشت نتایج به فرانت‌اند
    return jsonify({
        'face_fingerprint': face_fingerprint,
        'audio_fingerprint': audio_fingerprint,
        'message': 'Successfully extracted face and audio fingerprints.'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
