###این برنامه برای تشخیص فرد و درست کردن یک امضا صوتی و تصویری از فرد هست
### این برنامه از قسمت از سه قسمت 1 رودی 2 امضا تصوری 3 امضا صوتی تقسیم شده
import cv2
import os
import face_recognition
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

# 1. تابع برای ضبط ویدیو از صورت کاربر
def record_video(video_path='user_video.avi'):
    cap = cv2.VideoCapture(0)  # 0 برای دوربین پیش‌فرض
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    print("Recording started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Recording', frame)  # نمایش ویدیو در حال ضبط
            out.write(frame)  # ذخیره فریم در فایل ویدیو

            if cv2.waitKey(1) & 0xFF == ord('q'):  # توقف با فشردن 'q'
                break
        else:
            break

    cap.release()  # آزادسازی دوربین
    out.release()  # ذخیره ویدیو
    cv2.destroyAllWindows()  # بستن پنجره‌ها
    print(f"Recording saved as '{video_path}'")

# 2. تابع برای تقسیم ویدیو به فریم‌ها و تشخیص چهره
def extract_face_encodings(video_path='user_video.avi', output_frames_dir='frames'):
    # ایجاد پوشه‌ای برای ذخیره فریم‌ها
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    cap = cv2.VideoCapture(video_path)  # باز کردن ویدیو
    frame_count = 0  # شمارش فریم‌ها
    face_encodings_list = []  # لیست برای ذخیره اثر انگشت‌های چهره

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # در صورت اتمام ویدیو
            break

        frame_path = os.path.join(output_frames_dir, f'frame_{frame_count}.jpg')  # مسیر ذخیره فریم
        cv2.imwrite(frame_path, frame)  # ذخیره فریم

        # تشخیص چهره‌ها در فریم
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if face_encodings:  # اگر چهره‌ای شناسایی شد
            face_encodings_list.append(face_encodings[0])  # ذخیره اثر انگشت چهره
            print(f"Face detected in frame {frame_count}")

        frame_count += 1  # افزایش شمارش فریم

    cap.release()  # آزادسازی ویدیو
    print(f"Extracted {len(face_encodings_list)} face encodings from {frame_count} frames.")

    # ایجاد اثر انگشت دیجیتال برای چهره
    if face_encodings_list:
        face_fingerprint = np.mean(face_encodings_list, axis=0)  # میانگین‌گیری از اثر انگشت‌ها
        print("Face fingerprint created:", face_fingerprint)
        return face_fingerprint
    else:
        print("No faces were detected in the video.")
        return None

# 3. تابع برای استخراج صدا و ایجاد اثر انگشت صوتی
def extract_audio_fingerprint(video_path='user_video.avi', audio_path='user_audio.wav'):
    # استخراج صوت با استفاده از moviepy
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

    # بارگذاری فایل صوتی با librosa
    y, sr = librosa.load(audio_path)
    # استخراج ویژگی‌های MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    audio_fingerprint = np.mean(mfccs, axis=1)  # میانگین‌گیری از ویژگی‌ها

    print("Audio fingerprint created:", audio_fingerprint)
    return audio_fingerprint

# اجرای تمام مراحل
if __name__ == "__main__":
    video_path = 'user_video.avi'  # مسیر فایل ویدیویی
    audio_path = 'user_audio.wav'   # مسیر فایل صوتی
    
    # مرحله 1: ضبط ویدیو
    record_video(video_path)

    # مرحله 2: استخراج اثر انگشت چهره
    face_fingerprint = extract_face_encodings(video_path)

    # مرحله 3: استخراج اثر انگشت صوتی
    audio_fingerprint = extract_audio_fingerprint(video_path, audio_path)
