from flask import Flask, request, jsonify
import os
import cv2
import face_recognition
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. تابع استخراج اثر انگشت تصویری
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

# 2. مقایسه اثر انگشت‌ها
def compare_fingerprints(fingerprint1, fingerprint2):
    similarity = 1 - cosine(fingerprint1, fingerprint2)
    return similarity * 100

@app.route('/face_identifier', methods=['POST'])
def upload_video_and_fingerprint():
    if 'video' not in request.files or 'fingerprint' not in request.form:
        print("Missing video or fingerprint data")
        return jsonify({"error": "Missing video or fingerprint data"}), 400

    video = request.files['video']
    fingerprint_str = request.form['fingerprint']

    try:
        fingerprint = list(map(float, fingerprint_str.split(',')))
    except ValueError:
        print("Invalid fingerprint format")
        return jsonify({"error": "Invalid fingerprint format"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    try:
        # استخراج اثر انگشت تصویری
        extracted_fingerprint = extract_face_encodings(video_path)
        if not extracted_fingerprint:
            print("No face fingerprint found in the video")
            return jsonify({"error": "No face fingerprint found in the video"}), 400

        # مقایسه اثر انگشت‌ها
        similarity = compare_fingerprints(fingerprint, extracted_fingerprint)

        # ارسال پیام مناسب
        if similarity > 90:
            message = "تصویر تطابق دارد"
        else:
            message = f"تصویر تطابق ندارد (میزان تطابق: {similarity:.2f}%)"

        result = {
            "message": message,
            "similarity": similarity
        }
        return jsonify(result)
    finally:
        # حذف فایل ویدیو پس از پردازش
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    app.run(debug=True)
