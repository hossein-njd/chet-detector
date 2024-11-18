from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        # پردازش ویدیو و استخراج اثر انگشت
        extracted_fingerprint = extract_fingerprint_from_video(video_path)

        # مقایسه اثر انگشت‌ها
        similarity = compare_fingerprints(fingerprint, extracted_fingerprint)

        result = {
            "message": "Fingerprint match confirmed" if similarity > 85 else "Fingerprint does not match",
            "similarity": similarity
        }
        return jsonify(result)
    finally:
        # حذف فایل ویدیو پس از پردازش
        if os.path.exists(video_path):
            os.remove(video_path)

def extract_fingerprint_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    fingerprint = process_video_to_fingerprint(frames)
    return fingerprint

def process_video_to_fingerprint(frames: list):
    fingerprint = np.random.rand(128).tolist()  # اثر انگشت تصادفی برای آزمایش
    return fingerprint

def compare_fingerprints(fingerprint1: list, fingerprint2: list) -> float:
    similarity = 1 - cosine(fingerprint1, fingerprint2)
    return similarity * 100

if __name__ == "__main__":
    app.run(debug=True)
