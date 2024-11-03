from flask import Flask, request, jsonify
import os
import cv2
import face_recognition
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_face_encodings(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        face_encodings_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            if face_encodings:
                face_encodings_list.append(face_encodings[0])
        
        cap.release()
        if face_encodings_list:
            return np.mean(face_encodings_list, axis=0).tolist()
        return None
    except Exception as e:
        print("Error in extracting face encodings:", e)
        return None

def extract_audio_fingerprint(video_path, audio_path='user_audio.wav'):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)

        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        audio_fingerprint = np.mean(mfccs, axis=1)
        
        os.remove(audio_path)
        return audio_fingerprint.tolist()
    except Exception as e:
        print("Error in extracting audio fingerprint:", e)
        return None

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        face_fingerprint = extract_face_encodings(file_path)
        audio_fingerprint = extract_audio_fingerprint(file_path)

        os.remove(file_path)

        output = {
            "face_fingerprint": face_fingerprint,
            "audio_fingerprint": audio_fingerprint
        }
        
        return jsonify(output), 200
    except Exception as e:
        print("Internal server error:", e)
        return jsonify({'error': 'An error occurred while processing the video'}), 500

if __name__ == '__main__':
    app.run(debug=False)
