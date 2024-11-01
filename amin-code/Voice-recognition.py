### در این اپلیکیشن از ای پس ای برای دریافت صدا به صورت زنده و سپس پردازش آن برای استخراج و مقایسه ویژگی های اف ام سی سی استفاده می کنیم.
import requests
import librosa
from scipy.spatial.distance import cosine
import io
from pydub import AudioSegment

# آدرس API که داده‌های صوتی را به صورت زنده فراهم می‌کند
api_url = "https://your-api-url.com/audio_stream"

# اثر انگشت صوتی اصلی کاربر (از قبل ذخیره شده)
original_audio_fingerprint = np.load("user_audio_fingerprint.npy")

# حد آستانه برای تطبیق صدا
audio_threshold = 0.25

# تابع برای درخواست و دریافت صدا از API
def get_live_audio_from_api():
    try:
        # دریافت داده صوتی به صورت باینری
        response = requests.get(api_url, stream=True)
        if response.status_code == 200:
            audio_data = response.content
            return audio_data
        else:
            print(f"Failed to retrieve audio data, status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving audio from API: {e}")
        return None

# تابع برای تبدیل داده صوتی باینری به یک بردار ویژگی MFCC
def extract_audio_fingerprint(audio_data):
    # تبدیل باینری داده صوتی به فرمت wav
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # استخراج ویژگی‌های MFCC
    mfccs = librosa.feature.mfcc(y=samples.astype(float), sr=sample_rate, n_mfcc=13)
    audio_fingerprint = np.mean(mfccs, axis=1)  # ایجاد بردار اثر انگشت صوتی
    return audio_fingerprint

# تابع برای مقایسه اثر انگشت‌ها
def is_audio_matching(new_fingerprint, original_fingerprint, threshold=audio_threshold):
    distance = cosine(new_fingerprint, original_fingerprint)
    print(f"Calculated audio distance: {distance}")
    return distance < threshold

# تابع اصلی برای دریافت و پردازش صدا
def process_live_audio():
    audio_data = get_live_audio_from_api()
    if audio_data is not None:
        # استخراج اثر انگشت صوتی جدید
        new_audio_fingerprint = extract_audio_fingerprint(audio_data)
        
        # مقایسه اثر انگشت‌ها
        if is_audio_matching(new_audio_fingerprint, original_audio_fingerprint):
            print("The voice matches the original user.")
        else:
            print("Different voice detected!")
            # ارسال پیام به API برای هشدار
            notify_api("Alert: Different voice detected!")

# تابع برای ارسال پیام به API در صورت تشخیص صدای متفاوت
def notify_api(message):
    notification_url = 'https://your-notification-api-url.com/api/notification'
    data = {'message': message}
    try:
        response = requests.post(notification_url, json=data)
        if response.status_code == 200:
            print("Notification sent successfully.")
        else:
            print(f"Failed to notify API, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error notifying API: {e}")

# حلقه برای اجرای مداوم دریافت و پردازش صدا
while True:
    process_live_audio()
