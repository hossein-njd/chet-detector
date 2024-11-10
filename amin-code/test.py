# server.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        # پردازش داده‌های ویدیو یا ذخیره‌سازی آنها
        print(f"Received frame of size: {len(data)} bytes")
        # اینجا می‌توانید فریم‌ها را به پردازشگر ویدیو (مثل OpenCV) ارسال کنید

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2
import numpy as np

async def process_frame(data):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # اینجا می‌توانید فریم را پردازش کنید
    # نمایش فریم (اختیاری)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
