import requests
import sys 
url = 'https://sngh4sn7-3000.euw.devtunnels.ms/api/direction'



def send_head_movement_to_api(direction):
    data = {'direction': direction}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Successfully sent {direction} to API")
        else:
            print(f"Failed to send {direction} to API, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to API: {e}" , "chit detected")

# send_head_movement_to_api("xxxdsdsdsss");

# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Route to receive head movement data
# @app.route('/receive_head_movement', methods=['POST'])
# def receive_head_movement():
#     data = request.get_json()
#     direction = data.get('direction')
#     print(f"Received head movement: {direction}")
    
#     # در اینجا می‌توانید داده‌ها را پردازش کنید یا در برنامه دیگری استفاده کنید
#     return jsonify({"status": "success"}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# # send_data.py
# import requests

# url = 'https://sngh4sn7-3000.euw.devtunnels.ms/api/direction'

# def send_head_movement_to_api(direction):
#     data = {'direction': direction}
#     try:
#         response = requests.post(url, json=data)
#         if response.status_code == 200:
#             print(f"Successfully sent {direction} to API")
#         else:
#             print(f"Failed to send {direction} to API, status code: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error sending data to API: {e}")
