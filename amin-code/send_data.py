import requests

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
        print(f"Error sending data to API: {e}")

send_head_movement_to_api("Alireza");