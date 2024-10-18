from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api', methods=['POST'])
def api():
    # data = request.get_json()
    print(request.get_data())
    # socketio.emit('update', "data")  # ارسال اطلاعات بلادرنگ به همه کلاینت‌ها
    # print(data)
    return jsonify({"message": "Data received successfully!"}), 200

@socketio.on('connect')
def handle_connect():
    emit('my response', {'data': 'Connected'})

if __name__ == '__main__':
    socketio.run(app, port=5000)
