### websocket.py
"""
Flask-SocketIO와 WebSocket을 사용하여 ROS Bridge와 통신하고, 비전 데이터를 실시간으로 처리 및 전송.
"""

import eventlet

# 이벤트 루프와 비동기 작업을 위한 패치
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO
import json
import asyncio
import websockets
from Vision.local_vision_monitor import classify_cropped_image, get_message, update_vision_data, get_metadata

# Flask 앱 및 SocketIO 초기화
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# ROS Bridge WebSocket 서버 URI 설정
WS_SERVER_URI = "ws://192.168.58.27:9090"

# WebSocket 메시지 전송을 위한 비동기 함수
async def send_message_to_rosbridge(dripper_info, cup_info):
    """
    ROS Bridge에 WebSocket 메시지를 전송.
    - dripper_info: 드리퍼 상태 정보
    - cup_info: 컵 상태 정보
    """
    try:
        async with websockets.connect(WS_SERVER_URI) as websocket:
            message = {
                "op": "publish",
                "topic": "/vision",
                "type": "std_msgs/String",
                "msg": {
                    "data": json.dumps({
                        "dripper": dripper_info,
                        "cup": cup_info,
                        "pot": []
                    })
                }
            }
            await websocket.send(json.dumps(message))
            print(f"Sent message: {message}")
    except Exception as e:
        print(f"Exception: {e}")

# Flask-SocketIO 이벤트 핸들러
@socketio.on('data plz')
def handle_progress_update():
    """
    'data plz' 이벤트를 처리하며, 드리퍼 및 컵 상태를 전송.
    """
    print('\nSending...')
    dripper, cup = get_metadata()
    # 비동기 WebSocket 전송 함수 실행
    asyncio.run(send_message_to_rosbridge(dripper, cup))
    get_message()
    socketio.emit('vision_get')
    print("Sending done!\n")

@socketio.on('success plz')
def handle_progress_update():
    """
    'success plz' 이벤트를 처리하며, 드리퍼 성공/실패를 분류.
    """
    classify_cropped_image()

# 메인 진입점
if __name__ == '__main__':
    # 비동기 백그라운드 작업 실행
    print("[INFO] Starting Flask-SocketIO Server...")
    socketio.start_background_task(update_vision_data)
    socketio.run(app, debug=False, host='192.168.58.22', port=9999)
