## 개요

이 프로젝트는 다음 두 가지 주요 구성 요소로 이루어져 있습니다:

1. **`local_vision_monitor.py`**: 통신 없이 로컬에서 객체 탐지 및 모니터링을 처리.
2. **`vision_ros_bridge.py`**: 로컬 비전 시스템과 ROS 브릿지 서버 간 통신을 관리하며, 객체 탐지 작업을 백그라운드에서 처리.

---

## 파일 설명

### 1. `local_vision_monitor.py`

- **목적**: Azure Kinect 카메라와 YOLO 모델을 사용하여 통신 없이 독립적으로 실시간 객체 탐지 및 모니터링을 제공.
- **주요 기능**:
    - `update_vision_data()`: 카메라에서 데이터를 지속적으로 캡처하고, YOLO를 사용하여 처리하며 컵과 드리퍼의 상태를 업데이트.
    - 실시간 객체 탐지 결과를 로컬 화면에 시각화.
- **사용 사례**: 외부 통신 없이 객체 탐지 성능을 독립적으로 확인하려면 `local_vision_monitor.py`를 실행.

### 실행 방법

- 명령어: `python local_vision_monitor.py`
- 결과: 카메라에서 탐지된 객체(예: 컵과 드리퍼)를 실시간으로 모니터링.

---

### 2. `vision_ros_bridge.py`

- **목적**: 로컬 비전 시스템을 ROS 브릿지 서버와 통합하여 통신 및 제어 제공.
- **주요 기능**:
    - Flask-SocketIO를 사용하여 이벤트를 처리.
    - WebSocket 기반 통신으로 탐지된 객체 정보를 ROS 브릿지 서버에 전송.
    - `classify_cropped_image` 및 `get_message`와 같은 작업을 백그라운드에서 실행.
- **사용 사례**: 로봇 제어와 통합된 비전 시스템의 전체 기능을 사용하려면 `vision_ros_bridge.py`를 실행.

### 실행 방법

- 명령어: `python vision_ros_bridge.py`
- 결과: 객체 탐지 데이터를 ROS 브릿지 서버에 전송하고 양방향 통신 활성화.

---

## 설치 방법

1. `requirements.txt`에 나열된 필수 종속성을 설치:
    
    ```
    pip install -r requirements.txt
    ```
    
2. [Azure Kinect SDK 및 Python 바인딩](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)을 설정.
3. YOLO 모델 파일(`241119_best.pt`)을 적절한 디렉토리에 배치.

---
