### main.py
"""
주요 목적:
1. Azure Kinect 카메라로 입력 데이터를 처리.
2. YOLO 모델로 객체를 탐지하여 컵 및 드리퍼 상태를 실시간으로 업데이트.
3. ResNet 기반 분류 모델로 드리퍼의 상태를 판단.
"""

import time
import cv2
import sys
import os
import copy
import numpy as np
import torch
from torchvision import models
from ultralytics import YOLO
from model.resnet import ResNetBackboneModel
from utils import find_nearest, get_boxes_and_corners, create_empty_items, make_message, speaking

# Azure Kinect 라이브러리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../pyKinectAzure'))
import pykinect_azure as pykinect

# 전역 변수 초기화
dripper_is_empty = create_empty_items(item_type="dripper")
cup_is_empty = create_empty_items(item_type="cup")
success_info = ['', '', '']

model = None  # YOLO 모델 객체
c_model = None  # ResNet 기반 분류 모델
image = None  # 현재 카메라 이미지

# Kinect 초기화 함수
def initialize_kinect():
    """
    Azure Kinect 장치를 초기화하고, 설정을 구성한 뒤 반환.
    - 색상 형식: BGRA32
    - 해상도: 720P
    - 깊이 모드: WFOV_2X2BINNED
    """
    print("[INFO] Initializing Azure Kinect...")
    pykinect.initialize_libraries()

    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = pykinect.start_device(config=device_config)
    print("[INFO] Azure Kinect Initialized.")
    return device

# 비전 데이터를 지속적으로 업데이트
def update_vision_data():
    """
    Kinect에서 데이터를 수집하고 YOLO 모델을 사용하여 컵 및 드리퍼 상태를 업데이트.
    - YOLO 모델을 사용해 객체 탐지 및 상태 업데이트.
    - 감지된 객체를 기반으로 컵 및 드리퍼 정보를 기록.
    - 결과 이미지를 저장하고 실시간으로 시각화.
    """
    global dripper_is_empty, cup_is_empty, model, image
    device = initialize_kinect()
    if model is None:
        print("[INFO] Loading YOLO model...")
        model = YOLO("./model/241119_best.pt")
        print("[INFO] YOLO model loaded.")

    while True:
        capture = device.update()
        ret_color, color_image = capture.get_color_image()
        ret_depth, _ = capture.get_transformed_depth_image()
        
        dripper_temp = create_empty_items(item_type="dripper")
        cup_temp = create_empty_items(item_type="cup")
        image = color_image

        if ret_color and ret_depth:
            print("[INFO] Capturing image and depth data...")
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            resized_image = cv2.resize(color_image, (640, 640))
            bluemark = dripper_position(resized_image)  # 드리퍼 위치 감지

            yolo_results = model(resized_image)  # YOLO로 객체 탐지
            holder_coord = []

            # "Holder" 탐지 및 좌표 기록
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_points = get_boxes_and_corners(x1, y1, x2, y2)
                pix_x, pix_y = map(int, all_points['center'])
                class_id = int(box.cls)
                class_name = model.names[class_id]
                if class_name == "Holder":
                    center_coord = (pix_x, pix_y)
                    holder_coord.append(center_coord)

            holder_coord = sorted(holder_coord, key=lambda x: x[0])  # X좌표 기준 정렬

            # 컵 및 드리퍼 상태 탐지
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_points = get_boxes_and_corners(x1, y1, x2, y2)
                pix_x, pix_y = map(int, all_points['center'])
                class_id = int(box.cls)
                class_name = model.names[class_id]

                if class_name == "Cup":    
                    index = find_nearest(pix_x, y2, holder_coord, True)
                    try:
                        if index is not None:
                            cup_temp[index]["exist_cup"] = True
                            cup_temp[index]['center'] = [pix_x, pix_y]
                            cup_temp[index]['coordinate'] = (x1, y1, x2, y2)  
                            print(f"[INFO] Cup detected at index {index}: {cup_temp[index]}")
                    except Exception as e:
                        print(f"[ERROR] Exception in Cup detection: {e}")                                    
                elif class_name == "Holder":
                    pass
                else:
                    index = find_nearest(x2, y2, bluemark, False)
                    if index is not None:
                        dripper_temp[index]["exist_cup"] = True
                        dripper_temp[index]['center'] = [pix_x, pix_y]
                        dripper_temp[index]['coordinate'] = (x1, y1, x2, y2)
                        if class_name == "Dripper_empty":
                            dripper_temp[index]['exist_dripper'] = True
                        else:
                            dripper_temp[index]['exist_coffee_beans'] = True
                        print(f"[INFO] Dripper detected at index {index}: {dripper_temp[index]}")

            # 상태 업데이트 및 시각화
            dripper_is_empty = copy.deepcopy(dripper_temp)
            cup_is_empty = copy.deepcopy(cup_temp)
            result_image = yolo_results[0].plot()
            cv2.imshow('YOLO Results', result_image)
            cv2.imwrite('captured_rgb_image_with_yolo.png', result_image)

            time.sleep(1)

        if cv2.waitKey(1) == ord('q'):
            print("[INFO] Exiting update loop.")
            break

# 드리퍼 상태 분류
def classify_cropped_image():
    """
    ResNet 기반 모델을 사용하여 드리퍼 내 상태를 분류.
    - 드리퍼 내부 상태를 성공/실패로 분류.
    - 입력 이미지를 전처리하여 ResNet에 전달.
    """
    global c_model, dripper_is_empty, image
    if c_model is None:
        print("[INFO] Loading ResNet model for classification...")
        c_model = ResNetBackboneModel(pretrained=False)
        c_model.load_state_dict(torch.load('./model/drip_classification_model (1).pth'))
        c_model.eval()
        print("[INFO] ResNet model loaded.")

    dripper_success = [None] * 3
    if image is not None and image.size > 0:
        for i, dripper_coord in enumerate(dripper_is_empty):
            if dripper_coord['exist_coffee_beans']:
                x1, y1, x2, y2 = dripper_coord['coordinate']
                cropped_image = image[y1:y2, x1:x2]
                resized_image = cv2.resize(cropped_image, (32, 32))
                
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                image_tensor = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
                image_tensor = image_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = c_model(image_tensor).item()
                    print(f"[INFO] Classification output for Dripper {i}: {output}")
                    dripper_success[i] = round(output)

    # 성공 및 실패 인덱스 출력
    print(f"[INFO] Dripper Success Indices: {[i for i, value in enumerate(dripper_success) if value == 1]}")
    print(f"[INFO] Dripper Failure Indices: {[i for i, value in enumerate(dripper_success) if value == 0]}")

    return dripper_success

# 드리퍼 위치 감지 함수
def dripper_position(image):
    """
    이미지에서 파란색 마커를 감지하여 드리퍼 위치를 확인.
    - HSV 마스크를 생성하여 특정 색상을 필터링.
    - 마커의 중심 좌표를 반환.
    """
    blue_mark_coord = [(0, 0), (0, 0), (0, 0)]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 150, 50])
    upper_blue = np.array([130, 255, 200])
    mask1 = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_mask = mask1

    height, width = blue_mask.shape
    x_threshold = int(width * (1 / 2))
    filtered_mask = np.zeros_like(blue_mask)
    filtered_mask[:, x_threshold:] = blue_mask[:, x_threshold:]

    kernel = np.ones((5, 5), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    j = 0
    for contour in contours:        
        area = cv2.contourArea(contour)
        if area > 80:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            blue_mark_coord[j] = (x + w, y + h)
            j += 1
            print(f"[INFO] Blue marker detected: {blue_mark_coord[j-1]}")
            if j == 3: break

    blue_mark_coord = sorted(blue_mark_coord, key=lambda x: x[1], reverse=False)
    return blue_mark_coord

# 드리퍼 및 컵 상태 반환
def get_metadata():
    """
    드리퍼와 컵 상태 데이터를 반환.
    """
    global dripper_is_empty, cup_is_empty
    print("[INFO] Returning metadata for dripper and cup.")
    return dripper_is_empty, cup_is_empty

# 음성 메시지 생성 및 출력
def get_message():
    """
    드리퍼 및 컵 상태를 바탕으로 음성 메시지를 생성하고 출력.
    """
    global dripper_is_empty, cup_is_empty
    text_ko, text_eng = make_message(dripper_is_empty, cup_is_empty)
    print(f"[INFO] Generated Messages:\nKorean: {text_ko}\nEnglish: {text_eng}")
    speaking(text_ko, text_eng)

# 메인 함수
if __name__ == '__main__':
    print("[INFO] Starting vision data update loop...")
    update_vision_data()
