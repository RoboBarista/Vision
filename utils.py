"""
utils.py

주요 목적:
- 보조적인 유틸리티 함수들을 제공하여 메인 비전 시스템의 코드 가독성과 재사용성을 높임.
"""

import math
import pyttsx3
import numpy as np

# 네 개의 꼭짓점과 중심 좌표를 계산하여 반환하는 함수
def get_boxes_and_corners(x1, y1, x2, y2):
    """
    좌표 (x1, y1)와 (x2, y2)를 사용해 상자 꼭짓점과 중심 좌표를 계산.
    반환값:
    - 딕셔너리 형태로 상자 네 꼭짓점(top_left, top_right, bottom_left, bottom_right) 및 중심 좌표(center)를 포함.
    """
    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = (center_x, center_y)
    all_points = {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
        'center': center
    }
    return all_points

# 빈 상태의 드리퍼나 컵 정보를 생성하는 함수
def create_empty_items(count=3, item_type="dripper"):
    """
    지정된 개수만큼 빈 드리퍼 또는 컵 상태를 생성.
    - count: 생성할 항목의 개수 (기본값: 3)
    - item_type: 항목 유형("dripper" 또는 "cup")
    반환값:
    - 리스트 형태로 빈 상태의 드리퍼 또는 컵 정보.
    """
    base_item = {
        "order": None,  # 순서
        "coordinate": [],  # 좌표 정보
        "center": []  # 중심 좌표
    }
    extra_fields = {
        "dripper": {"exist_dripper": False, "exist_coffee_beans": False},
        "cup": {"exist_cup": False}
    }
    return [
        {
            **base_item,
            "order": i,
            **extra_fields.get(item_type, {})
        }
        for i in range(1, count + 1)
    ]

# 가장 가까운 마커 또는 드리퍼를 찾는 함수
def find_nearest(pix_x, pix_y, blue_mark_coord, is_cup):
    """
    특정 픽셀 좌표 (pix_x, pix_y)와 주어진 좌표들 간의 거리 계산.
    - blue_mark_coord: 드리퍼나 마커의 좌표 리스트.
    - is_cup: True이면 컵에 대한 거리 제한(150)을 적용.
    반환값:
    - 가장 가까운 드리퍼/마커의 인덱스 또는 None.
    """
    distances = []
    for i, (x, y) in enumerate(blue_mark_coord):
        distance = np.sqrt((pix_x - x) ** 2 + (pix_y - y) ** 2)  # 유클리디안 거리 계산
        if is_cup:
            if distance < 150:  # 컵은 거리 제한 적용
                distances.append((i, distance))
        else:
            distances.append((i, distance))

    if distances:
        nearest_dripper = min(distances, key=lambda x: x[1])  # 최소 거리 계산
        nearest_index = nearest_dripper[0]
        return nearest_index
    return

# 드리퍼와 컵 상태 정보를 바탕으로 메시지를 생성하는 함수
def make_message(dripper_info, cup_info):
    """
    드리퍼와 컵 상태를 바탕으로 한국어 및 영어 메시지를 생성.
    반환값:
    - 한국어 메시지 (문자열)
    - 영어 메시지 (문자열)
    """
    number_to_kor = {
        1: '일',
        2: '이',
        3: '삼'
    }

    # 한국어 메시지 생성
    dripper_orders_kr = [f"{number_to_kor.get(d['order'])}번" for d in dripper_info if d['exist_dripper']]
    cup_orders_kr = [f"{number_to_kor.get(c['order'])}번" for c in cup_info if c['exist_cup']]
    
    if dripper_orders_kr:
        dripper_message_kr = f"{' '.join(dripper_orders_kr)} 위치에 드리퍼가 있습니다."
    else:
        dripper_message_kr = "드리퍼가 없습니다."
        
    if cup_orders_kr:
        cup_message_kr = f"{' '.join(cup_orders_kr)} 위치에 컵이 있습니다."
    else:
        cup_message_kr = "컵이 없습니다."
    
    message_kr = dripper_message_kr + " " + cup_message_kr

    # 영어 메시지 생성
    dripper_orders_en = [str(d['order']) for d in dripper_info if d['exist_dripper']]
    cup_orders_en = [str(c['order']) for c in cup_info if c['exist_cup']]

    if len(dripper_orders_en) == 1:
        dripper_message_en = f"A dripper is placed in position {dripper_orders_en[0]}."
    elif len(dripper_orders_en) > 1:
        dripper_message_en = f"Drippers are placed in positions {', '.join(dripper_orders_en[:-1])} and {dripper_orders_en[-1]}."
    else:
        dripper_message_en = "No dripper available."

    if len(cup_orders_en) == 1:
        cup_message_en = f"A cup is placed in position {cup_orders_en[0]}."
    elif len(cup_orders_en) > 1:
        cup_message_en = f"Cups are placed in positions {', '.join(cup_orders_en[:-1])} and {cup_orders_en[-1]}."
    else:
        cup_message_en = "No cup available."

    message_en = dripper_message_en + " " + cup_message_en

    return message_kr, message_en

# 텍스트를 음성으로 읽어주는 함수
def speaking(text_kr, text_en):
    """
    주어진 텍스트를 pyttsx3를 사용하여 음성 출력.
    - text_kr: 한국어 텍스트
    - text_en: 영어 텍스트
    """
    engine = pyttsx3.init()

    # 한국어 읽기 설정
    engine.setProperty('rate', 200)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'korean' in voice.name.lower():  # 시스템에 따라 'korean' 인식 가능
            engine.setProperty('voice', voice.id)
            break
    engine.say(text_kr)
    engine.runAndWait()

    # 영어 읽기 설정
    for voice in voices:
        if 'english' in voice.name.lower():  # 시스템에 따라 'english' 인식 가능
            engine.setProperty('voice', voice.id)
            break
    engine.say(text_en)
    engine.runAndWait()
