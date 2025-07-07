# YOLOv8-face + 연령 분류 (대상 부재 시 즉시 초기화 기능 추가 최종 버전)
# 수정사항:
# 1. 화면에 인식된 얼굴이 하나도 없을 경우, 모든 오디오 및 추적 상태를 즉시 초기화
# 2. 재생 중인 TTS도 즉시 중지하여 더 나은 사용자 경험 제공

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from collections import deque
import os
import time
import pygame

# -------------------------------
# [0] 경로, 오디오, 시간 설정
# -------------------------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

YOLO_MODEL_PATH = os.path.join(script_dir, "yolov8n-face.pt")
AGE_MODEL_PATH = os.path.join(script_dir, "models/age_classifier_v5.keras") 
ADULT_SOUND_FILE = os.path.join(script_dir, 'mask_normal_tts.mp3')
CHILD_SOUND_FILE = os.path.join(script_dir, 'mask_child_tts.mp3')

pygame.mixer.init()
script_start_time = time.time()
INITIAL_GRACE_PERIOD = 5.0
CONSECUTIVE_SECONDS_REQUIRED = 5.0

# -------------------------------
# [1] 모델 로드
# -------------------------------
print(f"나이 분류 모델 로드: {AGE_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
age_model = load_model(AGE_MODEL_PATH)

# -------------------------------
# [2] 얼굴 전처리 및 분류 함수 (이전과 동일)
# -------------------------------
def preprocess_face(face_bgr):
    h, w = face_bgr.shape[:2]
    if h < 40 or w < 40: return None
    try:
        face_resized = cv2.resize(face_bgr, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(preprocess_input(face_rgb), axis=0)
    except: return None

def predict_age_prob(face_bgr):
    preprocessed = preprocess_face(face_bgr)
    if preprocessed is None: return None
    return age_model.predict(preprocessed, verbose=0)[0][0]

# -------------------------------
# [4] 후처리 및 라벨링 함수
# -------------------------------
face_histories = {} 
last_triggered_sound_class = None 

def get_smoothed_label(face_id):
    if face_id not in face_histories or not face_histories[face_id]['probs']:
        return "N/A", (255, 255, 255)
    
    avg_prob = np.mean(face_histories[face_id]['probs'])
    
    if avg_prob > 0.6: final_label, color = "adult", (0, 0, 255)
    elif avg_prob < 0.4: final_label, color = "child", (255, 0, 0)
    else: final_label, color = "Uncertain", (0, 255, 255)
        
    return f"ID {face_id}: {final_label} ({avg_prob:.2f})", color

def get_final_class(face_id):
    if face_id not in face_histories or not face_histories[face_id]['probs']: return None
    avg_prob = np.mean(face_histories[face_id]['probs'])
    if avg_prob > 0.6: return "adult"
    elif avg_prob < 0.4: return "child"
    else: return "Uncertain"

# -------------------------------
# [5] 실시간 웹캠 연동 (상태 초기화 로직 추가)
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = yolo_model.track(frame, persist=True, conf=0.5)
    
    current_frame_ids = set()
    child_alert_triggered = False
    adult_alert_triggered = False

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, face_id in zip(boxes, ids):
            current_frame_ids.add(face_id)
            
            if face_id not in face_histories:
                face_histories[face_id] = {'probs': deque(maxlen=15), 'class_start_time': None, 'last_class': None}

            x1, y1, x2, y2 = box
            padding = 20
            x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
            x2_pad, y2_pad = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_roi.size > 0:
                prob = predict_age_prob(face_roi)
                if prob is not None:
                    face_histories[face_id]['probs'].append(prob)
            
            display_label, color = get_smoothed_label(face_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            current_class = get_final_class(face_id)
            last_class = face_histories[face_id]['last_class']

            if current_class is not None and current_class != "Uncertain":
                if current_class != last_class:
                    face_histories[face_id]['last_class'] = current_class
                    face_histories[face_id]['class_start_time'] = time.time()
                elif face_histories[face_id]['class_start_time'] is not None:
                    duration = time.time() - face_histories[face_id]['class_start_time']
                    if duration >= CONSECUTIVE_SECONDS_REQUIRED:
                        if current_class == 'child':
                            child_alert_triggered = True
                        elif current_class == 'adult':
                            adult_alert_triggered = True
    
    # [수정] 모든 대상이 사라졌을 때 시스템 상태 즉시 초기화
    if not current_frame_ids:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop() # 재생 중인 오디오 중지
            print("모든 대상이 사라져 오디오를 중지하고 상태를 초기화합니다.")
        last_triggered_sound_class = None
        face_histories.clear() # 모든 추적 기록 삭제

    # 프레임의 모든 얼굴을 확인한 후, 우선순위에 따라 단 한 번만 오디오 재생 결정
    if time.time() - script_start_time > INITIAL_GRACE_PERIOD:
        if child_alert_triggered:
            if last_triggered_sound_class != 'child' and not pygame.mixer.music.get_busy():
                if os.path.exists(CHILD_SOUND_FILE):
                    print(">>> '아이' 감지! 음성을 재생합니다.")
                    pygame.mixer.music.load(CHILD_SOUND_FILE)
                    pygame.mixer.music.play()
                    last_triggered_sound_class = 'child'
        elif adult_alert_triggered:
            if last_triggered_sound_class != 'adult' and not pygame.mixer.music.get_busy():
                if os.path.exists(ADULT_SOUND_FILE):
                    print(">>> '어른' 감지! 음성을 재생합니다.")
                    pygame.mixer.music.load(ADULT_SOUND_FILE)
                    pygame.mixer.music.play()
                    last_triggered_sound_class = 'adult'
        else:
            # 아무도 경보 조건을 만족하지 않으면, 다음에 바로 소리가 날 수 있도록 상태를 초기화
            if current_frame_ids: # 화면에 사람은 있지만 아직 5초가 안 된 경우
                 last_triggered_sound_class = None


    cv2.imshow("Multi-face Age Classification with Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()