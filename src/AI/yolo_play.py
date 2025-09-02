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
ADULT_SOUND_FILE = os.path.join(script_dir, 'normal_tts.mp3')
CHILD_SOUND_FILE = os.path.join(script_dir, 'child_tts.mp3')

pygame.mixer.init()
script_start_time = time.time()
INITIAL_GRACE_PERIOD = 5.0
CONSECUTIVE_SECONDS_REQUIRED = 5.0
empty_scene_start_time = None
RESET_GRACE_PERIOD = 1.0

# -------------------------------
# [1] 모델 로드
# -------------------------------
print(f"나이 분류 모델 로드: {AGE_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
age_model = load_model(AGE_MODEL_PATH)

# -------------------------------
# [2] 얼굴 이미지 전처리 함수 (CLAHE 역광 보정 추가됨)
# -------------------------------
def preprocess_face(face_bgr):
    h, w = face_bgr.shape[:2]
    if h < 40 or w < 40:
        return None
    try:
        # --- [수정됨] 역광 보정을 위한 CLAHE 적용 ---
        # 1. BGR 이미지를 LAB 색 공간으로 변환 (L: 밝기, a,b: 색상 정보)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        
        # 2. L(밝기) 채널만 분리
        l, a, b = cv2.split(lab)
        
        # 3. CLAHE 객체 생성 및 L 채널에 적용하여 대비를 향상시킴
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 4. 향상된 L 채널을 다시 병합
        lab_enhanced = cv2.merge((cl, a, b))
        
        # 5. 다시 BGR 색 공간으로 변환
        face_bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        # --- CLAHE 적용 끝 ---

        # 이후 로직은 보정된 이미지를 사용
        face_resized = cv2.resize(face_bgr_enhanced, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        preprocessed_face = preprocess_input(face_rgb)
        
        return np.expand_dims(preprocessed_face, axis=0)
    except Exception as e:
        print(f"얼굴 전처리 중 오류: {e}")
        return None

# -------------------------------
# [3] 얼굴 분류 함수
# -------------------------------
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
# [5] 실시간 웹캠 연동
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
            last_class = face_histories[face_id].get('last_class')

            if current_class is not None and current_class != "Uncertain":
                if current_class != last_class:
                    face_histories[face_id]['last_class'] = current_class
                    face_histories[face_id]['class_start_time'] = time.time()
                elif face_histories[face_id].get('class_start_time') is not None:
                    duration = time.time() - face_histories[face_id]['class_start_time']
                    if duration >= CONSECUTIVE_SECONDS_REQUIRED:
                        if current_class == 'child': child_alert_triggered = True
                        elif current_class == 'adult': adult_alert_triggered = True

    if not current_frame_ids:
        if empty_scene_start_time is None:
            empty_scene_start_time = time.time()
        elif time.time() - empty_scene_start_time >= RESET_GRACE_PERIOD:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                print(f"{RESET_GRACE_PERIOD}초간 대상이 없어 오디오를 중지하고 상태를 초기화합니다.")
            last_triggered_sound_class = None
            face_histories.clear()
            empty_scene_start_time = None
    else:
        empty_scene_start_time = None

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
            if current_frame_ids:
                 last_triggered_sound_class = None

    cv2.imshow("Multi-face Age Classification with Tracking (Backlight Corrected)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
