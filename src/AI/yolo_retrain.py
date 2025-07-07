# YOLOv8-face + 연령 분류 + 실패 케이스 수집 (저장 대상 시각화 버전)
# 수정사항:
# 1. 화면 중앙에 가장 가까운 얼굴을 '저장 대상'으로 지정
# 2. '저장 대상'의 바운딩 박스를 보라색으로 표시하여 사용자에게 명확한 피드백 제공

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from collections import deque
import os
import datetime
import time

# -------------------------------
# [0] 경로 및 데이터 저장 설정
# -------------------------------
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 모델 파일 경로 ---
YOLO_MODEL_PATH = os.path.join(script_dir, "yolov8n-face.pt")
AGE_MODEL_PATH = os.path.join(script_dir, "models/age_classifier_v5.keras") 

# --- 실패 케이스 저장 폴더 설정 ---
SAVE_DIR = os.path.join(script_dir, 'datasets', 'hard_cases_for_v5') 
os.makedirs(os.path.join(SAVE_DIR, 'train', 'child'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'train', 'adult'), exist_ok=True)
print(f"실패 케이스 저장 폴더: {os.path.join(SAVE_DIR, 'train')}")

# -------------------------------
# [1] 모델 로드
# -------------------------------
print(f"나이 분류 모델 로드: {AGE_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)
age_model = load_model(AGE_MODEL_PATH)

# -------------------------------
# [2] 얼굴 전처리 및 분류 함수
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
# [3] 후처리 및 라벨링 함수
# -------------------------------
face_histories = {} 

def get_smoothed_label(face_id):
    if face_id not in face_histories or not face_histories[face_id]['probs']:
        return "N/A", (255, 255, 255)
    
    avg_prob = np.mean(face_histories[face_id]['probs'])
    
    if avg_prob > 0.6: final_label, color = "adult", (0, 0, 255)
    elif avg_prob < 0.4: final_label, color = "child", (255, 0, 0)
    else: final_label, color = "Uncertain", (0, 255, 255)
        
    return f"ID {face_id}: {final_label} ({avg_prob:.2f})", color

# -------------------------------
# [4] 실시간 웹캠 연동 (저장 대상 시각화)
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_height, frame_width, _ = frame.shape
    frame_center = (frame_width / 2, frame_height / 2)
    
    results = yolo_model.track(frame, persist=True, conf=0.5)
    
    current_frame_ids = set()
    face_rois_to_save = {}
    
    min_dist_to_center = float('inf')
    target_id = -1 # [추가] 저장 대상 ID를 저장할 변수

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        # [수정] 첫 번째 루프: 데이터 처리 및 저장 대상 찾기
        for box, face_id in zip(boxes, ids):
            current_frame_ids.add(face_id)
            if face_id not in face_histories:
                face_histories[face_id] = {'probs': deque(maxlen=15), 'last_seen': time.time()}

            face_histories[face_id]['last_seen'] = time.time()
            
            x1, y1, x2, y2 = box
            padding = 20
            x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
            x2_pad, y2_pad = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            face_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad] 
            
            if face_roi.size > 0:
                face_rois_to_save[face_id] = face_roi
                prob = predict_age_prob(face_roi)
                if prob is not None:
                    face_histories[face_id]['probs'].append(prob)

            # 화면 중앙과 가장 가까운 얼굴을 '저장 대상'으로 지정
            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            dist = np.linalg.norm(np.array(box_center) - np.array(frame_center))
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                target_id = face_id
        
        # [추가] 두 번째 루프: 화면에 그리기
        for box, face_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            display_label, default_color = get_smoothed_label(face_id)
            
            # 저장 대상이면 보라색, 아니면 예측 결과 색상으로 표시
            if face_id == target_id:
                draw_color = (255, 0, 255) # 보라색
            else:
                draw_color = default_color

            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)


    # 사라진 ID 정리
    disappeared_ids = set(face_histories.keys()) - current_frame_ids
    for face_id in disappeared_ids:
        if time.time() - face_histories[face_id]['last_seen'] > 5.0:
            if face_id in face_histories:
                del face_histories[face_id]

    cv2.imshow("Data Collection | Target: Purple Box | 'a'/'c': Save, 'q': Quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
        
    # 키 입력 시, '저장 대상(target_id)'으로 지정된 얼굴만 저장
    if (key == ord('a') or key == ord('c')) and target_id != -1:
        if target_id in face_rois_to_save:
            face_to_save = face_rois_to_save[target_id]
            save_folder = 'adult' if key == ord('a') else 'child'
            save_path = os.path.join(SAVE_DIR, 'train', save_folder)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"hardcase_ID{target_id}_{timestamp}.jpg"
            full_save_path = os.path.join(save_path, filename)

            cv2.imwrite(full_save_path, face_to_save)
            print(f"실패 케이스 저장 (ID: {target_id}) -> 올바른 레이블: '{save_folder}', 경로: {full_save_path}")
        else:
            print(f"경고: 저장 대상(ID: {target_id})의 얼굴 이미지를 찾을 수 없습니다.")


cap.release()
cv2.destroyAllWindows()