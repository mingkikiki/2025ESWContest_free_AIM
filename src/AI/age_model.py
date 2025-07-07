# 전이학습 기반 연령 분류 모델 - 오류 수정 및 최종 버전
# 모델: EfficientNetB0
# 기법: 2단계 파인튜닝, 모델 전용 전처리, Dropout, 학습률 스케줄러, 클래스 가중치

import os
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------------------------------------
# 1. 로컬 압축 해제 및 데이터 준비
# --------------------------------------
def extract_if_needed(zip_path, extract_to):
    """지정된 경로에 폴더가 없으면 압축을 해제합니다."""
    if not os.path.exists(extract_to):
        print(f"🔄 압축 해제 중: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(extract_to)) # 상위 폴더에 압축 해제
        print(f"✅ 압축 해제 완료: {extract_to}")
    else:
        print(f"✅ 폴더 존재: {extract_to} → 압축 해제 생략")

def classify_age(age):
    """나이를 'child' 또는 'adult'로 분류합니다. 중간 연령대는 제외합니다."""
    if age <= 14:
        return 'child'
    elif age >= 25:
        return 'adult'
    else:
        return None

def prepare_dataset():
    """데이터셋 압축을 풀고, 연령에 따라 분류하여 train/val/test 폴더로 복사합니다."""
    print("--- 데이터셋 준비 시작 ---")
    
    # 로컬 환경에 'UTKFace.zip' 파일이 있어야 합니다.
    extract_if_needed("UTKFace.zip", "./UTKFace")

    # UTKFace 데이터셋 경로 확인
    utk_base_path = './UTKFace'
    # 압축 해제 시 'UTKFace/UTKFace' 구조로 풀리는 경우가 많으므로 내부 폴더를 우선적으로 사용
    utk_image_path = os.path.join(utk_base_path, 'UTKFace')
    if not os.path.exists(utk_image_path):
        utk_image_path = utk_base_path # 내부 폴더가 없으면 상위 폴더 사용

    merged_base = './merged_dataset_improved'
    if os.path.exists(merged_base):
        print(f"이미 '{merged_base}' 폴더가 존재합니다. 기존 폴더를 삭제하고 다시 생성합니다.")
        shutil.rmtree(merged_base)
    
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        for label in ['child', 'adult']:
            os.makedirs(os.path.join(merged_base, subset, label), exist_ok=True)

    all_files = []
    if os.path.exists(utk_image_path):
        print(f"이미지 검색 경로: {utk_image_path}")
        for f in os.listdir(utk_image_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_' in f:
                try:
                    age = int(f.split('_')[0])
                    label = classify_age(age)
                    if label:
                        all_files.append((os.path.join(utk_image_path, f), label))
                except (ValueError, IndexError):
                    continue
    else:
        print(f"⚠️ UTKFace 이미지 경로를 찾을 수 없습니다: {utk_image_path}")

    if not all_files:
        print("🚨 처리할 이미지 파일을 찾지 못했습니다. 데이터셋 경로 및 파일명을 확인하세요.")
        return

    print(f"총 {len(all_files)}개의 유효 이미지 파일을 찾았습니다.")

    train, temp = train_test_split(all_files, test_size=0.3, stratify=[l for _, l in all_files], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=[l for _, l in temp], random_state=42)

    def copy_data(file_list, subset):
        count = 0
        for src, label in file_list:
            if os.path.exists(src):
                filename = os.path.basename(src)
                dst = os.path.join(merged_base, subset, label, filename)
                shutil.copyfile(src, dst)
                count += 1
        print(f"'{subset}' 세트에 {count}개 이미지 복사 완료.")

    copy_data(train, 'train')
    copy_data(val, 'val')
    copy_data(test, 'test')
    print("✅ 데이터셋 준비 및 병합/분류 완료")

prepare_dataset()

# --------------------------------------
# 2. 데이터 로딩
# --------------------------------------
base_dir = './merged_dataset_improved'
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
test_data = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# [오류 수정] 클래스 가중치 계산 시 타입을 int로 명시
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes.astype(int)  # <-- 오류 수정된 부분
)
class_weight = dict(enumerate(weights))
print(f"계산된 클래스 가중치: {class_weight}")

# --------------------------------------
# 3. 모델 구성
# --------------------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)
model.summary()

# --------------------------------------
# 4. 학습
# --------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

# --- 1단계: Feature Extraction ---
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- 1단계 학습 시작 (Feature Extraction) ---")
history_phase1 = model.fit(train_data,
                           epochs=10,
                           validation_data=val_data,
                           class_weight=class_weight,
                           callbacks=[lr_scheduler])

# --- 2단계: Fine-tuning ---
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- 2단계 학습 시작 (Fine-tuning) ---")
history_phase2 = model.fit(train_data,
                           epochs=50,
                           validation_data=val_data,
                           class_weight=class_weight,
                           callbacks=[early_stop, lr_scheduler],
                           initial_epoch=len(history_phase1.epoch))

# --------------------------------------
# 5. 평가 및 저장
# --------------------------------------
model.save('efficientnetb0_age_classifier_improved_fixed.h5')
print("\n✅ 개선된 모델 'efficientnetb0_age_classifier_improved_fixed.h5' 저장 완료")

print("\n--- 최종 모델 평가 ---")
test_loss, test_acc = model.evaluate(test_data)
print(f"테스트 데이터 손실 (Test Loss): {test_loss:.4f}")
print(f"테스트 데이터 정확도 (Test Accuracy): {test_acc:.4f}")

y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Improved Model - Fixed)")
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys())) 