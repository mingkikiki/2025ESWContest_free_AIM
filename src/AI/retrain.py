# [최종 통합] 누적 데이터 병합 및 GPU 재학습 자동화 스크립트
# 이 스크립트 하나만 실행하면, 지정된 버전의 실패 케이스를 모두 합쳐
# 기존 모델을 재학습하고 새로운 버전의 모델을 저장합니다.

import os
import shutil
import glob
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# ==============================================================================
# [설정 부분] - 재학습 시 이 부분만 수정하면 됩니다!
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 1. 모델 버전 설정 ---
# 불러올 기본 모델 (가장 최신 버전)
BASE_MODEL_PATH = os.path.join(script_dir, 'models', 'age_classifier_v4.keras') 
# 새로 저장될 모델 (다음 버전)
NEW_MODEL_PATH = os.path.join(script_dir, 'models', 'age_classifier_v5.keras')

# --- 2. 데이터셋 버전 설정 ---
# 재학습에 사용할 모든 "실패 케이스" 버전 목록
# 예: v3 모델을 만들려면, v2의 실패 케이스까지 포함합니다.
VERSIONS_TO_COMBINE = [
    'hard_cases_for_v2', 
    'hard_cases_for_v3', # v3의 실패 케이스가 수집되면 이 줄의 주석을 해제
    'hard_cases_for_v4',
]
# 버전별 데이터셋이 저장된 기본 폴더
BASE_DATASET_DIR = os.path.join(script_dir, 'hard_datasets') 

# --- 3. 학습 파라미터 설정 ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-6
# ==============================================================================


# [1단계: 누적 데이터 준비]
def prepare_cumulative_dataset():
    """설정된 버전의 모든 실패 케이스를 임시 폴더 하나로 통합합니다."""
    print("="*60)
    print("[1단계] 누적 재학습을 위한 데이터 통합을 시작합니다.")
    
    # 데이터가 통합될 임시 폴더
    cumulative_dir_base = os.path.join(script_dir, 'temp_cumulative_data')
    cumulative_dir_train = os.path.join(cumulative_dir_base, 'train')

    # 기존 임시 폴더가 있으면 완전히 삭제
    if os.path.exists(cumulative_dir_base):
        shutil.rmtree(cumulative_dir_base)
        print(f"기존 임시 폴더 '{cumulative_dir_base}'를 삭제했습니다.")

    # 임시 폴더 생성
    os.makedirs(os.path.join(cumulative_dir_train, 'child'))
    os.makedirs(os.path.join(cumulative_dir_train, 'adult'))
    print(f"새로운 임시 폴더 '{cumulative_dir_train}'를 생성했습니다.")

    total_copied = 0
    for version_folder in VERSIONS_TO_COMBINE:
        source_dir = os.path.join(BASE_DATASET_DIR, version_folder, 'train')
        if not os.path.exists(source_dir):
            print(f"경고: 소스 폴더를 찾을 수 없습니다 - {source_dir}")
            continue
        
        print(f"-> '{version_folder}'의 데이터 복사 중...")
        for class_name in ['child', 'adult']:
            class_source_dir = os.path.join(source_dir, class_name)
            class_dest_dir = os.path.join(cumulative_dir_train, class_name)
            
            image_files = glob.glob(os.path.join(class_source_dir, '*.*'))
            for f in image_files:
                shutil.copy(f, class_dest_dir)
                total_copied += 1
                
    if total_copied == 0:
        print("오류: 통합할 이미지가 없습니다. 데이터 경로와 버전 목록을 확인하세요.")
        return None

    print(f"\n데이터 통합 완료! 총 {total_copied}개의 파일을 '{cumulative_dir_base}'로 복사했습니다.")
    print("="*60)
    return cumulative_dir_train


# [2단계: tf.data 파이프라인 구축]
def create_tf_dataset(data_dir):
    """지정된 데이터 디렉토리로부터 tf.data.Dataset을 생성합니다."""
    # ... (이전 재학습 스크립트의 함수들을 여기에 포함)
    def load_data_paths(d_dir):
        image_paths, labels = [], []
        class_map = {'child': 0, 'adult': 1}
        for class_name, class_idx in class_map.items():
            class_dir = os.path.join(d_dir, class_name)
            paths = glob.glob(os.path.join(class_dir, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(class_dir, '*.[jJ][pP][eE][gG]')) + glob.glob(os.path.join(class_dir, '*.[pP][nN][gG]'))
            image_paths.extend(paths)
            labels.extend([class_idx] * len(paths))
        if not image_paths: return None, None
        return image_paths, labels

    def load_and_preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = preprocess_input(img)
        return img, label

    def add_sample_weight(image, label, class_weight_tensor):
        sample_weight = tf.gather(class_weight_tensor, label)
        return image, label, sample_weight

    image_paths, labels = load_data_paths(data_dir)
    if not image_paths: return None

    weights = compute_class_weight('balanced', classes=np.unique(labels), y=np.array(labels))
    class_weight_dict = dict(enumerate(weights))
    class_weight_tensor = tf.convert_to_tensor(list(class_weight_dict.values()), dtype=tf.float32)
    print(f"통합 데이터 클래스 가중치: {class_weight_dict}")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, np.array(labels, dtype=np.int32)))
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: add_sample_weight(x, y, class_weight_tensor), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# [메인 실행 블록]
if __name__ == "__main__":
    # 1. 데이터 통합 실행
    retrain_data_dir = prepare_cumulative_dataset()
    if not retrain_data_dir:
        exit()

    # 2. GPU 확인
    print("\n[2단계] GPU 장치 확인 중...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)}개의 GPU를 찾았으며, 학습에 사용합니다: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ 경고: GPU를 찾을 수 없습니다. CPU로 학습을 진행합니다.")
    print("="*60)

    # 3. 데이터셋 생성
    print("\n[3단계] TensorFlow 데이터셋 생성 중...")
    dataset = create_tf_dataset(retrain_data_dir)
    if dataset is None:
        exit()
    print("="*60)
        
    # 4. 모델 로드 및 컴파일
    print(f"\n[4단계] 기존 모델 로드 및 컴파일 중...")
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"오류: 기본 모델 파일을 찾을 수 없습니다: {BASE_MODEL_PATH}")
        exit()
    model = load_model(BASE_MODEL_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(f"'{os.path.basename(BASE_MODEL_PATH)}' 모델을 성공적으로 불러왔습니다.")
    print("="*60)

    # 5. 추가 학습 (Fine-tuning)
    print(f"\n[5단계] 모델 재학습 시작... (Epochs: {EPOCHS})")
    checkpoint = ModelCheckpoint(filepath=NEW_MODEL_PATH, 
                                 monitor='accuracy', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    history = model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    print("="*60)
    print(f"\n재학습 완료! GPU로 학습된 개선 모델이 다음 경로에 저장되었습니다: {NEW_MODEL_PATH}")

    # (선택) 임시 데이터 폴더 삭제
    # try:
    #     shutil.rmtree(os.path.dirname(retrain_data_dir))
    #     print(f"임시 데이터 폴더 '{os.path.dirname(retrain_data_dir)}'를 삭제했습니다.")
    # except Exception as e:
    #     print(f"임시 폴더 삭제 중 오류 발생: {e}")