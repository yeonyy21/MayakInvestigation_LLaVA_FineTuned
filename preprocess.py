# preprocess.py

import os
import json
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import logging
import cv2 # OpenCV for reading/writing images for Albumentations
import albumentations as A # Data augmentation library

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s - %(message)s')

# --- 설정값들 ---
RAW_DATA_DIR = "raw_data"
OUTPUT_DATA_DIR = "data"
TRAIN_JSON_FILENAME = "train_drug_data.json"
VAL_JSON_FILENAME = "val_drug_data.json"
TEST_JSON_FILENAME = "test_drug_data.json"

# 데이터 분할 비율
TEST_SPLIT_RATIO = 0.15  # 전체 원본 데이터 중 테스트셋 비율
VAL_SPLIT_FROM_REMAINING_RATIO = 0.15 # (학습+검증) 데이터 중 검증셋 비율

RANDOM_SEED = 42

CLASS_MAPPING = {
    "0": "0_irrelevant",
    "1": "1_chat_history",
    "2": "2_transaction_place", # 이 클래스의 '학습 데이터'를 증강
    "3": "3_drug_raw"
}
TARGET_AUG_CLASS_LABEL_STR = "2" # 증강 대상 클래스의 레이블 문자열
TARGET_AUG_CLASS_FOLDER_NAME = CLASS_MAPPING[TARGET_AUG_CLASS_LABEL_STR]
NUM_AUGMENTATIONS_PER_IMAGE = 4 # 원본 학습 이미지 1개당 생성할 증강 이미지 수 (총 1+4=5개)

FIXED_QUESTION = "이 이미지는 다음 중 어떤 유형입니까? (0: 관련 없음, 1: 마약 대화 내역 사진, 2: 마약 거래 장소, 3: 마약 원본)"

# Albumentations 증강 파이프라인 (학습셋의 특정 클래스에만 적용)
transform_augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=[random.randint(0,50),random.randint(0,50),random.randint(0,50)]), # 빈 공간 어둡게
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    ], p=0.4),
    A.RandomGamma(gamma_limit=(80,120), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomResizedCrop(height=300, width=300, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.3) # 최종적으로 모델 입력 크기로 리사이즈됨
])

def collect_original_image_records(raw_data_path: str, class_mapping: dict) -> list:
    records = []
    logging.info(f"'{raw_data_path}' 폴더에서 원본 이미지 정보 수집 시작...")
    for label_str, folder_name in class_mapping.items():
        class_folder_path = os.path.join(raw_data_path, folder_name)
        if not os.path.isdir(class_folder_path):
            logging.warning(f"클래스 폴더 없음: '{class_folder_path}'. 건너<0xEB><0><0x84><0x90>니다.")
            continue
        image_count_in_folder = 0
        for filename in os.listdir(class_folder_path):
            if '_aug_' in filename: # 이미 증강된 파일은 원본 수집 시 제외
                continue
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                original_image_full_path = os.path.join(class_folder_path, filename)
                try:
                    img_pil = Image.open(original_image_full_path)
                    img_pil.verify() # 간단한 유효성 검사
                    # OpenCV로 읽어서 실제 사용 가능한지 한번 더 확인 (선택적)
                    # img_cv_check = cv2.imread(original_image_full_path)
                    # if img_cv_check is None:
                    #     raise ValueError("cv2.imread returned None")
                except Exception as e:
                    logging.warning(f"유효하지 않은 이미지: '{original_image_full_path}'. 건너<0xEB><0><0x84><0x90>니다. 오류: {e}")
                    continue

                # JSON에 저장될 경로는 RAW_DATA_DIR 부터 시작하는 상대 경로
                json_image_path = os.path.join(RAW_DATA_DIR, folder_name, filename).replace("\\", "/")
                record = {
                    "id": f"{folder_name}_{filename}",
                    "image_path": json_image_path,
                    "question": FIXED_QUESTION,
                    "answer": label_str
                }
                records.append(record)
                image_count_in_folder += 1
        logging.info(f"클래스 '{label_str}' (폴더: '{folder_name}') 원본 이미지 {image_count_in_folder}개 정보 수집 완료.")
    logging.info(f"총 {len(records)}개의 원본 이미지 정보 수집 완료.")
    return records

def augment_and_save_images(original_records: list, target_class_label: str, num_augmentations: int, augment_pipeline: A.Compose) -> list:
    augmented_records = []
    logging.info(f"클래스 '{target_class_label}'에 대한 데이터 증강 시작 (이미지당 {num_augmentations}개 생성)...")

    num_processed_for_aug = 0
    for record in original_records:
        if record['answer'] == target_class_label:
            num_processed_for_aug += 1
            original_image_path = record['image_path'] # 예: "raw_data/2_transaction_place/img.jpg"
            # 스크립트 실행 위치 기준의 실제 디스크 경로
            # IMAGE_BASE_DIRECTORY가 "."이라고 가정하면 original_image_path가 이미 상대경로임
            image_disk_path = original_image_path

            try:
                image_cv = cv2.imread(image_disk_path)
                if image_cv is None:
                    logging.warning(f"증강 위해 이미지 읽기 실패: '{image_disk_path}'. 이 이미지 증강 건너<0xEB><0><0x84><0x90>니다.")
                    continue
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # Albumentations는 RGB 기대
            except Exception as e:
                logging.warning(f"증강 위해 이미지 열기 실패: '{image_disk_path}'. 오류: {e}. 이 이미지 증강 건너<0xEB><0><0x84><0x90>니다.")
                continue

            folder_path = os.path.dirname(image_disk_path) # 예: "raw_data/2_transaction_place"
            base_filename, ext = os.path.splitext(os.path.basename(image_disk_path)) # 예: ("img", ".jpg")

            for i in range(num_augmentations):
                try:
                    augmented = augment_pipeline(image=image_cv)
                    augmented_image_cv = augmented['image']

                    augmented_filename = f"{base_filename}_aug_{i+1}{ext}"
                    augmented_image_save_path_disk = os.path.join(folder_path, augmented_filename)

                    # 저장 시에는 다시 BGR로 (cv2.imwrite는 BGR 순서로 저장)
                    cv2.imwrite(augmented_image_save_path_disk, cv2.cvtColor(augmented_image_cv, cv2.COLOR_RGB2BGR))

                    # JSON에 저장될 경로 (RAW_DATA_DIR 부터 시작)
                    json_augmented_image_path = augmented_image_save_path_disk.replace("\\", "/")

                    aug_record = {
                        "id": f"{CLASS_MAPPING[target_class_label]}_{augmented_filename}",
                        "image_path": json_augmented_image_path,
                        "question": FIXED_QUESTION,
                        "answer": target_class_label
                    }
                    augmented_records.append(aug_record)
                except Exception as e_aug:
                    logging.error(f"'{original_image_path}' 증강 중 오류 (증강 {i+1}번째): {e_aug}")

    logging.info(f"클래스 '{target_class_label}'의 원본 이미지 {num_processed_for_aug}개에 대해 총 {len(augmented_records)}개의 증강 이미지 생성 완료.")
    return augmented_records

def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    logging.info("1. 원본 데이터 레코드 수집 중...")
    all_original_records = collect_original_image_records(RAW_DATA_DIR, CLASS_MAPPING)
    if not all_original_records:
        logging.error("원본 이미지 정보를 찾을 수 없습니다. 스크립트를 종료합니다.")
        return

    original_labels = [r['answer'] for r in all_original_records]
    if not original_labels: # stratify를 위해 레이블이 하나라도 있어야 함
        logging.error("수집된 레코드에 레이블 정보가 없습니다. 분할을 진행할 수 없습니다.")
        return

    # 클래스별 최소 샘플 수 확인 (stratify 가능하도록)
    from collections import Counter
    label_counts = Counter(original_labels)
    min_samples_for_split = 2 # train_test_split이 각 클래스에 대해 최소 1개씩은 배정하려고 함 (test_size < 1일때)

    # 분할 전 데이터가 너무 적은 클래스가 있는지 확인
    valid_for_stratify = True
    for label, count in label_counts.items():
        if count < min_samples_for_split * 2 : # 대략 (학습+검증)과 테스트, 그리고 학습과 검증으로 나눌것을 고려
            logging.warning(f"레이블 '{label}'의 샘플 수가 너무 적습니다 ({count}개). Stratified split이 정확하지 않거나 오류를 발생시킬 수 있습니다.")
            # 이 경우 stratify 옵션을 빼거나, 해당 클래스 데이터를 더 모아야 함.
            # 여기서는 일단 진행하되, train_test_split에서 오류 발생 가능.

    logging.info(f"2. 원본 데이터를 (학습+검증)셋과 테스트셋으로 분할 (테스트 비율: {TEST_SPLIT_RATIO})...")
    try:
        train_val_original_records, test_records = train_test_split(
            all_original_records,
            test_size=TEST_SPLIT_RATIO,
            random_state=RANDOM_SEED,
            stratify=original_labels if valid_for_stratify else None # 샘플 적으면 stratify 없이
        )
    except ValueError as e_split1:
        logging.error(f"테스트셋 분할 중 오류 (샘플 부족 가능성): {e_split1}. Stratify 없이 재시도합니다.")
        train_val_original_records, test_records = train_test_split(
            all_original_records, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )

    test_json_path = os.path.join(OUTPUT_DATA_DIR, TEST_JSON_FILENAME)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=4)
    logging.info(f"테스트셋 저장 완료: {test_json_path} ({len(test_records)}개)")

    train_val_labels = [r['answer'] for r in train_val_original_records]
    if not train_val_labels : # (학습+검증)셋이 비어있으면 더이상 진행 불가
        logging.error("(학습+검증)셋이 비어있어 학습/검증셋 분할을 진행할 수 없습니다.")
        return

    # (학습+검증)셋의 클래스별 최소 샘플 수 확인
    train_val_label_counts = Counter(train_val_labels)
    valid_for_stratify_tv = True
    for label, count in train_val_label_counts.items():
        if count < min_samples_for_split:
            valid_for_stratify_tv = False
            logging.warning(f"레이블 '{label}'의 (학습+검증)셋 내 샘플 수가 너무 적습니다 ({count}개). Stratified split이 정확하지 않거나 오류를 발생시킬 수 있습니다.")


    logging.info(f"3. (학습+검증)셋을 학습셋과 검증셋으로 분할 (검증 비율: {VAL_SPLIT_FROM_REMAINING_RATIO})...")
    try:
        train_original_records, val_records = train_test_split(
            train_val_original_records,
            test_size=VAL_SPLIT_FROM_REMAINING_RATIO,
            random_state=RANDOM_SEED,
            stratify=train_val_labels if valid_for_stratify_tv else None
        )
    except ValueError as e_split2:
        logging.error(f"학습/검증셋 분할 중 오류 (샘플 부족 가능성): {e_split2}. Stratify 없이 재시도합니다.")
        train_original_records, val_records = train_test_split(
            train_val_original_records, test_size=VAL_SPLIT_FROM_REMAINING_RATIO, random_state=RANDOM_SEED
        )


    val_json_path = os.path.join(OUTPUT_DATA_DIR, VAL_JSON_FILENAME)
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_records, f, ensure_ascii=False, indent=4)
    logging.info(f"검증셋 저장 완료: {val_json_path} ({len(val_records)}개)")

    logging.info(f"4. 학습 데이터에 대한 증강 처리 (클래스: '{TARGET_AUG_CLASS_LABEL_STR}')...")
    # train_original_records (순수 원본 학습 데이터)에 대해서만 증강 수행
    augmented_image_records = augment_and_save_images(
        list(train_original_records), # 원본 학습 레코드의 복사본 전달
        TARGET_AUG_CLASS_LABEL_STR,
        NUM_AUGMENTATIONS_PER_IMAGE,
        transform_augment
    )

    # 원본 학습 레코드와 증강된 레코드를 합쳐 최종 학습 데이터셋 구성
    final_train_records = list(train_original_records) + augmented_image_records
    random.shuffle(final_train_records) # 최종 학습 데이터 순서 섞기
    logging.info(f"최종 학습셋 (원본+증강) 생성 완료: {len(final_train_records)}개.")

    train_json_path = os.path.join(OUTPUT_DATA_DIR, TRAIN_JSON_FILENAME)
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_train_records, f, ensure_ascii=False, indent=4)
    logging.info(f"학습 데이터(증강 포함) JSON 파일 저장 완료: {train_json_path}")

    logging.info("모든 데이터 준비 과정 완료!")
    logging.info(f"요약: 학습셋 {len(final_train_records)}개, 검증셋 {len(val_records)}개, 테스트셋 {len(test_records)}개.")

if __name__ == "__main__":
    main()
