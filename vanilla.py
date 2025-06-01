# evaluate_zeroshot_llava16.py

# --- 0. 환경 변수 설정 ---
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# --- 1. 필요한 라이브러리 가져오기 ---
import json
import random
from typing import Any, Dict, List
import sys
import logging
import re
from tqdm import tqdm
from functools import partial

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
# PEFT는 제로샷 평가에는 직접적으로 필요 없으므로 주석 처리 또는 삭제 가능
# from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
import numpy as np

# --- 로깅 설정 ---
logger = logging.getLogger("ZeroShotEvaluator")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

# --- 2. 평가 설정값 정의 ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
TEST_DATA_JSON_PATH = "data/test_drug_data.json"

IMAGE_BASE_DIRECTORY = "."
FIXED_QUESTION = "이 이미지는 다음 중 어떤 유형입니까? (0: 관련 없음, 1: 마약 대화 내역 사진, 2: 마약 거래 장소, 3: 마약 원본)"
MAX_TEXT_LENGTH = 256
ZERO_SHOT_ANSWER_MAX_LENGTH = 50
BATCH_SIZE_EVAL = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS_STR_LIST = ["0", "1", "2", "3", "-1"]
CLASS_NAMES = ["0:관련없음", "1:마약대화", "2:거래장소", "3:마약원본", "분류불가"]
OUTPUT_DIR = "evaluation_results_zeroshot"

# --- 한글 폰트 설정 함수 ---
def set_korean_font():
    font_path = None
    possible_font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "c:/Windows/Fonts/NanumGothic.ttf", # Windows 예시
        # 현재 스크립트 폴더에 NanumGothic.ttf가 있다면 아래 경로 사용 가능
        # "./NanumGothic.ttf"
    ]
    for path in possible_font_paths:
        if os.path.exists(path):
            font_path = path
            break

    if font_path:
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            mpl.rc('font', family=font_name)
            mpl.rcParams['axes.unicode_minus'] = False
            logger.info(f"Matplotlib에 한글 폰트 '{font_name}' 설정 완료 (경로: {font_path}).")
        except Exception as e_font:
            logger.warning(f"한글 폰트 로드 중 오류 ({font_path}): {e_font}. 기본 폰트 사용.")
    else:
        logger.warning("나눔고딕 또는 지정된 한글 폰트를 찾을 수 없습니다. 그래프의 한글 레이블이 깨질 수 있습니다.")
        logger.info("해결 방법: 시스템에 한글 폰트(예: sudo apt-get install fonts-nanum)를 설치하거나, .ttf 파일을 제공하고 경로를 지정해주세요.")

# --- 3. 데이터셋 클래스 정의 (재사용) ---
class DrugImageDatasetEval(Dataset):
    def __init__(self, json_file_path: str, image_base_dir: str, question_text: str, split_name: str = "test"):
        super().__init__()
        self.image_base_dir = image_base_dir; self.question_text = question_text
        self.dataset_list = []; self.dataset_length = 0
        logger.info(f"'{split_name}' 데이터셋 로딩: {json_file_path}")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f: self.dataset_list = json.load(f)
            self.dataset_length = len(self.dataset_list)
            logger.info(f"'{split_name}' 데이터셋 로드 완료. 샘플 수: {self.dataset_length}")
        except Exception as e: logger.error(f"'{split_name}' 데이터셋 로딩 오류 ({json_file_path}): {e}", exc_info=True); raise

    def __len__(self) -> int: return self.dataset_length
    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset_list[idx]; relative_image_path = item.get("image_path")
        answer_text = item.get("answer"); image_id = item.get("id", f"sample_{idx}"); image = None
        try:
            full_image_path = os.path.join(self.image_base_dir, relative_image_path) if self.image_base_dir != "." else relative_image_path
            full_image_path = full_image_path.replace("\\", "/")
            if os.path.exists(full_image_path): image = Image.open(full_image_path).convert("RGB")
            else: logger.warning(f"이미지 파일 없음 (테스트셋 인덱스 {idx}, 경로 {full_image_path}). 더미 이미지 사용."); image = Image.new('RGB', (336, 336), (0,0,0))
        except Exception as e_img: logger.warning(f"테스트 이미지 로드 실패 (인덱스 {idx}, 경로 {relative_image_path}): {e_img}. 더미 이미지 사용."); image = Image.new('RGB', (336, 336), (0,0,0))
        return {"image": image, "question": self.question_text, "answer": answer_text, "id": image_id, "original_image_path": relative_image_path}

# --- 4. Collate 함수 정의 ---
IMAGE_TOKEN_PLACEHOLDER_EVAL = "<image>"
PROMPT_FORMAT_FOR_EVAL  = f"USER: {IMAGE_TOKEN_PLACEHOLDER_EVAL}\n{{question}}\nASSISTANT:"
processor_zeroshot = None

def llava_collate_fn_for_zeroshot_eval(batch_samples: List[Dict], max_text_len_arg: int):
    global processor_zeroshot
    if processor_zeroshot is None: raise ValueError("글로벌 변수 'processor_zeroshot'가 설정되지 않았습니다.")

    images = [sample["image"] for sample in batch_samples]; questions = [sample["question"] for sample in batch_samples]
    ground_truth_answers = [sample["answer"] for sample in batch_samples]; ids_batch = [sample["id"] for sample in batch_samples]
    original_paths_batch = [sample["original_image_path"] for sample in batch_samples]
    current_image_token = getattr(processor_zeroshot, 'image_token', IMAGE_TOKEN_PLACEHOLDER_EVAL)

    prompt_texts = [PROMPT_FORMAT_FOR_EVAL.replace(IMAGE_TOKEN_PLACEHOLDER_EVAL, current_image_token).format(question=q) for q in questions]
    inputs = processor_zeroshot(
        text=prompt_texts, images=images, padding="longest", truncation=True, max_length=max_text_len_arg, return_tensors="pt"
    )
    if hasattr(images[0], 'height') and hasattr(images[0], 'width'):
        inputs["image_sizes"] = torch.tensor([[img.height, img.width] for img in images])

    inputs["ground_truth_answers"] = ground_truth_answers; inputs["ids"] = ids_batch; inputs["original_image_paths"] = original_paths_batch
    return dict(inputs)

# --- 5. 제로샷 답변 파싱 함수 ---
def parse_llava_zeroshot_output(text_output: str) -> str:
    text_output_lower = text_output.lower()
    match_num = re.search(r'\b([0-3])\b', text_output)
    if match_num: return match_num.group(1)

    keywords_class3 = ["마약 원본", "마약 사진", "약물 사진", "drug itself", "raw drug", "마약으로 보입니다", "약물로 보입니다"]
    if any(keyword in text_output_lower for keyword in keywords_class3): return "3"

    keywords_class1 = ["대화 내역", "채팅 내용", "메시지", "카카오톡", "텔레그램", "문자 내용", "conversation", "chat log"]
    if any(keyword in text_output_lower for keyword in keywords_class1):
        if "마약" in text_output_lower or "drug" in text_output_lower : return "1"

    keywords_class2 = ["거래 장소", "만나는 장소", "약속 장소", "위치 정보", "지도 사진", "transaction place", "location"]
    if any(keyword in text_output_lower for keyword in keywords_class2):
        if "마약" in text_output_lower or "drug" in text_output_lower : return "2"

    keywords_class0_or_unknown = [
        "관련 없", "아닌 것 같", "해당하지 않", "다른 종류", "모르겠", "판단 불가",
        "정보 부족", "확인할 수 없", "애매", "not related", "cannot determine",
        "not sure", "insufficient information"
    ]
    if any(keyword in text_output_lower for keyword in keywords_class0_or_unknown): return "0"

    if "마약" in text_output_lower or "drug" in text_output_lower: return "-1" # 명확히 마약 언급 있지만 분류 안될시
    return "0" # 기본적으로 관련 없음 처리

# --- 6. 메인 평가 함수 ---
def evaluate_zeroshot_model():
    global processor_zeroshot
    set_korean_font()

    try:
        import transformers, tokenizers
        logger.info(f"사용 중인 Transformers 버전: {transformers.__version__}")
        logger.info(f"사용 중인 Tokenizers 버전: {tokenizers.__version__}")
        RECOMMENDED_TRANSFORMERS_VERSION = "4.44.0" # Colab 성공 버전
        RECOMMENDED_TOKENIZERS_VERSION = "0.19.1" # Colab 성공 버전
        # (버전 확인 및 경고 로직 - 이전 코드 참조, 필요시 추가)
    except ImportError: logger.error("Transformers 또는 Tokenizers 라이브러리 설치 필요."); return

    logger.info(f"제로샷 평가: 프로세서 로딩: {MODEL_ID}")
    try:
        # 제로샷이므로 FINETUNED_MODEL_PATH 대신 MODEL_ID에서 직접 로드
        processor_zeroshot = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, force_download=True)
        processor_zeroshot.tokenizer.padding_side = "right"
        if processor_zeroshot.tokenizer.pad_token is None or processor_zeroshot.tokenizer.pad_token_id is None:
            logger.warning(f"프로세서의 PAD 토큰이 없음. EOS를 PAD로 사용합니다.")
            processor_zeroshot.tokenizer.pad_token = processor_zeroshot.tokenizer.eos_token
            processor_zeroshot.tokenizer.pad_token_id = processor_zeroshot.tokenizer.eos_token_id
        logger.info(f"프로세서 PAD 토큰 ID: {processor_zeroshot.tokenizer.pad_token_id}")
    except Exception as e: logger.error(f"프로세서 로딩 중 오류: {e}", exc_info=True); return

    logger.info(f"제로샷 평가: 베이스 모델 로딩: {MODEL_ID}")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=model_dtype
    )
    model = None # model 변수 초기화
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=model_dtype,
            quantization_config=quantization_config, low_cpu_mem_usage=True,
            trust_remote_code=True, force_download=True
        )
        logger.info(f"베이스 모델 ({MODEL_ID}) 로드 완료.")

        model.eval()
        # model.to(DEVICE) # <<< 이 라인 제거됨 (양자화 모델은 자동 배치)
        logger.info(f"최종 평가 모델 준비 완료. 모델 device: {model.device}") # 로드된 모델의 실제 장치 확인
    except Exception as e:
        logger.error(f"모델 로딩 중 오류: {e}", exc_info=True)
        return

    if model is None: # 모델 로드 실패 시 여기서 종료
        logger.error("모델 객체가 성공적으로 로드되지 않았습니다.")
        return

    try:
        test_dataset = DrugImageDatasetEval(TEST_DATA_JSON_PATH, IMAGE_BASE_DIRECTORY, FIXED_QUESTION)
        if len(test_dataset) == 0: logger.error("테스트 데이터셋에 샘플이 없습니다. 종료합니다."); return
        collate_fn_with_args = partial(llava_collate_fn_for_zeroshot_eval, max_text_len_arg=MAX_TEXT_LENGTH)
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_with_args,
            shuffle=False, num_workers=0 # 제로샷은 보통 num_workers=0으로 해도 무방
        )
    except Exception as e: logger.error(f"테스트 데이터 로더 준비 중 오류: {e}", exc_info=True); return

    all_predictions = []; all_ground_truths = []; all_ids_eval = []; mismatched_predictions_eval = []
    logger.info("제로샷 테스트 데이터셋 평가 시작...")

    # 모델이 실제로 할당된 장치를 사용 (DEVICE 변수 대신)
    current_model_device = model.device

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating Zero-Shot Test Set"):
            input_ids = batch.get("input_ids").to(current_model_device)
            attention_mask = batch.get("attention_mask").to(current_model_device)
            pixel_values = batch.get("pixel_values").to(dtype=model.dtype, device=current_model_device)
            image_sizes = batch.get("image_sizes").to(current_model_device) if batch.get("image_sizes") is not None else None
            ground_truths_batch = batch.get("ground_truth_answers"); ids_batch = batch.get("ids")
            original_paths_batch = batch.get("original_image_paths")
            pad_token_id_to_use = processor_zeroshot.tokenizer.pad_token_id if processor_zeroshot.tokenizer.pad_token_id is not None else processor_zeroshot.tokenizer.eos_token_id

            generated_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_sizes=image_sizes,
                max_new_tokens=ZERO_SHOT_ANSWER_MAX_LENGTH,
                eos_token_id=processor_zeroshot.tokenizer.eos_token_id,
                pad_token_id=pad_token_id_to_use, do_sample=False
            )

            for i in range(generated_ids.shape[0]):
                actual_input_len_tensor = input_ids[i].ne(pad_token_id_to_use)
                actual_input_len = actual_input_len_tensor.sum().item() if actual_input_len_tensor.ndim > 0 else (actual_input_len_tensor.item() if actual_input_len_tensor.sum().item() > 0 else 0)
                decoded_prediction_full = processor_zeroshot.decode(generated_ids[i, actual_input_len:], skip_special_tokens=True).strip()
                predicted_label = parse_llava_zeroshot_output(decoded_prediction_full)
                true_label = ground_truths_batch[i]; sample_id = ids_batch[i]
                all_predictions.append(predicted_label); all_ground_truths.append(true_label); all_ids_eval.append(sample_id)
                if predicted_label != true_label:
                    mismatched_predictions_eval.append({
                        "id": sample_id, "image_path": original_paths_batch[i], "true_label": true_label,
                        "predicted_label": predicted_label, "raw_model_output": decoded_prediction_full
                    })
    logger.info("제로샷 평가 완료.")

    if mismatched_predictions_eval:
        logger.info(f"\n--- 잘못 예측된 샘플 (제로샷, 상위 {min(10, len(mismatched_predictions_eval))}개) ---")
        for i, mismatch in enumerate(mismatched_predictions_eval[:10]):
            logger.info(f"{i+1}. ID: {mismatch['id']}, Path: {mismatch['image_path']}, 정답: {mismatch['true_label']}, 예측: {mismatch['predicted_label']}, 모델 출력: '{mismatch['raw_model_output']}'")

    if all_ground_truths and all_predictions:
        accuracy = accuracy_score(all_ground_truths, all_predictions)
        logger.info(f"\n--- 최종 제로샷 평가 결과 ---"); logger.info(f"전체 정확도 (Accuracy): {accuracy:.4f}")
        logger.info("\n--- Classification Report (Zero-Shot) ---")
        report = classification_report(all_ground_truths, all_predictions, labels=CLASS_LABELS_STR_LIST, target_names=CLASS_NAMES, zero_division=0)
        print(report)
    else: logger.warning("예측 또는 정답 리스트가 비어있어 평가 지표를 계산할 수 없습니다.")

    if all_ground_truths and all_predictions:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cm = confusion_matrix(all_ground_truths, all_predictions, labels=CLASS_LABELS_STR_LIST)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
        plt.xlabel('Predicted Labels', fontsize=14); plt.ylabel('True Labels', fontsize=14)
        plt.title('Confusion Matrix (Zero-Shot LLaVA 1.6)', fontsize=16); plt.xticks(fontsize=12, rotation=45, ha="right"); plt.yticks(fontsize=12)
        plt.tight_layout()
        confusion_matrix_filename = os.path.join(OUTPUT_DIR, "confusion_matrix_zeroshot_llava16.png")
        try: plt.savefig(confusion_matrix_filename); logger.info(f"혼동 행렬 이미지 저장 완료: {confusion_matrix_filename}")
        except Exception as e_plot: logger.error(f"혼동 행렬 이미지 저장 실패: {e_plot}")
        plt.close()
    else: logger.warning("예측 또는 정답 리스트가 비어있어 혼동 행렬을 생성할 수 없습니다.")

# --- 스크립트 실행 지점 ---
if __name__ == '__main__':
    evaluate_zeroshot_model()
