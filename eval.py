# evaluate_script.py

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
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
import numpy as np

# --- 로깅 설정 ---
logger = logging.getLogger("DrugEvaluator") # 로거 이름 변경
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

# --- 2. 평가 설정값 정의 ---
BASE_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
# 학습 스크립트의 OUTPUT_MODEL_DIR와 맞추고, 그 하위의 final_model_checkpoint 사용
FINETUNED_MODEL_PATH = "outputs/llava_next_drug_classifier_epochs5/final_model_checkpoint"
TEST_DATA_JSON_PATH = "data/test_drug_data.json"

IMAGE_BASE_DIRECTORY = "."
FIXED_QUESTION = "이 이미지는 다음 중 어떤 유형입니까? (0: 관련 없음, 1: 마약 대화 내역 사진, 2: 마약 거래 장소, 3: 마약 원본)"
MAX_TEXT_LENGTH = 256 # Collate 함수에 전달될 값
ANSWER_MAX_LENGTH_EVAL = 8
BATCH_SIZE_EVAL = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS_STR_LIST = ["0", "1", "2", "3"]
CLASS_NAMES = ["0:관련없음", "1:마약대화", "2:거래장소", "3:마약원본"]
EVAL_OUTPUT_DIR = "evaluation_results_epochs5" # 평가 결과 저장 폴더명

# --- 한글 폰트 설정 ---
def set_korean_font():
    font_path = None
    # 시스템에 설치된 나눔고딕 경로 탐색 (일반적인 경로)
    possible_font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "c:/Windows/Fonts/NanumGothic.ttf", # Windows 예시
        # 다른 시스템 경로 또는 직접 다운로드한 .ttf 파일 경로 추가 가능
        # 예: "./NanumGothic.ttf" (스크립트와 같은 폴더에 폰트 파일이 있는 경우)
    ]
    for path in possible_font_paths:
        if os.path.exists(path):
            font_path = path
            break

    if font_path:
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            mpl.rc('font', family=font_name)
            mpl.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
            logger.info(f"Matplotlib에 한글 폰트 '{font_name}' 설정 완료 (경로: {font_path}).")
        except Exception as e_font:
            logger.warning(f"한글 폰트 로드 중 오류 ({font_path}): {e_font}. 기본 폰트 사용.")
    else:
        logger.warning("나눔고딕 또는 지정된 한글 폰트를 찾을 수 없습니다. 그래프의 한글 레이블이 깨질 수 있습니다.")
        logger.info("해결 방법: 시스템에 한글 폰트(예: fonts-nanum)를 설치하거나, .ttf 파일을 제공하고 경로를 지정해주세요.")

# --- 3. 데이터셋 클래스 정의 ---
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
processor_eval = None # 평가 함수에서 로드될 전역 프로세서 객체

def llava_collate_fn_for_evaluation(batch_samples: List[Dict], max_text_len_arg: int):
    global processor_eval
    if processor_eval is None: raise ValueError("글로벌 변수 'processor_eval'이 설정되지 않음.")

    images = [sample["image"] for sample in batch_samples]; questions = [sample["question"] for sample in batch_samples]
    ground_truth_answers = [sample["answer"] for sample in batch_samples]; ids_batch = [sample["id"] for sample in batch_samples]
    original_paths_batch = [sample["original_image_path"] for sample in batch_samples]
    current_image_token = getattr(processor_eval, 'image_token', IMAGE_TOKEN_PLACEHOLDER_EVAL)

    prompt_texts = [PROMPT_FORMAT_FOR_EVAL.replace(IMAGE_TOKEN_PLACEHOLDER_EVAL, current_image_token).format(question=q) for q in questions]
    inputs = processor_eval(
        text=prompt_texts, images=images, padding="longest", truncation=True, max_length=max_text_len_arg, return_tensors="pt"
    )
    if hasattr(images[0], 'height') and hasattr(images[0], 'width'):
        inputs["image_sizes"] = torch.tensor([[img.height, img.width] for img in images])
    inputs["ground_truth_answers"] = ground_truth_answers; inputs["ids"] = ids_batch; inputs["original_image_paths"] = original_paths_batch
    return dict(inputs)

# --- 5. 메인 평가 함수 ---
def evaluate_model():
    global processor_eval
    set_korean_font() # 한글 폰트 설정 시도

    try:
        import transformers, tokenizers
        logger.info(f"사용 중인 Transformers 버전: {transformers.__version__}")
        logger.info(f"사용 중인 Tokenizers 버전: {tokenizers.__version__}")
    except ImportError: logger.error("Transformers 또는 Tokenizers 라이브러리 설치 필요."); return

    logger.info(f"프로세서 로딩 경로: {FINETUNED_MODEL_PATH}")
    try:
        processor_eval = AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
        processor_eval.tokenizer.padding_side = "right"
        if processor_eval.tokenizer.pad_token is None or processor_eval.tokenizer.pad_token_id is None:
            logger.warning(f"프로세서 PAD 토큰이 없음. EOS를 PAD로 사용.")
            processor_eval.tokenizer.pad_token = processor_eval.tokenizer.eos_token
            processor_eval.tokenizer.pad_token_id = processor_eval.tokenizer.eos_token_id
        logger.info(f"프로세서 PAD 토큰 ID: {processor_eval.tokenizer.pad_token_id}")
    except Exception as e: logger.error(f"프로세서 로딩 중 오류: {e}", exc_info=True); return

    logger.info(f"베이스 모델 로딩: {BASE_MODEL_ID}")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=model_dtype)
    try:
        base_model_for_eval = LlavaNextForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=model_dtype, quantization_config=quantization_config,
            low_cpu_mem_usage=True, trust_remote_code=True
        )
        logger.info(f"베이스 모델 ({BASE_MODEL_ID}) 로드 완료.")
        logger.info(f"Fine-tuned LoRA 어댑터 로딩: {FINETUNED_MODEL_PATH}")
        model = PeftModel.from_pretrained(base_model_for_eval, FINETUNED_MODEL_PATH)
        logger.info("LoRA 어댑터 적용 완료.")
        model.eval(); model.to(DEVICE); logger.info("최종 평가 모델 준비 완료.")
    except Exception as e: logger.error(f"모델 로딩 또는 PEFT 적용 중 오류: {e}", exc_info=True); return

    try:
        test_dataset = DrugImageDatasetEval(TEST_DATA_JSON_PATH, IMAGE_BASE_DIRECTORY, FIXED_QUESTION)
        if len(test_dataset) == 0: logger.error("테스트 데이터셋에 샘플이 없습니다. 종료합니다."); return
        collate_fn_with_args = partial(llava_collate_fn_for_evaluation, max_text_len_arg=MAX_TEXT_LENGTH)
        test_dataloader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_with_args,
            shuffle=False, num_workers=2, pin_memory
