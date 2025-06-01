# train_script.py

# --- 0. 환경 변수 및 기본 설정 ---
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
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import logging
import re
from pytorch_lightning.callbacks import ModelCheckpoint # 명시적 체크포인트 콜백 사용

# --- 로깅 설정 ---
logger = logging.getLogger("DrugTrainer") # 로거 이름 변경
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.setLevel(logging.INFO)

# --- 2. 주요 설정값 정의 ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
TRAIN_DATA_JSON_PATH = "data/train_drug_data.json"
VAL_DATA_JSON_PATH = "data/val_drug_data.json"
OUTPUT_MODEL_DIR = "outputs/llava_next_drug_classifier_epochs5" # 에폭 수 반영

MAX_TEXT_LENGTH = 256
ANSWER_MAX_LENGTH = 8
MAX_EPOCHS = 5 # 에폭 수를 5로 변경
BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 16
LEARNING_RATE = 1e-5
GRADIENT_CLIP_VAL = 1.0
USE_QLORA = True
IMAGE_BASE_DIRECTORY = "."

FIXED_QUESTION = "이 이미지는 다음 중 어떤 유형입니까? (0: 관련 없음, 1: 마약 대화 내역 사진, 2: 마약 거래 장소, 3: 마약 원본)"
CLASS_LABELS_LIST = ["0", "1", "2", "3"]

# --- 체크포인트 설정 ---
# 이어할 체크포인트 파일 경로. 처음 실행 시 None.
# 예: "outputs/llava_next_drug_classifier_epochs5/lightning_logs/version_0/checkpoints/last.ckpt"
RESUME_CKPT_PATH = "outputs/llava_next_drug_classifier_epochs5/checkpoints/last.ckpt"# <--- 학습 이어하기 시 이 부분을 실제 .ckpt 파일 경로로 수정하세요.

# --- 3. 전역 변수 ---
processor_main = None
model_main = None
train_drug_dataset = None
val_drug_dataset = None

# --- 4. 데이터셋 클래스 정의 (`DrugImageDataset`) ---
class DrugImageDataset(Dataset):
    def __init__(self, json_file_path: str, image_base_dir: str, question_text: str, split_name: str = "all"):
        super().__init__()
        self.image_base_dir = image_base_dir
        self.question_text = question_text
        self.dataset_list = []
        self.dataset_length = 0
        logger.info(f"'{split_name}' 데이터셋 로딩: {json_file_path}")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                self.dataset_list = json.load(f)
            self.dataset_length = len(self.dataset_list)
            logger.info(f"'{split_name}' 데이터셋 로드 완료. 샘플 수: {self.dataset_length}")
        except FileNotFoundError:
            logger.error(f"'{split_name}' 데이터셋 파일({json_file_path})을 찾을 수 없습니다.", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"'{split_name}' 데이터셋 로딩 중 예기치 않은 오류 ({json_file_path}): {e}", exc_info=True)
            raise

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset_list[idx]
        relative_image_path = item.get("image_path")
        answer_text = item.get("answer")
        image = None
        try:
            full_image_path = os.path.join(self.image_base_dir, relative_image_path) if self.image_base_dir != "." else relative_image_path
            full_image_path = full_image_path.replace("\\", "/")
            if os.path.exists(full_image_path):
                image = Image.open(full_image_path).convert("RGB")
            else:
                logger.warning(f"이미지 파일 없음: {full_image_path} (인덱스 {idx})")
        except Exception as e_img:
            logger.error(f"이미지 로드 실패 (인덱스 {idx}, 경로 {relative_image_path}): {e_img}. 더미 이미지 사용 시도.")
            # 더미 이미지 생성 로직은 아래 if image is None: 에서 처리

        if image is None: # 이미지 로드 실패 또는 경로/데이터 누락 시
            image = Image.new('RGB', (336, 336), (random.randint(0,20), random.randint(0,20), random.randint(0,20)))
            original_answer = answer_text # 원래 답변은 유지하되, 문제가 있음을 인지
            answer_text = random.choice(CLASS_LABELS_LIST) if answer_text not in CLASS_LABELS_LIST else answer_text # answer_text가 유효하지 않으면 랜덤 할당
            logger.warning(f"더미 이미지 사용됨 (인덱스 {idx}, 원래 경로: {relative_image_path}). 원래 답변: {original_answer}, 사용된 답변: {answer_text}")

        return {"image": image, "question": self.question_text, "answer": answer_text, "original_image_path": relative_image_path}

# --- 5. LLaVA 프롬프트 형식 및 Collate 함수 정의 ---
IMAGE_TOKEN_PLACEHOLDER = "<image>"
PROMPT_FORMAT_TRAIN = f"USER: {IMAGE_TOKEN_PLACEHOLDER}\n{{question}}\nASSISTANT: {{answer}}"
PROMPT_FORMAT_EVAL  = f"USER: {IMAGE_TOKEN_PLACEHOLDER}\n{{question}}\nASSISTANT:"

# Collate 함수 내 프로세서 출력 검증 (기본값 False로, 필요시 True로 변경하여 디버깅)
CHECK_PROCESSOR_OUTPUT = False

def drug_collate_fn_train(batch_samples: List[Dict]):
    global processor_main, model_main
    if processor_main is None: raise ValueError("Processor가 초기화되지 않았습니다.")
    images = [sample["image"] for sample in batch_samples]
    questions = [sample["question"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]
    current_image_token_str = getattr(processor_main, 'image_token', IMAGE_TOKEN_PLACEHOLDER)

    full_texts = [
        PROMPT_FORMAT_TRAIN.replace(IMAGE_TOKEN_PLACEHOLDER, current_image_token_str).format(question=q, answer=a) + processor_main.tokenizer.eos_token
        for q, a in zip(questions, answers)
    ]
    try:
        inputs = processor_main(
            text=full_texts, images=images,
            padding="longest", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Processor_main 처리 중 오류 (학습 콜레이트): {e}", exc_info=True)
        raise

    if CHECK_PROCESSOR_OUTPUT and not hasattr(drug_collate_fn_train, 'has_logged_train'):
        # (디버깅 로그 부분 - 필요시 활성화)
        logger.info("--- DEBUG drug_collate_fn_train (첫 배치) ---")
        # ... (이전 디버깅 로그 상세 내용) ...
        drug_collate_fn_train.has_logged_train = True

    labels = inputs.input_ids.clone()
    if processor_main.tokenizer.pad_token_id is not None:
        labels[labels == processor_main.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    if hasattr(images[0], 'height') and hasattr(images[0], 'width'):
        inputs["image_sizes"] = torch.tensor([[img.height, img.width] for img in images])
    return dict(inputs) # BatchFeature.to() 문제 회피 위해 dict로 반환

def drug_collate_fn_eval(batch_samples: List[Dict]):
    global processor_main, model_main
    if processor_main is None: raise ValueError("Processor가 초기화되지 않았습니다.")
    images = [sample["image"] for sample in batch_samples]
    questions = [sample["question"] for sample in batch_samples]
    ground_truth_answers = [sample["answer"] for sample in batch_samples]
    original_image_paths_batch = [sample["original_image_path"] for sample in batch_samples]
    current_image_token_str = getattr(processor_main, 'image_token', IMAGE_TOKEN_PLACEHOLDER)

    prompt_texts = [
        PROMPT_FORMAT_EVAL.replace(IMAGE_TOKEN_PLACEHOLDER, current_image_token_str).format(question=q)
        for q in questions
    ]
    try:
        inputs = processor_main(
            text=prompt_texts, images=images,
            padding="longest", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Processor_main 처리 중 오류 (검증 콜레이트): {e}", exc_info=True)
        raise

    if CHECK_PROCESSOR_OUTPUT and not hasattr(drug_collate_fn_eval, 'has_logged_eval'):
        # (디버깅 로그 부분 - 필요시 활성화)
        logger.info("--- DEBUG drug_collate_fn_eval (첫 배치) ---")
        # ... (이전 디버깅 로그 상세 내용) ...
        drug_collate_fn_eval.has_logged_eval = True

    inputs["ground_truth_answers"] = ground_truth_answers
    inputs["original_image_paths"] = original_image_paths_batch
    if hasattr(images[0], 'height') and hasattr(images[0], 'width'):
        inputs["image_sizes"] = torch.tensor([[img.height, img.width] for img in images])
    return dict(inputs) # BatchFeature.to() 문제 회피 위해 dict로 반환

# --- 6. PyTorch Lightning 모듈 정의 ---
class LlavaClassifierPLModule(L.LightningModule):
    def __init__(self, model_instance, processor_instance, learning_rate_cfg, batch_size_cfg_for_log):
        super().__init__()
        self.model = model_instance
        self.processor = processor_instance
        self.learning_rate = learning_rate_cfg
        self.batch_size_for_logging = batch_size_cfg_for_log
        self.save_hyperparameters(ignore=['model_instance', 'processor_instance'])

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pixel_values_processed = batch.get("pixel_values").to(dtype=self.model.dtype, device=self.device)
        input_ids_processed = batch.get("input_ids").to(self.device)
        attention_mask_processed = batch.get("attention_mask").to(self.device)
        labels_processed = batch.get("labels").to(self.device)
        image_sizes_processed = batch.get("image_sizes").to(self.device) if batch.get("image_sizes") is not None else None

        outputs = self.model(
            input_ids=input_ids_processed,
            attention_mask=attention_mask_processed,
            pixel_values=pixel_values_processed,
            labels=labels_processed,
            image_sizes=image_sizes_processed
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size_for_logging)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        input_ids = batch.get("input_ids").to(self.device)
        attention_mask = batch.get("attention_mask").to(self.device)
        pixel_values_processed = batch.get("pixel_values").to(dtype=self.model.dtype, device=self.device)
        image_sizes = batch.get("image_sizes").to(self.device) if batch.get("image_sizes") is not None else None
        ground_truth_answers = batch.get("ground_truth_answers")

        pad_token_id_to_use = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id

        generated_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values_processed, image_sizes=image_sizes,
            max_new_tokens=ANSWER_MAX_LENGTH,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=pad_token_id_to_use
        )

        correct_predictions = 0
        total_predictions = generated_ids.shape[0]
        for i in range(total_predictions):
            actual_input_len_tensor = input_ids[i].ne(pad_token_id_to_use)
            actual_input_len = actual_input_len_tensor.sum().item() if actual_input_len_tensor.ndim > 0 else (actual_input_len_tensor.item() if actual_input_len_tensor.sum().item() > 0 else 0)
            decoded_prediction_full = self.processor.decode(generated_ids[i, actual_input_len:], skip_special_tokens=True).strip()
            match = re.search(r'\b([0-3])\b', decoded_prediction_full)
            predicted_label = match.group(1) if match else "INVALID_OUTPUT"
            true_label = ground_truth_answers[i]

            if predicted_label == true_label: correct_predictions += 1
            if batch_idx == 0 and i < 2 and hasattr(self, 'global_rank') and self.global_rank == 0:
                logger.info(f"\nVal Sample - True: '{true_label}', Raw Pred: '{decoded_prediction_full}', Extracted Pred: '{predicted_label}'")

        self.log("val_correct_preds", torch.tensor(correct_predictions, dtype=torch.float, device=self.device), on_step=False, on_epoch=True, reduce_fx="sum")
        self.log("val_total_preds", torch.tensor(total_predictions, dtype=torch.float, device=self.device), on_step=False, on_epoch=True, reduce_fx="sum")

    def on_validation_epoch_end(self):
        correct_preds = self.trainer.callback_metrics.get("val_correct_preds", torch.tensor(0.0, device=self.device))
        total_preds = self.trainer.callback_metrics.get("val_total_preds", torch.tensor(0.0, device=self.device))
        if hasattr(self.trainer, 'is_global_zero') and self.trainer.is_global_zero:
            val_accuracy = correct_preds.float() / total_preds if total_preds > 0 else 0.0
            self.log("val_accuracy", val_accuracy, prog_bar=True, logger=True)
            logger.info(f"Validation Epoch End: Accuracy = {val_accuracy:.4f} ({int(correct_preds)}/{int(total_preds)})")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        global train_drug_dataset
        if train_drug_dataset is None: raise ValueError("학습 데이터셋이 초기화되지 않음")
        return DataLoader(train_drug_dataset, collate_fn=drug_collate_fn_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        global val_drug_dataset
        if val_drug_dataset is None:
            logger.info("검증 데이터셋이 없어 검증 단계를 건너<0xEB><0><0x84><0x90>니다.")
            return None
        return DataLoader(val_drug_dataset, collate_fn=drug_collate_fn_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- 7. 메인 학습 함수 정의 (`main_train`) ---
def main_train(resume_from_checkpoint: str = None):
    global model_main, processor_main, train_drug_dataset, val_drug_dataset

    try:
        import transformers, tokenizers
        logger.info(f"사용 중인 Transformers 버전: {transformers.__version__}")
        logger.info(f"사용 중인 Tokenizers 버전: {tokenizers.__version__}")
        # (버전 확인 및 경고 로직 - 이전 코드 참조)
    except ImportError:
        logger.error("Transformers 또는 Tokenizers 라이브러리 설치 필요.")
        return

    logger.info(f"LLaVA ({MODEL_ID}) 파인튜닝 시작 (스크립트: train_script.py).")
    L.seed_everything(42, workers=True)
    should_force_download = resume_from_checkpoint is None

    try:
        logger.info(f"프로세서 로딩: {MODEL_ID}")
        processor_main = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, force_download=should_force_download)
        processor_main.tokenizer.padding_side = "right"
        if processor_main.tokenizer.pad_token is None or processor_main.tokenizer.pad_token_id is None:
            pad_token_str_candidate = "<pad>" # LLaVA 1.6은 <pad>를 가짐 (ID 32001)
            if pad_token_str_candidate in processor_main.tokenizer.get_vocab():
                pad_token_id_candidate = processor_main.tokenizer.convert_tokens_to_ids(pad_token_str_candidate)
                processor_main.tokenizer.pad_token = pad_token_str_candidate
                processor_main.tokenizer.pad_token_id = pad_token_id_candidate
                logger.info(f"모델의 <pad> 토큰 (ID: {pad_token_id_candidate})을 PAD 토큰으로 설정.")
            elif processor_main.tokenizer.eos_token is not None and processor_main.tokenizer.eos_token_id is not None:
                processor_main.tokenizer.pad_token = processor_main.tokenizer.eos_token
                processor_main.tokenizer.pad_token_id = processor_main.tokenizer.eos_token_id
                logger.warning(f"모델 어휘에 <pad> 토큰이 없어 EOS 토큰을 PAD 토큰으로 사용합니다.")
        logger.info(f"최종 PAD: '{processor_main.tokenizer.pad_token}', ID: {processor_main.tokenizer.pad_token_id}")

        logger.info(f"모델 로딩: {MODEL_ID}")
        quantization_config_to_pass = None
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if USE_QLORA:
            quantization_config_to_pass = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype, bnb_4bit_use_double_quant=True,
            )
        model_main = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=model_dtype,
            quantization_config=quantization_config_to_pass,
            low_cpu_mem_usage=True, trust_remote_code=True,
            force_download=should_force_download
        )
        if model_main is None: raise ValueError(f"{MODEL_ID} 모델 로드 실패.")
        logger.info(f"{MODEL_ID} 모델 로드 성공. 타입: {type(model_main)}")

        if USE_QLORA:
            logger.info("PEFT LoRA 설정 적용 중...")
            # (PEFT 설정 부분 동일)
            target_modules_str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
            lora_target_modules = [name.strip() for name in target_modules_str.split(",")]
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=lora_target_modules, bias="none", task_type="CAUSAL_LM"
            )
            model_main = prepare_model_for_kbit_training(model_main)
            model_main = get_peft_model(model_main, peft_config)
            model_main.print_trainable_parameters()
    except Exception as e:
        logger.error(f"모델/프로세서 초기화 중 오류: {e}", exc_info=True)
        return

    try:
        train_drug_dataset = DrugImageDataset(TRAIN_DATA_JSON_PATH, IMAGE_BASE_DIRECTORY, FIXED_QUESTION, "train")
        if VAL_DATA_JSON_PATH and os.path.exists(VAL_DATA_JSON_PATH):
            val_drug_dataset = DrugImageDataset(VAL_DATA_JSON_PATH, IMAGE_BASE_DIRECTORY, FIXED_QUESTION, "validation")
        else: val_drug_dataset = None
    except Exception as e:
        logger.error(f"데이터셋 준비 중 오류: {e}", exc_info=True); return
    if not train_drug_dataset or len(train_drug_dataset) == 0:
        logger.error("학습 데이터셋 로드 실패 또는 비어있음."); return

    pl_model_module = LlavaClassifierPLModule(model_main, processor_main, LEARNING_RATE, BATCH_SIZE)
    precision_to_use = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"

    # ModelCheckpoint 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(OUTPUT_MODEL_DIR, "checkpoints"),
        filename="epoch{epoch:02d}-val_acc{val_accuracy:.2f}",
        save_top_k=2, # val_accuracy 기준 상위 2개 저장
        monitor="val_accuracy",
        mode="max",
        save_last=True, # last.ckpt 항상 저장
        every_n_epochs=1 # 매 에폭마다 체크포인트 (val_check_interval과 연동하여 더 자주 가능)
    )
    callbacks_to_use = [checkpoint_callback] if val_drug_dataset else []

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRADIENT_CLIP_VAL, precision=precision_to_use,
        val_check_interval=0.25 if val_drug_dataset else 1.0,
        limit_val_batches=1.0 if val_drug_dataset else 0, # 이제 전체 검증 데이터 사용
        log_every_n_steps=10,
        num_sanity_val_steps=0 if not val_drug_dataset else 2,
        deterministic=True,
        callbacks=callbacks_to_use,
        default_root_dir=os.path.join(OUTPUT_MODEL_DIR, "lightning_logs")
    )

    logger.info("PyTorch Lightning Trainer 학습 시작!")
    try:
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"체크포인트에서 학습을 이어갑니다: {resume_from_checkpoint}")
            trainer.fit(pl_model_module, ckpt_path=resume_from_checkpoint)
        else:
            if resume_from_checkpoint: logger.warning(f"지정된 체크포인트({resume_from_checkpoint}) 없음. 처음부터 학습.")
            else: logger.info("체크포인트 없이 처음부터 학습 시작.")
            trainer.fit(pl_model_module)

        logger.info("학습 정상 완료.")
        final_save_path = os.path.join(OUTPUT_MODEL_DIR, "final_model_checkpoint")
        os.makedirs(final_save_path, exist_ok=True)
        logger.info(f"최종 모델/프로세서 저장 중: {final_save_path}")
        pl_model_module.model.save_pretrained(final_save_path)
        processor_main.save_pretrained(final_save_path)
        print(f"✅ 파인튜닝 및 저장 완료: {final_save_path}")
    except Exception as e_fit:
        logger.error(f"Trainer.fit() 실행 중 오류: {e_fit}", exc_info=True)

# --- 스크립트 실행 지점 ---
if __name__ == '__main__':
    # 이어할 체크포인트 파일 경로를 여기에 지정합니다.
    # 예: resume_path = "outputs/llava_next_drug_classifier_resumable/lightning_logs/version_X/checkpoints/last.ckpt"
    # 또는 명시적 ModelCheckpoint 콜백 사용 시 해당 경로의 파일
    # 예: resume_path = "outputs/llava_next_drug_classifier_resumable/checkpoints/last.ckpt"
    resume_path = RESUME_CKPT_PATH # 스크립트 상단에서 정의한 변수 사용

    main_train(resume_from_checkpoint=resume_path)
