# MayakInvestigation_LLaVA_FineTuned
2025년 하계 한국정보기술학회 학술대회에 투고한 논문입니다.

# LLaVA 1.6 기반 마약 범죄 수사 지원 디지털 포렌식 이미지 자동 분류 시스템

## 📜 개요

본 프로젝트는 "LLaVA 1.6(Vision-Language Model) 기반 마약 범죄 수사 지원을 위한 디지털 포렌식 이미지 자동 분류 시스템 연구" 논문에 기술된 자동 디지털 포렌식 이미지 분류 시스템을 구현합니다. 이 시스템은 QLoRA 기법으로 효율적으로 파인튜닝된 LLaVA 1.6 비전-언어 모델(특히 `llava-hf/llava-v1.6-mistral-7b-hf`)을 활용합니다.

시스템은 마약 관련 포렌식 이미지를 다음과 같은 4가지 사전 정의된 범주로 분류합니다:

- **0: 관련 없음**
- **1: 마약 대화 내역 사진**
- **2: 마약 거래 장소**
- **3: 마약 원본**

이 프로젝트의 목표는 방대한 양의 이미지 증거 분석을 자동화하여 수사 효율성을 높이고 수사관의 업무 부담을 줄이는 것입니다.

## 💻 시스템 구성 요소 및 스크립트

본 프로젝트는 다음의 Python 스크립트로 구성됩니다:

1. **`preprocess.py`**:
    - `raw_data` 디렉토리에서 원본 이미지 정보를 수집합니다.
    - 데이터셋을 학습(train), 검증(validation), 테스트(test) 세트로 분할합니다 (약 72.25% / 12.75% / 15%).
    - 학습 세트 내의 '2: 마약 거래 장소' 이미지에 대해 데이터 불균형 해소를 위해 데이터 증강(Albumentations 사용)을 적용합니다. 이 클래스의 원본 이미지 당 4개의 새로운 증강 이미지를 생성합니다.
    - 처리된 데이터셋 정보를 `data` 디렉토리에 JSON 파일(`train_drug_data.json`, `val_drug_data.json`, `test_drug_data.json`)로 저장합니다.
2. **`train_script.py`**:
    - LLaVA 1.6 모델(`llava-hf/llava-v1.6-mistral-7b-hf`)을 QLoRA를 사용하여 파인튜닝합니다.
    - `preprocess.py`에 의해 생성된 JSON 파일로부터 데이터를 로드합니다.
    - PyTorch Lightning을 사용하여 학습 루프를 구성합니다.
    - 주요 학습 파라미터:
        - 에폭(Epochs): 5
        - 학습률(Learning Rate): 1e-5
        - 유효 배치 크기(Effective Batch Size): 16 (배치 크기 1, 그래디언트 누적 16)
        - 옵티마이저(Optimizer): AdamW
        - 정밀도(Precision): bf16 혼합 정밀도
        - LoRA `r`: 16, `lora_alpha`: 32, `lora_dropout`: 0.05 (참고: `r`과 `lora_alpha` 값은 논문에 언급된 각각 8과 16과 다릅니다).
    - 파인튜닝된 모델 체크포인트 및 최종 모델/프로세서를 `outputs/llava_next_drug_classifier_epochs5` 디렉토리에 저장합니다.
    - 체크포인트로부터 학습을 재개하는 기능을 포함합니다.
3. **`eval.py`**:
    - **파인튜닝된 LLaVA 1.6 모델**의 성능을 평가합니다.
    - `FINETUNED_MODEL_PATH`에 지정된 경로(예: `outputs/llava_next_drug_classifier_epochs5/final_model_checkpoint`)에서 파인튜닝된 모델을 로드합니다.
    - 테스트 데이터셋(`data/test_drug_data.json`)을 사용합니다.
    - 정확도(accuracy)와 같은 평가지표를 계산하고, 분류 보고서(classification report) 및 혼동 행렬(confusion matrix)을 생성합니다.
    - 평가 결과(예: 혼동 행렬 이미지)를 `evaluation_results_epochs5` 디렉토리에 저장합니다.
4. **`vanilla.py`**:
    - 파인튜닝되지 않은 **기본 LLaVA 1.6 모델**(`llava-hf/llava-v1.6-mistral-7b-hf`)의 **제로샷(zero-shot) 성능**을 평가합니다.
    - 테스트 데이터셋(`data/test_drug_data.json`)을 사용합니다.
    - 모델을 4비트 양자화하여 로드합니다.
    - 모델의 다소 장황할 수 있는 출력을 사전 정의된 클래스 레이블 중 하나로 해석하기 위한 사용자 정의 파싱 함수(`parse_llava_zeroshot_output`)를 포함합니다.
    - 평가지표를 계산하고 결과를 `evaluation_results_zeroshot` 디렉토리에 저장합니다.

## 🚀 주요 방법론

- **모델**: LLaVA 1.6 (`llava-hf/llava-v1.6-mistral-7b-hf`), `LlavaNextForConditionalGeneration` 사용.
- **파인튜닝**: QLoRA (Quantized Low-Rank Adaptation)를 사용하여 단일 GPU(논문에서 NVIDIA A100 80GB 언급됨) 환경에서 메모리 효율적인 파인튜닝 수행.
    - 기본 모델 가중치를 4비트(nf4)로 양자화.
    - LoRA를 언어 모델 부분의 특정 선형 레이어에 적용.
- **데이터셋**: 마약 범죄 관련 포렌식 이미지로 구성된 사용자 정의 데이터셋, 4가지 범주로 분류됨. 사용된 질문 프롬프트: `"이 이미지는 다음 중 어떤 유형입니까? (0: 관련 없음, 1: 마약 대화 내역 사진, 2: 마약 거래 장소, 3: 마약 원본)"`
- **데이터 증강**: 상대적으로 수가 적은 '마약 거래 장소' 클래스에 적용하여 모델의 일반화 성능 향상. 사용된 기법에는 좌우 반전, 회전, 밝기/대비 조절, 노이즈 추가, 블러 처리 등이 포함됨.
- **평가**: 제로샷 기본 모델과 파인튜닝된 모델 간의 성능을 표준 분류 평가지표를 사용하여 비교. 논문에서는 제로샷 약 49.6%에서 파인튜닝 후 96.34%로 정확도가 크게 향상되었다고 보고됨.

## ⚙️ 설정 및 실행 방법

1. **환경 설정**:
    - 필요한 라이브러리(PyTorch, Transformers, PEFT, PyTorch Lightning, Albumentations, scikit-learn, OpenCV 등)가 설치된 Python 환경을 구성합니다.
    - 필요한 경우 `TOKENIZERS_PARALLELISM=false` 환경 변수를 설정합니다.
2. **데이터 준비**:
    - 원본 이미지 데이터를 `CLASS_MAPPING`에 따라 `raw_data/` 내의 하위 폴더(예: `raw_data/0_irrelevant/`, `raw_data/1_chat_history/` 등)에 배치합니다.
    - `python preprocess.py`를 실행하여 `data/*.json` 파일을 생성합니다.
3. **학습**:
    - 필요한 경우 `train_script.py` 내의 `OUTPUT_MODEL_DIR` 및 `RESUME_CKPT_PATH`를 수정합니다.
    - `python train_script.py`를 실행하여 모델을 파인튜닝합니다. 체크포인트는 출력 디렉토리에 저장됩니다.
4. **평가**:
    - **파인튜닝된 모델**:
        - `eval.py`의 `FINETUNED_MODEL_PATH`가 학습된 모델 체크포인트(예: `outputs/llava_next_drug_classifier_epochs5/final_model_checkpoint`)를 가리키도록 합니다.
        - `python eval.py`를 실행합니다. 결과는 `evaluation_results_epochs5`에 저장됩니다.
    - **제로샷 모델**:
        - `python vanilla.py`를 실행합니다. 결과는 `evaluation_results_zeroshot`에 저장됩니다.

## 📝 참고 사항

- 스크립트들은 저의 경로 및 모델 ID로 구성되어 있습니다. 사용자의 환경에 맞게 적절히 수정하십시오.
- `train_script.py`의 LoRA 하이퍼파라미터 `r` (16) 및 `lora_alpha` (32)는 논문의 세부 방법론에 언급된 값(각각 8, 16)과 다릅니다.
- matplotlib으로 생성된 혼동 행렬에서 한글 레이블을 올바르게 표시하려면 시스템에 적절한 한글 폰트가 설치되어 있어야 합니다 (`eval.py` 및 `vanilla.py`의 `set_korean_font` 함수 참조).
