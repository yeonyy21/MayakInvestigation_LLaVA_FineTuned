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
