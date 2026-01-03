# 🚀 MOAI-LLM 완전 구현 가이드

**3B 파라미터 언어모델을 HuggingFace Datasets로 처음부터 끝까지 구현하는 완전 가이드**

---

## 📋 목차

1. [개요](#1-개요)
2. [환경 설정](#2-환경-설정)
3. [토크나이저 학습](#3-토크나이저-학습)
4. [사전학습 (Pretrain)](#4-사전학습-pretrain)
5. [파인튜닝 (SFT)](#5-파인튜닝-sft)
6. [모델 평가 및 추론](#6-모델-평가-및-추론)
7. [고급 기능](#7-고급-기능)
8. [문제 해결](#8-문제-해결)

---

## 1. 개요

### 1.1 MOAI-LLM이란?

**MOAI-LLM**은 3B 파라미터 언어모델입니다.

#### 핵심 특징:

- ✅ **최신 아키텍처**: 최신 LLM 기술 완전 구현
- ✅ **HuggingFace 통합**: 모든 단계에서 datasets 사용
- ✅ **긴 컨텍스트**: 32K tokens (YaRN으로 128K+ 확장 가능)
- ✅ **효율적**: GQA (7:1), Flash Attention 지원
- ✅ **다국어**: 한국어, 영어, 코드 동시 지원
- ✅ **자동화**: 데이터 다운로드부터 학습까지 원스톱

#### 아키텍처 사양:

```python
모델 크기: 3B parameters
- Layers: 28
- Hidden size: 3,840
- Attention heads: 28 (Q) / 4 (KV)
- Vocabulary: 128,000 (SentencePiece BPE)
- Max sequence: 32,768 tokens
- RoPE theta: 1,000,000
- Activation: SwiGLU
```

### 1.2 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│                    MOAI-LLM 학습 파이프라인                  │
└─────────────────────────────────────────────────────────────┘

1. 토크나이저 학습
   └─> HuggingFace datasets (wikipedia, C4, etc.)
   └─> SentencePiece BPE 학습 (128K vocab)
   └─> 출력: tokenizers/moai_tokenizer.model

2. 사전학습 (Pretrain)
   └─> HuggingFace datasets (wikipedia, bookcorpus, etc.)
   └─> Causal Language Modeling (Next Token Prediction)
   └─> 출력: outputs/pretrain/final_model/

3. 파인튜닝 (SFT)
   └─> HuggingFace datasets (alpaca, KULLM, etc.)
   └─> Instruction Following 학습
   └─> 출력: outputs/sft/final_model/

4. 추론 및 배포
   └─> chat.py로 대화형 인터페이스
   └─> HuggingFace Hub 배포
```

---

## 2. 환경 설정

### 2.1 시스템 요구사항

#### 최소 요구사항:
```
Python: 3.10+
GPU: RTX 3090 (24GB) × 1
RAM: 64GB
Disk: 200GB
CUDA: 11.8+
```

#### 권장 사양:
```
GPU: RTX 4090 (24GB) × 4 또는 A100 (80GB) × 2
RAM: 128GB+
Disk: 1TB SSD
```

#### 예상 비용:
```
토크나이저 학습: ~1시간 (무료, CPU 가능)
사전학습: ~3일 (A100 × 4 기준)
SFT: ~6시간 (A100 × 1 기준)
```

### 2.2 설치

```bash
# 1. 저장소 클론
git clone https://github.com/sh2orc/moai-llm.git
cd moai-llm

# 2. 가상환경 설정
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. 기본 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. 개발 모드로 설치
pip install -e .

# 5. Flash Attention 설치 (선택, GPU 필수)
pip install flash-attn --no-build-isolation

# 6. Weights & Biases (로깅, 선택)
pip install wandb
wandb login
```

### 2.3 환경 확인

```bash
# CUDA 확인
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.device_count()}')"

# Flash Attention 확인 (선택)
python -c "try: import flash_attn; print('Flash Attention: ✅')
except: print('Flash Attention: ❌')"

# HuggingFace datasets 확인
python -c "from datasets import load_dataset; print('Datasets: ✅')"
```

---

## 3. 토크나이저 학습

### 3.1 개요

토크나이저는 텍스트를 토큰(숫자)으로 변환합니다. MOAI-LLM은 **SentencePiece BPE**를 사용합니다.

**핵심**: HuggingFace datasets를 직접 사용하여 학습합니다!

### 3.2 기본 토크나이저 학습

#### 방법 1: HuggingFace Dataset 사용 (권장)

```bash
# 한국어 Wikipedia로 토크나이저 학습
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 128000 \
    --output_dir tokenizers/korean/

# 영어 Wikipedia
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 128000 \
    --output_dir tokenizers/english/

# C4 (대용량 영어)
python train_tokenizer.py \
    --dataset allenai/c4 \
    --dataset_config en \
    --vocab_size 128000 \
    --max_samples 1000000 \
    --output_dir tokenizers/c4/
```

#### 방법 2: 로컬 파일 사용

```bash
# 로컬 txt 파일
python train_tokenizer.py \
    --input_files data/pretrain/*.txt \
    --vocab_size 128000 \
    --output_dir tokenizers/local/
```

### 3.3 다국어 토크나이저

```bash
# 한국어 + 영어 혼합 (추천)
# Step 1: 한국어 베이스
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 100000 \
    --output_dir tokenizers/base/

# Step 2: 영어 추가 (업데이트 모드)
python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 180000 \
    --max_samples 1000000 \
    --output_dir tokenizers/bilingual/
```

### 3.4 코드 특화 토크나이저

```bash
# Python 코드 토큰 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/english/moai_tokenizer.model \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --vocab_size 150000 \
    --max_samples 200000 \
    --output_dir tokenizers/code/
```

### 3.5 도메인 특화 토크나이저

```bash
# 금융 도메인
python train_tokenizer.py \
    --base_tokenizer tokenizers/korean/moai_tokenizer.model \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 150000 \
    --output_dir tokenizers/finance/
```

### 3.6 토크나이저 확인

```python
# test_tokenizer.py
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('tokenizers/korean/moai_tokenizer.model')

# 테스트
texts = [
    "안녕하세요. MOAI-LLM입니다.",
    "Hello, this is a test.",
    "print('Hello, World!')",
]

for text in texts:
    tokens = sp.encode(text, out_type=str)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {sp.encode(text)}")
    print(f"Decoded: {sp.decode(sp.encode(text))}")
    print("-" * 50)
```

### 3.7 데이터셋 Config 확인

어떤 dataset_config를 사용해야 할지 모를 때:

```bash
# Dataset config 확인 도구
python check_dataset.py wikipedia
python check_dataset.py allenai/c4
python check_dataset.py BCCard/BCCard-Finance-Kor-QnA
```

---

## 4. 사전학습 (Pretrain)

### 4.1 개요

사전학습은 대량의 텍스트로 **다음 토큰 예측(Causal LM)**을 학습합니다.

**모든 데이터는 HuggingFace datasets에서 자동 다운로드됩니다!**

### 4.2 빠른 테스트 (10분)

```bash
# 작은 데이터셋으로 테스트
python train.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir outputs/test \
    --max_steps 100 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### 4.3 한국어 사전학습

```bash
# Wikipedia 한국어 (1GB, ~수일)
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --tokenizer_path tokenizers/korean/moai_tokenizer.model \
    --output_dir outputs/pretrain-ko \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --gradient_checkpointing \
    --save_steps 1000 \
    --logging_steps 10
```

### 4.4 영어 사전학습

```bash
# C4 영어 (300GB, ~수주)
python train.py \
    --mode pretrain \
    --dataset allenai/c4 \
    --dataset_config en \
    --tokenizer_path tokenizers/english/moai_tokenizer.model \
    --output_dir outputs/pretrain-c4 \
    --max_steps 100000 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --bf16 \
    --gradient_checkpointing
```

### 4.5 다국어 사전학습

```bash
# mC4 다국어
python train.py \
    --mode pretrain \
    --dataset allenai/c4 \
    --dataset_config multilingual \
    --tokenizer_path tokenizers/bilingual/moai_tokenizer.model \
    --output_dir outputs/pretrain-multilingual \
    --max_steps 50000 \
    --bf16
```

### 4.6 코드 사전학습

```bash
# The Stack (Python)
python train.py \
    --mode pretrain \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --tokenizer_path tokenizers/code/moai_tokenizer.model \
    --output_dir outputs/pretrain-code \
    --num_epochs 1 \
    --batch_size 2 \
    --bf16
```

### 4.7 멀티 GPU 학습

```bash
# 4 GPU로 분산 학습
torchrun --nproc_per_node=4 train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain-distributed \
    --batch_size 4 \
    --bf16 \
    --gradient_checkpointing

# DeepSpeed 사용 (ZeRO Stage 2)
deepspeed --num_gpus=4 train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain-deepspeed \
    --deepspeed configs/deepspeed_config.json \
    --bf16
```

### 4.8 로컬 txt 파일 사용

```bash
# 로컬 파일로 사전학습
python train.py \
    --mode pretrain \
    --train_file data/pretrain/train.txt \
    --tokenizer_path tokenizers/korean/moai_tokenizer.model \
    --output_dir outputs/pretrain-local \
    --num_epochs 3 \
    --bf16
```

---

## 5. 파인튜닝 (SFT)

### 5.1 개요

SFT(Supervised Fine-Tuning)는 **Instruction Following** 능력을 학습합니다.

**자동 포맷 감지**: Alpaca, Chat, ShareGPT, input/output 등 자동 변환!

### 5.2 빠른 테스트

```bash
# BCCard 금융 Q&A (빠른 테스트)
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-test \
    --max_steps 50 \
    --batch_size 2
```

### 5.3 한국어 SFT

#### 5.3.1 KULLM (150K 샘플)

```bash
python train.py \
    --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-kullm \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --bf16
```

#### 5.3.2 KoAlpaca (52K 샘플)

```bash
python train.py \
    --mode sft \
    --dataset beomi/KoAlpaca-v1.1a \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-koalpaca \
    --num_epochs 2 \
    --bf16
```

#### 5.3.3 BCCard 금융 (4K 샘플)

```bash
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-finance \
    --num_epochs 5 \
    --bf16
```

### 5.4 영어 SFT

#### 5.4.1 Alpaca (52K 샘플)

```bash
python train.py \
    --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-alpaca \
    --num_epochs 3 \
    --bf16
```

#### 5.4.2 LIMA (1K 고품질)

```bash
python train.py \
    --mode sft \
    --dataset GAIR/lima \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-lima \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --bf16
```

#### 5.4.3 OpenAssistant (161K 샘플)

```bash
python train.py \
    --mode sft \
    --dataset OpenAssistant/oasst1 \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-oasst \
    --num_epochs 2 \
    --bf16
```

### 5.5 코드 SFT

```bash
# Code Alpaca
python train.py \
    --mode sft \
    --dataset sahil2801/CodeAlpaca-20k \
    --pretrained_model outputs/pretrain-code/final_model \
    --output_dir outputs/sft-code \
    --num_epochs 3 \
    --bf16
```

### 5.6 로컬 JSON 파일 사용

```bash
# 로컬 JSON 파일로 SFT
python train.py \
    --mode sft \
    --train_file data/sft/my_dataset.json \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-custom \
    --num_epochs 3 \
    --bf16
```

**JSON 포맷 예시:**

```json
[
  {
    "instruction": "한국의 수도는?",
    "output": "한국의 수도는 서울입니다."
  },
  {
    "input": "GPT가 뭐야?",
    "output": "GPT는 Generative Pre-trained Transformer의 약자입니다."
  },
  {
    "messages": [
      {"role": "user", "content": "안녕?"},
      {"role": "assistant", "content": "안녕하세요!"}
    ]
  }
]
```

### 5.7 다중 데이터셋 혼합

```bash
# 여러 데이터셋을 순차적으로 학습
# Step 1: Alpaca
python train.py --mode sft --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain/final_model \
    --output_dir outputs/sft-stage1

# Step 2: LIMA (고품질)
python train.py --mode sft --dataset GAIR/lima \
    --pretrained_model outputs/sft-stage1/final_model \
    --output_dir outputs/sft-stage2 \
    --learning_rate 1e-5  # Lower LR for refinement
```

---

## 6. 모델 평가 및 추론

### 6.1 대화형 테스트

```bash
# 채팅 인터페이스
python chat.py \
    --model_path outputs/sft-kullm/final_model \
    --max_new_tokens 256 \
    --temperature 0.7
```

**대화 예시:**
```
💬 You: 한국의 수도는?
🤖 MOAI: 한국의 수도는 서울입니다.

💬 You: Python으로 피보나치 수열을 구현해줘
🤖 MOAI:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

💬 You: exit
```

### 6.2 배치 추론 테스트

```bash
# 미리 정의된 프롬프트로 테스트
python test_inference.py \
    --model_path outputs/sft-kullm/final_model
```

### 6.3 Perplexity 평가

```python
# evaluate_ppl.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "outputs/sft-kullm/final_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/sft-kullm/final_model"
)

# 평가 데이터
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Perplexity 계산
encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
max_length = 2048
stride = 512

nlls = []
for i in range(0, encodings.input_ids.size(1), stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"Perplexity: {ppl.item():.2f}")
```

### 6.4 HuggingFace Hub 업로드

```bash
# HuggingFace에 모델 업로드
huggingface-cli login

# 업로드
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('outputs/sft-kullm/final_model')
tokenizer = AutoTokenizer.from_pretrained('outputs/sft-kullm/final_model')

model.push_to_hub('your-username/moai-llm-3b-ko')
tokenizer.push_to_hub('your-username/moai-llm-3b-ko')
"
```

---

## 7. 고급 기능

### 7.1 긴 컨텍스트 확장 (YaRN)

기본 32K tokens를 128K까지 확장:

```python
# configs/long_context_config.json
{
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "rope_scaling": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

```bash
# 긴 컨텍스트로 학습
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --config_file configs/long_context_config.json \
    --output_dir outputs/pretrain-128k \
    --bf16
```

### 7.2 LoRA 파인튜닝 (메모리 절약)

```python
# train_lora.py
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "outputs/pretrain-ko/final_model",
    torch_dtype=torch.bfloat16
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20M || all params: 3020M || trainable%: 0.66%
```

### 7.3 Gradient Checkpointing (메모리 절약)

```bash
# 메모리 부족 시
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain \
    --gradient_checkpointing \  # 메모리 50% 절약
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --bf16
```

### 7.4 혼합 정밀도 학습

```bash
# BF16 (권장, A100/H100)
python train.py --bf16 ...

# FP16 (V100/RTX)
python train.py --fp16 ...

# FP8 (H100)
python train.py --fp8 ...
```

### 7.5 Wandb 로깅

```bash
# Wandb 활성화
export WANDB_PROJECT="moai-llm"
export WANDB_ENTITY="your-username"

python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain \
    --report_to wandb \
    --run_name "pretrain-wikipedia-ko"
```

### 7.6 체크포인트 재개

```bash
# 중단된 학습 재개
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain \
    --resume_from_checkpoint outputs/pretrain/checkpoint-5000 \
    --bf16
```

---

## 8. 문제 해결

### 8.1 OOM (Out of Memory)

**증상**: `CUDA out of memory` 에러

**해결책**:
```bash
# 1. Batch size 줄이기
--batch_size 1 --gradient_accumulation_steps 32

# 2. Gradient checkpointing
--gradient_checkpointing

# 3. Sequence length 줄이기
--max_seq_length 1024  # 기본 2048

# 4. Mixed precision
--bf16  # 또는 --fp16

# 5. LoRA 사용
# train_lora.py 참고
```

### 8.2 Dataset Config 에러

**증상**: `ValueError: Config name is missing`

**해결책**:
```bash
# Config 확인
python check_dataset.py wikipedia

# Config 명시
--dataset wikipedia --dataset_config 20220301.ko
```

### 8.3 토크나이저 로드 실패

**증상**: `FileNotFoundError: moai_tokenizer.model`

**해결책**:
```bash
# 토크나이저 먼저 학습
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir tokenizers/

# 정확한 경로 지정
--tokenizer_path tokenizers/moai_tokenizer.model
```

### 8.4 학습 속도 느림

**증상**: 1 step에 10초 이상 소요

**해결책**:
```bash
# 1. Flash Attention 설치
pip install flash-attn --no-build-isolation

# 2. DataLoader workers 증가
--dataloader_num_workers 4

# 3. 멀티 GPU 사용
torchrun --nproc_per_node=4 train.py ...

# 4. 데이터 프리페치
--dataloader_prefetch_factor 2
```

### 8.5 HuggingFace 다운로드 실패

**증상**: `Connection timeout` 또는 `403 Forbidden`

**해결책**:
```bash
# 1. HuggingFace 로그인
huggingface-cli login

# 2. 미러 사용
export HF_ENDPOINT=https://hf-mirror.com

# 3. 캐시 초기화
rm -rf ~/.cache/huggingface/datasets/

# 4. 수동 다운로드
from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.ko", cache_dir="/path/to/cache")
```

---

## 9. 실전 시나리오

### 9.1 한국어 범용 모델

```bash
# 1. 토크나이저 (한국어 중심)
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.ko \
    --vocab_size 128000 --output_dir tokenizers/ko/

# 2. 사전학습 (Wikipedia)
python train.py --mode pretrain \
    --dataset wikipedia --dataset_config 20220301.ko \
    --tokenizer_path tokenizers/ko/moai_tokenizer.model \
    --output_dir outputs/pretrain-ko --num_epochs 3 --bf16

# 3. SFT (KULLM)
python train.py --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-ko --num_epochs 3 --bf16

# 4. 테스트
python chat.py --model_path outputs/sft-ko/final_model
```

### 9.2 영어 고품질 모델 (LIMA 스타일)

```bash
# 1. 토크나이저 (영어)
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.en \
    --vocab_size 128000 --output_dir tokenizers/en/

# 2. 사전학습 (C4)
python train.py --mode pretrain \
    --dataset allenai/c4 --dataset_config en \
    --tokenizer_path tokenizers/en/moai_tokenizer.model \
    --output_dir outputs/pretrain-c4 --max_steps 50000 --bf16

# 3. SFT (LIMA - 1K 고품질)
python train.py --mode sft \
    --dataset GAIR/lima \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-lima \
    --num_epochs 10 --learning_rate 1e-5 --bf16

# 4. 테스트
python chat.py --model_path outputs/sft-lima/final_model
```

### 9.3 코드 생성 모델

```bash
# 1. 토크나이저 (코드 특화)
python train_tokenizer.py \
    --dataset bigcode/the-stack --dataset_config data/python \
    --vocab_size 128000 --max_samples 200000 \
    --output_dir tokenizers/code/

# 2. 사전학습 (The Stack)
python train.py --mode pretrain \
    --dataset bigcode/the-stack --dataset_config data/python \
    --tokenizer_path tokenizers/code/moai_tokenizer.model \
    --output_dir outputs/pretrain-code --num_epochs 1 --bf16

# 3. SFT (Code Alpaca)
python train.py --mode sft \
    --dataset sahil2801/CodeAlpaca-20k \
    --pretrained_model outputs/pretrain-code/final_model \
    --output_dir outputs/sft-code --num_epochs 3 --bf16

# 4. 테스트
python chat.py --model_path outputs/sft-code/final_model
```

### 9.4 금융 도메인 특화 모델

```bash
# 1. 토크나이저 (한국어 + 금융)
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.ko \
    --vocab_size 100000 --output_dir tokenizers/base/

python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 128000 --output_dir tokenizers/finance/

# 2. 사전학습 (Wikipedia)
python train.py --mode pretrain \
    --dataset wikipedia --dataset_config 20220301.ko \
    --tokenizer_path tokenizers/finance/moai_tokenizer.model \
    --output_dir outputs/pretrain-ko --bf16

# 3. SFT (BCCard)
python train.py --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-finance --num_epochs 5 --bf16

# 4. 테스트
python chat.py --model_path outputs/sft-finance/final_model
```

---

## 10. 추가 자료

### 10.1 문서

- **QUICKSTART.md**: 10분 빠른 시작
- **ARCHITECTURE.md**: 아키텍처 상세
- **EMBEDDING_GUIDE.md**: 임베딩 가이드
- **TOKENIZER_UPDATE_GUIDE.md**: 토크나이저 업데이트
- **DATASET_CONFIGS.md**: 데이터셋 Config 가이드
- **POPULAR_DATASETS.md**: 추천 데이터셋 목록

### 10.2 스크립트

```
moai-llm/
├── train_tokenizer.py      # 토크나이저 학습
├── train.py                 # 통합 학습 스크립트
├── chat.py                  # 대화형 인터페이스
├── test_inference.py        # 추론 테스트
├── check_dataset.py         # 데이터셋 정보 확인
└── configs/
    └── model_config.json    # 모델 설정
```

### 10.3 HuggingFace Datasets

**사전학습용**:
- `wikipedia` (한국어/영어)
- `allenai/c4` (영어, 300GB)
- `bigcode/the-stack` (코드)
- `mc4` (다국어)

**SFT용**:
- `tatsu-lab/alpaca` (영어, 52K)
- `nlpai-lab/kullm-v2` (한국어, 150K)
- `BCCard/BCCard-Finance-Kor-QnA` (금융, 4K)
- `GAIR/lima` (고품질, 1K)

### 10.4 참고 논문

- **Qwen3**: https://arxiv.org/abs/2506.05176
- **RoPE**: https://arxiv.org/abs/2104.09864
- **YaRN**: https://arxiv.org/abs/2309.00071
- **Flash Attention**: https://arxiv.org/abs/2307.08691
- **GQA**: https://arxiv.org/abs/2305.13245

---

## 🎉 완료!

이제 MOAI-LLM을 처음부터 끝까지 학습할 수 있습니다!

**핵심 요약**:
1. ✅ **모든 단계가 HuggingFace datasets 기반**
2. ✅ **데이터 다운로드/변환 자동화**
3. ✅ **최신 아키텍처**
4. ✅ **토크나이저 → 사전학습 → SFT → 추론 전체 파이프라인**

**다음 단계**:
```bash
# 10분 빠른 시작
cat QUICKSTART.md

# 본격 학습
python train_tokenizer.py --dataset wikipedia --dataset_config 20220301.ko --output_dir tokenizers/
python train.py --mode pretrain --dataset wikipedia --dataset_config 20220301.ko --bf16
python chat.py --model_path outputs/pretrain/final_model
```

**질문/이슈**: https://github.com/yourusername/moai-llm/issues

Happy Training! 🚀
