# ⚡ MOAI-LLM 빠른 시작 가이드

## 🎯 한 줄 요약

**HuggingFace 데이터셋으로 토크나이저부터 SFT까지 자동화!**

---

## 📋 전체 워크플로우 (3단계)

```bash
# 1. 토크나이저 학습 → 2. 사전학습 → 3. SFT 파인튜닝
```

---

## ⚡ 10분 빠른 테스트

파이프라인이 제대로 작동하는지 먼저 확인하세요:

```bash
# Step 1: 토크나이저 (빠른 버전, 2분)
python train_tokenizer.py \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --vocab_size 32000 \
    --max_samples 10000 \
    --output_dir tokenizers/test

# Step 2: 사전학습 (100 steps, 3분)
python train.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir outputs/test \
    --max_steps 100

# Step 3: SFT (50 steps, 2분)
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/test/final_model \
    --output_dir outputs/test-sft \
    --max_steps 50

# Step 4: 채팅 테스트 (즉시)
python chat.py --model_path outputs/test-sft/final_model
```

**✅ 모든 단계가 정상 작동하면 실전 학습으로!**

---

## 🚀 실전 학습 (프로덕션)

### Step 1: 토크나이저 학습 (1-2시간)

```bash
# HuggingFace 데이터셋 사용 (권장)
python train_tokenizer.py \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --vocab_size 128000 \
    --output_dir tokenizers/ko-128k

# 또는 로컬 텍스트 파일 사용
python train_tokenizer.py \
    --input_files data/pretrain/*.txt \
    --vocab_size 128000 \
    --output_dir tokenizers/custom
```

### Step 2: 사전학습 (수일~수주)

```bash
# 한국어 Wikipedia (기본)
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --output_dir outputs/pretrain-ko \
    --bf16 \
    --gradient_checkpointing

# 영어 C4 (대규모, 멀티 GPU)
torchrun --nproc_per_node=8 train.py \
    --mode pretrain \
    --dataset allenai/c4 \
    --dataset_config en \
    --output_dir outputs/pretrain-c4 \
    --bf16 \
    --gradient_checkpointing
```

### Step 3: SFT 파인튜닝 (수시간~1일)

```bash
# BCCard 금융 Q&A
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-finance \
    --num_epochs 3

# Alpaca 범용 Q&A
python train.py \
    --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-alpaca \
    --num_epochs 3

# 로컬 JSONL 파일
python train.py \
    --mode sft \
    --train_file data/sft/custom.jsonl \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-custom \
    --num_epochs 3
```

### Step 4: 테스트 및 배포

```bash
# 대화형 테스트
python chat.py --model_path outputs/sft-finance/final_model

# 추론 성능 테스트
python test_inference.py --model_path outputs/sft-finance/final_model

# HuggingFace Hub에 업로드
python -c "
from moai_llm.modeling.model import MoaiForCausalLM
model = MoaiForCausalLM.from_pretrained('outputs/sft-finance/final_model')
model.push_to_hub('your-username/moai-llm-finance')
"
```

---

## 📊 추천 데이터셋

### 토크나이저 학습용

| 데이터셋 | 크기 | 명령어 |
|---------|------|--------|
| Wikipedia (한국어) | 1GB | `--dataset wikimedia/wikipedia --dataset_config 20231101.ko` |
| Wikipedia (영어) | 20GB | `--dataset wikimedia/wikipedia --dataset_config 20231101.en` |
| C4 (영어) | 300GB | `--dataset allenai/c4 --dataset_config en` |

### 사전학습용

| 데이터셋 | 크기 | 용도 |
|---------|------|------|
| WikiText-2 | 4MB | 빠른 테스트 (`wikitext-2-raw-v1`) |
| Wikipedia (한국어) | 1GB | 한국어 일반 지식 |
| C4 (영어) | 300GB | 영어 범용 |

### SFT용

| 데이터셋 | 샘플 | 도메인 |
|---------|------|--------|
| BCCard | 4K | 금융 Q&A |
| Alpaca | 52K | 범용 Instruction |
| KULLM-v2 | 150K | 한국어 범용 |

**더 많은 데이터셋**: `DATASETS.md` 참고

---

## 🎯 시나리오별 완전 가이드

### 시나리오 1: 한국어 금융 챗봇

```bash
# 1. 토크나이저 (한국어)
python train_tokenizer.py \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --output_dir tokenizers/ko

# 2. 사전학습 (한국어)
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --output_dir outputs/pretrain-ko \
    --bf16

# 3. SFT (금융)
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-finance

# 4. 테스트
python chat.py --model_path outputs/sft-finance/final_model
```

### 시나리오 2: 영어 범용 모델

```bash
# 1. 토크나이저 (영어)
python train_tokenizer.py \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.en \
    --output_dir tokenizers/en

# 2. 사전학습 (C4)
torchrun --nproc_per_node=8 train.py \
    --mode pretrain \
    --dataset allenai/c4 \
    --dataset_config en \
    --output_dir outputs/pretrain-c4

# 3. SFT (Alpaca)
python train.py \
    --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-alpaca

# 4. 테스트
python chat.py --model_path outputs/sft-alpaca/final_model
```

### 시나리오 3: 도메인 특화 (의료/법률)

```bash
# 1. 기존 토크나이저 업데이트 (도메인 용어 추가)
python train_tokenizer.py \
    --dataset your-org/medical-corpus \
    --base_tokenizer tokenizers/ko \
    --output_dir tokenizers/ko-medical

# 2. 도메인 데이터 사전학습
python train.py \
    --mode pretrain \
    --dataset your-org/medical-corpus \
    --output_dir outputs/pretrain-medical \
    --bf16

# 3. 도메인 SFT
python train.py \
    --mode sft \
    --train_file data/medical_qa.jsonl \
    --pretrained_model outputs/pretrain-medical/final_model \
    --output_dir outputs/sft-medical
```

---

## 🛠️ 주요 옵션 요약

### 공통 옵션

```bash
--bf16                      # BF16 혼합 정밀도 (권장, A100/H100)
--fp16                      # FP16 혼합 정밀도 (V100/RTX)
--gradient_checkpointing    # 메모리 절약 (필수)
--batch_size 4              # 배치 크기 (GPU 메모리에 맞게)
--max_steps 10000           # 최대 스텝 (테스트용)
```

### 토크나이저 옵션

```bash
--dataset wikimedia/wikipedia   # HuggingFace 데이터셋
--dataset_config 20231101.ko    # 데이터셋 설정
--input_files data/*.txt        # 로컬 파일
--vocab_size 128000             # 어휘 크기
--max_samples 10000             # 테스트용 샘플 제한
--base_tokenizer path/          # 기존 토크나이저 업데이트
```

### 사전학습 옵션

```bash
--mode pretrain                 # 사전학습 모드
--dataset wikimedia/wikipedia   # 데이터셋
--dataset_config 20231101.ko    # 설정
--output_dir outputs/           # 출력 디렉토리
```

### SFT 옵션

```bash
--mode sft                  # SFT 모드
--dataset BCCard/...        # HuggingFace 데이터셋
--train_file custom.jsonl   # 로컬 JSONL 파일
--pretrained_model path/    # 사전학습 모델 경로
--num_epochs 3              # 에폭 수
```

---

## 💡 핵심 요점

### ✅ 자동화된 것들

- **데이터 다운로드**: HuggingFace에서 자동
- **포맷 변환**: input/output, instruction/output 등 자동 감지
- **학습 재개**: 체크포인트에서 자동 재개
- **모델 저장**: 최종 모델 및 체크포인트 자동 저장

### ❌ 필요 없는 것들

- ~~텍스트 파일 수동 준비~~
- ~~복잡한 데이터 전처리 스크립트~~
- ~~포맷 변환 코드 작성~~
- ~~학습 파라미터 복잡한 설정~~

---

## 📚 추가 문서

| 문서 | 내용 |
|------|------|
| `USER_GUIDE.md` | 완전한 학습 가이드 (환경 설정부터 배포까지) |
| `DATASETS.md` | 데이터셋 선택 및 설정 가이드 |
| `ARCHITECTURE.md` | Qwen3 아키텍처 상세 설명 |
| `TOKENIZER_UPDATE_GUIDE.md` | 토크나이저 업데이트 방법 |
| `examples/bccard_example.md` | BCCard 데이터셋 완전 예제 |

---

## 🚨 트러블슈팅

### 메모리 부족 (OOM)

```bash
# 1. Gradient checkpointing 활성화
--gradient_checkpointing

# 2. 배치 크기 줄이기
--batch_size 1 --gradient_accumulation_steps 4

# 3. FP16 사용
--fp16
```

### 데이터셋 설정(config) 찾기

```bash
# 데이터셋 정보 확인 도구
python check_dataset.py wikimedia/wikipedia
python check_dataset.py allenai/c4
```

### Dataset scripts 에러

```
RuntimeError: Dataset scripts are no longer supported, but found wikipedia.py
```

최신 `datasets` 라이브러리(3.x)에서는 커스텀 스크립트 기반 데이터셋이 지원되지 않습니다. 
기존 `wikipedia` 대신 `wikimedia/wikipedia` 데이터셋을 사용하세요:

```bash
# 기존 (지원 안됨)
--dataset wikipedia --dataset_config 20220301.ko

# 새로운 방식 (권장)
--dataset wikimedia/wikipedia --dataset_config 20231101.ko
```

### 학습 재개

```bash
# 자동으로 마지막 체크포인트에서 재개됨
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --output_dir outputs/pretrain  # 같은 디렉토리 지정
```

### Wikipedia 데이터셋 에러 (Dataset scripts are no longer supported)

최신 `datasets` 라이브러리에서 `wikipedia` 데이터셋 로드 시 에러가 발생할 수 있습니다. 코드가 자동으로 처리하지만, 문제가 계속되면:

**해결 방법 1: 다른 데이터셋 사용 (권장)**
```bash
# mC4 한국어 데이터셋 사용
python train_tokenizer.py \
    --dataset allenai/c4 \
    --dataset_config ko \
    --vocab_size 128000 \
    --output_dir tokenizers/ko
```

**해결 방법 2: 로컬 파일 사용**
```bash
python train_tokenizer.py \
    --input_files data/*.txt \
    --vocab_size 128000 \
    --output_dir tokenizers/custom
```

**해결 방법 3: datasets 라이브러리 다운그레이드 (임시)**
```bash
pip install "datasets<4.0.0"
```

---

## 🎉 지금 바로 시작!

### 첫 실행 (10분 테스트)

```bash
python train_tokenizer.py --dataset wikimedia/wikipedia --dataset_config 20231101.ko --vocab_size 32000 --max_samples 10000 --output_dir tokenizers/test

python train.py --mode pretrain --dataset wikitext --dataset_config wikitext-2-raw-v1 --output_dir outputs/test --max_steps 100

python chat.py --model_path outputs/test/final_model
```

### 실전 학습 (프로덕션)

```bash
# 전체 가이드 읽기
cat USER_GUIDE.md

# 실전 학습 시작
python train_tokenizer.py --dataset wikimedia/wikipedia --dataset_config 20231101.ko --output_dir tokenizers/ko

python train.py --mode pretrain --dataset wikimedia/wikipedia --dataset_config 20231101.ko --output_dir outputs/pretrain --bf16 --gradient_checkpointing
```

---

**🚀 MOAI-LLM으로 나만의 언어 모델을 만들어보세요!**
