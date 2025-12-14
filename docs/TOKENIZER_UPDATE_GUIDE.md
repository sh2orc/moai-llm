# 🔄 토크나이저 업데이트 가이드

## 📌 개요

기존 토크나이저에 **새로운 도메인의 데이터를 추가**하여 어휘를 확장하는 방법입니다.

---

## 🤔 왜 업데이트가 필요한가?

### 사용 사례

1. **도메인 확장**
   - 일반 텍스트로 학습한 토크나이저에 **의료 용어** 추가
   - 한국어 토크나이저에 **금융 전문 용어** 추가
   - 영어 토크나이저에 **코드 토큰** 추가

2. **언어 추가**
   - 한국어 토크나이저에 **영어 어휘** 추가
   - 영어 토크나이저에 **중국어 어휘** 추가

3. **신조어 대응**
   - 2022년 토크나이저에 **2024년 신조어** 추가
   - 기존 토크나이저에 **최신 기술 용어** 추가

---

## ⚖️ 처음부터 vs 업데이트

### 처음부터 학습 (From Scratch)

```bash
# 예: Wikipedia만으로 토크나이저 학습
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 128000 \
    --output_dir tokenizers/
```

**장점:**
- ✅ 깔끔하고 일관된 어휘
- ✅ 중복 없는 최적화된 구조

**단점:**
- ❌ 기존 모델과 호환 불가 (토큰 ID가 바뀜)
- ❌ 처음부터 다시 학습해야 함

---

### 업데이트 (Update)

```bash
# 예: Wikipedia 토크나이저에 금융 데이터 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_tokenizer.model \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 150000 \
    --output_dir tokenizers/updated/
```

**장점:**
- ✅ 기존 어휘 유지 (기존 모델 활용 가능)
- ✅ 새 도메인에 특화된 토큰 추가
- ✅ 점진적 확장 가능

**단점:**
- ❌ vocab_size 증가 (메모리 사용량 증가)
- ❌ 기존 데이터 재구성 필요

---

## 📋 사용 방법

### 1. 기본 업데이트

```bash
# Step 1: 기존 토크나이저로 일반 모델 학습
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 128000 \
    --output_dir tokenizers/base/

# Step 2: 금융 데이터 추가하여 업데이트
python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 150000 \
    --output_dir tokenizers/finance/
```

**결과:**
- 기존 vocab: 128,000개 → 새 vocab: 150,000개
- 기존 일반 어휘 유지 + 금융 용어 22,000개 추가

---

### 2. 의료 도메인 추가

```bash
# Step 1: 기존 한국어 토크나이저
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 100000 \
    --output_dir tokenizers/base/

# Step 2: 의료 데이터 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset medical_dataset \
    --vocab_size 120000 \
    --output_dir tokenizers/medical/
```

---

### 3. 코드 토큰 추가

```bash
# Step 1: 영어 텍스트 토크나이저
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 100000 \
    --output_dir tokenizers/text/

# Step 2: Python 코드 토큰 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/text/moai_tokenizer.model \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --vocab_size 130000 \
    --max_samples 100000 \
    --output_dir tokenizers/code/
```

---

### 4. 다국어 확장

```bash
# Step 1: 한국어 토크나이저
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 80000 \
    --output_dir tokenizers/korean/

# Step 2: 영어 어휘 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/korean/moai_tokenizer.model \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 150000 \
    --max_samples 500000 \
    --output_dir tokenizers/bilingual/
```

---

## 🔍 내부 동작 원리

### 업데이트 프로세스

```python
# 1. 기존 토크나이저 로드
base_tokenizer = load("tokenizers/base/moai_tokenizer.model")

# 2. 기존 어휘 샘플 추출
existing_vocab = extract_vocabulary_samples(base_tokenizer)
# → ['안녕', '하세요', '세계', '컴퓨터', ...]

# 3. 새 데이터 추가
new_data = load_dataset("BCCard/BCCard-Finance-Kor-QnA")
# → ['대출', '이자율', '신용', '담보', ...]

# 4. 데이터 병합
merged_data = existing_vocab + new_data
# → ['안녕', '하세요', ..., '대출', '이자율', ...]

# 5. 새 토크나이저 학습 (vocab_size 증가)
train_tokenizer(merged_data, vocab_size=150000)
```

### 병합 비율

- **기존 어휘**: 기존 vocab에서 최대 10,000개 샘플 추출
- **새 데이터**: 전체 새 데이터셋 사용
- **자동 균형**: SentencePiece가 빈도수 기반으로 최적 어휘 생성

---

## 💡 실전 시나리오

### 시나리오 1: 일반 → 금융 특화

```bash
# 1. 일반 한국어 토크나이저 (Wikipedia)
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 100000 \
    --output_dir tokenizers/general/

# 2. 금융 데이터 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/general/moai_tokenizer.model \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 120000 \
    --output_dir tokenizers/finance/

# 3. 금융 특화 모델 학습
python train.py \
    --mode pretrain \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --tokenizer_path tokenizers/finance/moai_tokenizer.model \
    --output_dir outputs/finance-pretrain
```

---

### 시나리오 2: 영어 → 코드 + 영어

```bash
# 1. 영어 토크나이저
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 80000 \
    --output_dir tokenizers/english/

# 2. 코드 토큰 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/english/moai_tokenizer.model \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --vocab_size 120000 \
    --max_samples 200000 \
    --output_dir tokenizers/code-en/

# 3. 코드 + 영어 모델 학습
python train.py \
    --mode pretrain \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --tokenizer_path tokenizers/code-en/moai_tokenizer.model \
    --output_dir outputs/code-pretrain
```

---

### 시나리오 3: 한국어 → 한영 바이링궐

```bash
# 1. 한국어 베이스
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --vocab_size 100000 \
    --output_dir tokenizers/korean/

# 2. 영어 추가
python train_tokenizer.py \
    --base_tokenizer tokenizers/korean/moai_tokenizer.model \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --vocab_size 180000 \
    --max_samples 1000000 \
    --output_dir tokenizers/bilingual/

# 3. 바이링궐 모델 학습
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --tokenizer_path tokenizers/bilingual/moai_tokenizer.model \
    --output_dir outputs/bilingual-pretrain
```

---

## ⚠️ 주의사항

### 1. Vocab Size 증가 필수

```bash
# ❌ 잘못된 예 (기존과 같은 크기)
python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset new_data \
    --vocab_size 128000 \  # 기존과 동일
    --output_dir tokenizers/updated/

# ✅ 올바른 예 (크기 증가)
python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset new_data \
    --vocab_size 150000 \  # 기존보다 큼
    --output_dir tokenizers/updated/
```

**이유**: 기존 어휘를 유지하면서 새 어휘를 추가하려면 vocab_size가 커야 합니다.

---

### 2. 기존 모델과의 호환성

업데이트된 토크나이저는 **새로 학습해야 합니다**:

```bash
# ❌ 기존 모델에 업데이트된 토크나이저 사용 불가
python train.py \
    --mode sft \
    --pretrained_model outputs/old-model/ \  # 기존 토크나이저로 학습됨
    --tokenizer_path tokenizers/updated/moai_tokenizer.model \  # 새 토크나이저
    --output_dir outputs/sft  # 토큰 ID 불일치!

# ✅ 새 토크나이저로 처음부터 학습
python train.py \
    --mode pretrain \
    --tokenizer_path tokenizers/updated/moai_tokenizer.model \
    --output_dir outputs/new-pretrain
```

---

### 3. 메모리 고려

- vocab_size 증가 → 임베딩 레이어 크기 증가 → 메모리 사용량 증가

```python
# 예시 계산
# vocab_size=128000, hidden_size=2048
embedding_params = 128000 * 2048 = 262M parameters (약 1GB)

# vocab_size=200000으로 증가
embedding_params = 200000 * 2048 = 410M parameters (약 1.6GB)
```

**권장**: vocab_size는 필요한 만큼만 증가시키세요.

---

## 📊 비교표

| 방식 | Vocab Size | 학습 시간 | 기존 모델 | 새 도메인 | 추천 상황 |
|------|-----------|---------|---------|----------|----------|
| **처음부터** | 고정 | 짧음 | ❌ 호환 안됨 | ✅ 최적화 | 새 프로젝트 |
| **업데이트** | 증가 | 약간 김 | ✅ 유지 | ✅ 추가 | 도메인 확장 |

---

## 🎯 빠른 참조

### 업데이트 명령어 템플릿

```bash
python train_tokenizer.py \
    --base_tokenizer <기존_토크나이저_경로> \
    --dataset <새_데이터셋> \
    --dataset_config <설정> \
    --vocab_size <기존보다_큰_값> \
    --output_dir <출력_경로>
```

### 예시

```bash
# 금융
python train_tokenizer.py --base_tokenizer tokenizers/base/moai_tokenizer.model --dataset BCCard/BCCard-Finance-Kor-QnA --vocab_size 150000 --output_dir tokenizers/finance/

# 의료
python train_tokenizer.py --base_tokenizer tokenizers/base/moai_tokenizer.model --dataset medical_dataset --vocab_size 150000 --output_dir tokenizers/medical/

# 코드
python train_tokenizer.py --base_tokenizer tokenizers/base/moai_tokenizer.model --dataset bigcode/the-stack --dataset_config data/python --vocab_size 150000 --output_dir tokenizers/code/
```

---

## 📚 더 알아보기

- **기본 토크나이저 학습**: `QUICKSTART.md`
- **데이터셋 목록**: `POPULAR_DATASETS.md`
- **전체 학습 가이드**: `START_HERE.md`

---

**💡 핵심 정리:**
- ✅ 기존 어휘 유지하면서 새 도메인 추가
- ✅ vocab_size는 반드시 증가
- ✅ 업데이트된 토크나이저로 새 모델 학습 필요
