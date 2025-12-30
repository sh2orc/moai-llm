# BCCard Finance 데이터셋 사용 예시

## 📋 데이터셋 정보
- **이름**: BCCard/BCAI-Finance-Kor
- **샘플 수**: ~100,000개
- **언어**: 한국어
- **포맷**: input/output
- **용도**: 금융 도메인 SFT

---

## 🎯 전체 워크플로우

### 1단계: 토크나이저 학습 (다국어)

```bash
# Step 1: 다국어 기본 토크나이저 (64K)
python train_tokenizer.py \
    --multilingual ko en ja zh \
    --vocab_size 64000 \
    --max_samples_per_lang 60000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_multilingual
```

**또는 로컬 파일로 학습:**
```bash
python train_tokenizer.py \
    --input_files data/tokenizer_train/*.txt \
    --vocab_size 64000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai
```

**중요**: 토크나이저는 일반 텍스트로 학습합니다 (Q&A 아님)
- Wikipedia 다국어
- 뉴스 텍스트
- 웹 크롤링 데이터

---

### 2단계: 토크나이저 확장 (금융 도메인)

```bash
# Step 2: Alpaca 한국어 데이터로 확장 (+16K → 80K)
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_multilingual \
    --dataset unoooo/alpaca-korean \
    --vocab_size 80000 \
    --max_samples 30000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_alpaca

# Step 3: 금융 데이터로 확장 (+16K → 96K)
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_alpaca \
    --dataset Mineru/kor-open-finance \
    --vocab_size 96000 \
    --max_samples 30000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_finance

# Step 4: BCCard 금융 데이터로 확장 (+32K → 128K)
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_finance \
    --dataset BCCard/BCAI-Finance-Kor \
    --vocab_size 128000 \
    --max_samples 100000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_finance_bccard
```

---

### 3단계: 사전학습 (일반 언어 능력)

```bash
# 한국어 Wikipedia로 사전학습
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --tokenizer_path tokenizers/moai \
    --output_dir outputs/pretrain-korean \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --bf16 \
    --gradient_checkpointing
```

**또는 영어+한국어 혼합:**
```bash
# 1. 영어 Wikipedia
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.en \
    --tokenizer_path tokenizers/moai \
    --output_dir outputs/pretrain-en \
    --max_steps 50000

# 2. 한국어 Wikipedia (이어서)
python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --tokenizer_path tokenizers/moai \
    --pretrained_model outputs/pretrain-en/final_model \
    --output_dir outputs/pretrain-en-ko \
    --max_steps 20000
```

---

### 4단계: SFT with BCCard 데이터셋 ⭐

```bash
# BCCard 데이터셋으로 금융 도메인 파인튜닝
python train.py \
    --mode sft \
    --dataset BCCard/BCAI-Finance-Kor \
    --tokenizer_path tokenizers/moai_finance_bccard \
    --pretrained_model outputs/pretrain-korean/final_model \
    --output_dir outputs/sft-bccard \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --bf16
```

**완료!** 이제 금융 도메인 Q&A 모델이 완성되었습니다.

---

## 🔍 데이터 포맷 자동 변환

**원본 데이터:**
```json
{
  "input": "신용카드 연회비는 어떻게 되나요?",
  "output": "연회비는 카드 종류에 따라 다릅니다. 일반 카드는 무료부터 5만원까지..."
}
```

**자동 변환 결과:**
```text
<|im_start|>user
신용카드 연회비는 어떻게 되나요?<|im_end|>
<|im_start|>assistant
연회비는 카드 종류에 따라 다릅니다. 일반 카드는 무료부터 5만원까지...<|im_end|>
```

`train.py`가 자동으로 변환합니다!

---

## 🚀 빠른 테스트

```bash
# 작은 샘플로 빠른 테스트
python train.py \
    --mode sft \
    --dataset BCCard/BCAI-Finance-Kor \
    --tokenizer_path tokenizers/moai_finance_bccard \
    --pretrained_model outputs/pretrain-korean/final_model \
    --output_dir outputs/test-bccard \
    --max_steps 100 \
    --batch_size 2
```

---

## 💡 팁

### 1. 데이터 미리보기

```python
from datasets import load_dataset

dataset = load_dataset("BCCard/BCAI-Finance-Kor")
print(dataset["train"][0])
# {'input': '질문...', 'output': '답변...'}
```

### 2. 여러 SFT 데이터셋 순차 학습

```bash
# 1. 일반 한국어 SFT
python train.py \
    --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --tokenizer_path tokenizers/moai_finance_bccard \
    --pretrained_model outputs/pretrain-korean/final_model \
    --output_dir outputs/sft-general-korean

# 2. 금융 도메인 특화 (이어서)
python train.py \
    --mode sft \
    --dataset BCCard/BCAI-Finance-Kor \
    --tokenizer_path tokenizers/moai_finance_bccard \
    --pretrained_model outputs/sft-general-korean/final_model \
    --output_dir outputs/sft-finance-korean
```

### 3. 토크나이저 학습 모드

| 모드 | 설명 | 속도 |
|------|------|------|
| 기본 | BPE, 높은 품질 | 1x |
| `--fast` | min_freq=5, limit_alphabet=10K | 10x |
| `--turbo` | min_freq=10, limit_alphabet=5K | 20x |
| `--ultrafast` | Unigram 알고리즘 (merge 없음) | 50x |

---

## ✅ 전체 프로세스 요약

```
1. 토크나이저 학습 (다국어)
   ↓ (Wikipedia 다국어)

2. 토크나이저 확장 (도메인 특화)
   ↓ (금융 데이터)

3. 사전학습 (Pretrain)
   ↓ (Wikipedia 등)

4. SFT (Fine-tuning)
   ↓ (BCCard 데이터셋)

5. 금융 Q&A 모델 완성! 🎉
```

---

## 🎯 핵심 정리

| 단계 | 데이터 타입 | 포맷 | 데이터셋 예시 |
|------|------------|------|--------------|
| **토크나이저** | 일반 텍스트 | Plain text | Wikipedia, 뉴스 |
| **토크나이저 확장** | 도메인 텍스트 | Plain text | 금융 데이터 |
| **사전학습** | 일반 텍스트 | Plain text | Wikipedia, C4 |
| **SFT** | Q&A | input/output | **BCCard** ✅ |

**BCCard 데이터셋은 토크나이저 확장과 SFT에서 사용합니다!**
