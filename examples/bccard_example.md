# BCCard Finance 데이터셋 사용 예시

## 📋 데이터셋 정보
- **이름**: BCCard/BCCard-Finance-Kor-QnA
- **샘플 수**: ~4,000개
- **언어**: 한국어
- **포맷**: input/output
- **용도**: 금융 도메인 SFT

---

## 🎯 전체 워크플로우

### 1단계: 토크나이저 학습 (한국어 포함)

```bash
# 한국어가 포함된 데이터로 토크나이저 학습
python scripts/train_tokenizer.py \
    --input_files data/tokenizer_train/*.txt \
    --vocab_size 128000 \
    --character_coverage 0.9995 \
    --output_dir tokenizers/
```

**중요**: 토크나이저는 일반 텍스트로 학습합니다 (Q&A 아님)
- Wikipedia 한국어
- 뉴스 텍스트
- 웹 크롤링 데이터

---

### 2단계: 사전학습 (일반 언어 능력)

```bash
# 한국어 Wikipedia로 사전학습
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir outputs/pretrain-korean \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --max_steps 10000 \
    --bf16 \
    --gradient_checkpointing
```

**또는 영어+한국어 혼합:**
```bash
# 1. 영어 Wikipedia
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --output_dir outputs/pretrain-en \
    --max_steps 50000

# 2. 한국어 Wikipedia (이어서)
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --pretrained_model outputs/pretrain-en/final_model \
    --output_dir outputs/pretrain-en-ko \
    --max_steps 20000
```

---

### 3단계: SFT with BCCard 데이터셋 ⭐

```bash
# BCCard 데이터셋으로 금융 도메인 파인튜닝
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
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
    --dataset BCCard/BCCard-Finance-Kor-QnA \
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

dataset = load_dataset("BCCard/BCCard-Finance-Kor-QnA")
print(dataset["train"][0])
# {'input': '질문...', 'output': '답변...'}
```

### 2. 여러 SFT 데이터셋 순차 학습

```bash
# 1. 일반 한국어 SFT
python train.py \
    --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-korean/final_model \
    --output_dir outputs/sft-general-korean

# 2. 금융 도메인 특화 (이어서)
python train.py \
    --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/sft-general-korean/final_model \
    --output_dir outputs/sft-finance-korean
```

---

## ✅ 전체 프로세스 요약

```
1. 토크나이저 학습
   ↓ (일반 한국어 텍스트)

2. 사전학습 (Pretrain)
   ↓ (Wikipedia 등)

3. SFT (Fine-tuning)
   ↓ (BCCard 데이터셋)

4. 금융 Q&A 모델 완성! 🎉
```

---

## 🎯 핵심 정리

| 단계 | 데이터 타입 | 포맷 | 데이터셋 예시 |
|------|------------|------|--------------|
| **토크나이저** | 일반 텍스트 | Plain text | Wikipedia, 뉴스 |
| **사전학습** | 일반 텍스트 | Plain text | Wikipedia, C4 |
| **SFT** | Q&A | input/output | **BCCard** ✅ |

**BCCard 데이터셋은 3단계 SFT에서만 사용합니다!**
