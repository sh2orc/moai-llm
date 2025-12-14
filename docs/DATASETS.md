# 📊 MOAI-LLM 데이터셋 완전 가이드

**HuggingFace Datasets 사용법과 추천 데이터셋 목록**

---

## 📋 목차

1. [Dataset Config란?](#1-dataset-config란)
2. [주요 데이터셋별 Config](#2-주요-데이터셋별-config)
3. [Config 확인 방법](#3-config-확인-방법)
4. [추천 데이터셋 목록](#4-추천-데이터셋-목록)
5. [데이터셋 조합 추천](#5-데이터셋-조합-추천)

---

## 1. Dataset Config란?

### 1.1 개념

**데이터셋의 하위 버전/언어/설정을 지정하는 것입니다.**

```bash
--dataset wikipedia           # 데이터셋 이름
--dataset_config 20220301.ko  # 설정 (2022년 3월, 한국어)
```

### 1.2 왜 필요한가?

많은 HuggingFace 데이터셋은 **여러 언어, 버전, 설정**을 포함합니다:
- Wikipedia: 300+ 언어
- C4: 다국어 버전
- The Stack: 프로그래밍 언어별

Config를 지정하지 않으면 에러가 발생합니다.

---

## 2. 주요 데이터셋별 Config

### 2.1 Wikipedia

**Format**: `날짜.언어코드`

| 언어 | Config | 크기 | 용도 |
|-----|--------|------|------|
| 한국어 | `20220301.ko` | ~1GB | 한국어 모델 |
| 영어 | `20220301.en` | ~20GB | 영어 모델 |
| 중국어 | `20220301.zh` | ~3GB | 중국어 모델 |
| 일본어 | `20220301.ja` | ~2GB | 일본어 모델 |

**사용 예시:**
```bash
# 토크나이저 학습
python train_tokenizer.py \
    --dataset wikipedia \
    --dataset_config 20220301.ko \
    --output_dir tokenizers/

# 사전학습
python train.py \
    --mode pretrain \
    --dataset wikipedia \
    --dataset_config 20220301.en \
    --output_dir outputs/pretrain
```

---

### 2.2 C4 (Common Crawl)

**Format**: 언어코드

| 언어 | Config | 크기 | 용도 |
|-----|--------|------|------|
| 영어 | `en` | ~300GB | 대규모 영어 사전학습 |
| 다국어 | `multilingual` | 대용량 | 다국어 모델 |

**사용 예시:**
```bash
python train.py \
    --mode pretrain \
    --dataset allenai/c4 \
    --dataset_config en \
    --output_dir outputs/pretrain
```

---

### 2.3 WikiText

**Format**: 버전명

| 버전 | Config | 크기 | 용도 |
|-----|--------|------|------|
| WikiText-2 | `wikitext-2-raw-v1` | 4MB | 빠른 테스트 |
| WikiText-103 | `wikitext-103-raw-v1` | 500MB | 실험용 |

**사용 예시:**
```bash
# 빠른 테스트 (10분)
python train.py \
    --mode pretrain \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --output_dir outputs/test \
    --max_steps 100
```

---

### 2.4 The Stack (코드)

**Format**: `data/언어`

| 언어 | Config | 크기 | 용도 |
|-----|--------|------|------|
| Python | `data/python` | 대용량 | 코드 생성 모델 |
| JavaScript | `data/javascript` | 대용량 | JS 특화 |
| Java | `data/java` | 대용량 | Java 특화 |

**사용 예시:**
```bash
python train.py \
    --mode pretrain \
    --dataset bigcode/the-stack \
    --dataset_config data/python \
    --output_dir outputs/pretrain-code
```

---

### 2.5 Config 없는 데이터셋

다음 데이터셋들은 **config가 필요 없습니다**:

| 데이터셋 | Config | 사용법 |
|---------|--------|-------|
| BookCorpus | ❌ 불필요 | `--dataset bookcorpus` |
| BCCard Finance | ❌ 불필요 | `--dataset BCCard/BCCard-Finance-Kor-QnA` |
| Alpaca | ❌ 불필요 | `--dataset tatsu-lab/alpaca` |
| KULLM | ❌ 불필요 | `--dataset nlpai-lab/kullm-v2` |

**사용 예시:**
```bash
# Config 없이 사용
python train.py \
    --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain/final_model \
    --output_dir outputs/sft
```

---

## 3. Config 확인 방법

### 3.1 스크립트 사용 (권장)

```bash
# Wikipedia config 확인
python check_dataset.py wikipedia

# C4 config 확인
python check_dataset.py allenai/c4

# BCCard config 확인
python check_dataset.py BCCard/BCCard-Finance-Kor-QnA
```

### 3.2 Python 코드

```python
from datasets import get_dataset_config_names

# Wikipedia의 모든 config 확인
configs = get_dataset_config_names("wikipedia")

# 한국어 관련 config만 필터링
ko_configs = [c for c in configs if 'ko' in c]
print(ko_configs)
# ['20220301.ko']

# 영어 관련 config
en_configs = [c for c in configs if 'en' in c]
print(en_configs[:3])
# ['20220301.en', '20220301.en-simple', ...]
```

### 3.3 HuggingFace 웹사이트

1. https://huggingface.co/datasets 접속
2. 데이터셋 검색 (예: wikipedia)
3. "Viewer" 탭에서 "Configuration" 확인

---

## 4. 추천 데이터셋 목록

### 4.1 한국어 데이터셋

#### 사전학습용

| 데이터셋 | Config | 크기 | 설명 | 명령어 |
|---------|--------|------|------|--------|
| **Wikipedia** | `20220301.ko` | ~1GB | 한국어 백과사전 | `--dataset wikipedia --dataset_config 20220301.ko` |
| **mC4** | `ko` | ~수십GB | Common Crawl 한국어 | `--dataset allenai/c4 --dataset_config ko` |
| **OSCAR** | `unshuffled_deduplicated_ko` | 대용량 | Common Crawl 정제본 | `--dataset oscar --dataset_config unshuffled_deduplicated_ko` |

**사용 예시:**
```bash
# Wikipedia 한국어
python train.py --mode pretrain \
    --dataset wikipedia --dataset_config 20220301.ko \
    --output_dir outputs/pretrain-ko --bf16
```

#### SFT용

| 데이터셋 | 샘플 | 설명 | 명령어 |
|---------|------|------|--------|
| **KULLM** | 150K | 한국어 Q&A | `--dataset nlpai-lab/kullm-v2` |
| **KoAlpaca** | 52K | 한국어 Alpaca | `--dataset beomi/KoAlpaca-v1.1a` |
| **BCCard Finance** | 4K | 금융 Q&A | `--dataset BCCard/BCCard-Finance-Kor-QnA` |
| **KorQuAD** | 60K | 질의응답 | `--dataset squad_kor_v1` |

**사용 예시:**
```bash
# KULLM
python train.py --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-kullm --bf16
```

---

### 4.2 영어 데이터셋

#### 사전학습용

| 데이터셋 | Config | 크기 | 설명 | 명령어 |
|---------|--------|------|------|--------|
| **Wikipedia** | `20220301.en` | ~20GB | 영어 백과사전 | `--dataset wikipedia --dataset_config 20220301.en` |
| **C4** | `en` | ~300GB | Common Crawl 정제본 | `--dataset allenai/c4 --dataset_config en` |
| **RefinedWeb** | - | ~5TB | Falcon 사전학습 데이터 | `--dataset tiiuae/falcon-refinedweb` |
| **RedPajama** | - | ~1.2TB | LLaMA 복제 데이터 | `--dataset togethercomputer/RedPajama-Data-1T` |
| **BookCorpus** | - | ~5GB | 책 데이터 | `--dataset bookcorpus` |
| **The Pile** | - | ~800GB | 다양한 소스 혼합 | `--dataset EleutherAI/pile` |

**사용 예시:**
```bash
# C4 영어 (대규모)
python train.py --mode pretrain \
    --dataset allenai/c4 --dataset_config en \
    --output_dir outputs/pretrain-c4 --max_steps 100000 --bf16
```

#### SFT용

| 데이터셋 | 샘플 | 설명 | 명령어 |
|---------|------|------|--------|
| **Alpaca** | 52K | Stanford Alpaca | `--dataset tatsu-lab/alpaca` |
| **Dolly** | 15K | Databricks 고품질 | `--dataset databricks/databricks-dolly-15k` |
| **OpenAssistant** | 161K | RLHF 데이터 | `--dataset OpenAssistant/oasst1` |
| **ShareGPT** | 90K | ChatGPT 대화 | `--dataset RyokoAI/ShareGPT52K` |
| **LIMA** | 1K | 초고품질 (Less is More) | `--dataset GAIR/lima` |
| **HH-RLHF** | 169K | Anthropic RLHF | `--dataset Anthropic/hh-rlhf` |
| **Evol-Instruct** | 70K | WizardLM 데이터 | `--dataset WizardLM/WizardLM_evol_instruct_V2_196k` |
| **UltraChat** | 1.4M | 대규모 대화 | `--dataset stingning/ultrachat` |

**사용 예시:**
```bash
# Alpaca (범용)
python train.py --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-alpaca --bf16

# LIMA (고품질)
python train.py --mode sft \
    --dataset GAIR/lima \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-lima \
    --num_epochs 10 --learning_rate 1e-5 --bf16
```

---

### 4.3 코드 데이터셋

| 데이터셋 | Config | 크기 | 설명 | 명령어 |
|---------|--------|------|------|--------|
| **The Stack** | `data/python` | 대용량 | GitHub 코드 | `--dataset bigcode/the-stack --dataset_config data/python` |
| **StarCoder** | - | 대용량 | StarCoder 학습 데이터 | `--dataset bigcode/starcoderdata` |
| **Code Alpaca** | 20K | 코드 생성 Q&A | `--dataset sahil2801/CodeAlpaca-20k` |

**사용 예시:**
```bash
# 사전학습 (Python)
python train.py --mode pretrain \
    --dataset bigcode/the-stack --dataset_config data/python \
    --output_dir outputs/pretrain-code --bf16

# SFT (Code Alpaca)
python train.py --mode sft \
    --dataset sahil2801/CodeAlpaca-20k \
    --pretrained_model outputs/pretrain-code/final_model \
    --output_dir outputs/sft-code --bf16
```

---

### 4.4 다국어 데이터셋

| 데이터셋 | Config | 크기 | 언어 | 명령어 |
|---------|--------|------|------|--------|
| **mC4** | `multilingual` | 대용량 | 100+ 언어 | `--dataset allenai/c4 --dataset_config multilingual` |
| **OSCAR** | `unshuffled_deduplicated_*` | 대용량 | 150+ 언어 | `--dataset oscar --dataset_config unshuffled_deduplicated_*` |
| **CulturaX** | - | 6.3T tokens | 167 언어 | `--dataset uonlp/CulturaX` |

---

## 5. 데이터셋 조합 추천

### 5.1 한국어 범용 모델

```bash
# 토크나이저
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.ko \
    --vocab_size 128000 --output_dir tokenizers/ko/

# 사전학습 (Wikipedia)
python train.py --mode pretrain \
    --dataset wikipedia --dataset_config 20220301.ko \
    --output_dir outputs/pretrain-ko --num_epochs 3 --bf16

# SFT (KULLM - 150K 샘플)
python train.py --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-ko/final_model \
    --output_dir outputs/sft-ko --num_epochs 3 --bf16
```

**예상 성능**: 한국어 일반 지식 + Q&A 능력

---

### 5.2 영어 고성능 모델

```bash
# 토크나이저
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.en \
    --vocab_size 128000 --output_dir tokenizers/en/

# 사전학습 (C4 - 300GB)
python train.py --mode pretrain \
    --dataset allenai/c4 --dataset_config en \
    --output_dir outputs/pretrain-c4 --max_steps 100000 --bf16

# SFT 1단계 (Alpaca - 범용)
python train.py --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/pretrain-c4/final_model \
    --output_dir outputs/sft-stage1 --num_epochs 3 --bf16

# SFT 2단계 (LIMA - 고품질 정제)
python train.py --mode sft \
    --dataset GAIR/lima \
    --pretrained_model outputs/sft-stage1/final_model \
    --output_dir outputs/sft-lima \
    --num_epochs 10 --learning_rate 1e-5 --bf16
```

**예상 성능**: GPT-3.5 수준의 고품질 응답

---

### 5.3 코드 생성 특화 모델

```bash
# 토크나이저
python train_tokenizer.py \
    --dataset bigcode/the-stack --dataset_config data/python \
    --vocab_size 128000 --max_samples 200000 --output_dir tokenizers/code/

# 사전학습 (The Stack Python)
python train.py --mode pretrain \
    --dataset bigcode/the-stack --dataset_config data/python \
    --output_dir outputs/pretrain-code --num_epochs 1 --bf16

# SFT (Code Alpaca)
python train.py --mode sft \
    --dataset sahil2801/CodeAlpaca-20k \
    --pretrained_model outputs/pretrain-code/final_model \
    --output_dir outputs/sft-code --num_epochs 3 --bf16
```

**예상 성능**: Python 코드 생성 특화 (GitHub Copilot 스타일)

---

### 5.4 다국어 모델 (한영 바이링궐)

```bash
# 토크나이저 (한국어 베이스 + 영어 추가)
python train_tokenizer.py \
    --dataset wikipedia --dataset_config 20220301.ko \
    --vocab_size 100000 --output_dir tokenizers/base/

python train_tokenizer.py \
    --base_tokenizer tokenizers/base/moai_tokenizer.model \
    --dataset wikipedia --dataset_config 20220301.en \
    --vocab_size 180000 --max_samples 1000000 --output_dir tokenizers/bilingual/

# 사전학습 (mC4 다국어)
python train.py --mode pretrain \
    --dataset allenai/c4 --dataset_config multilingual \
    --output_dir outputs/pretrain-multilingual --max_steps 50000 --bf16

# SFT (혼합: KULLM + Alpaca)
# Stage 1: 한국어
python train.py --mode sft \
    --dataset nlpai-lab/kullm-v2 \
    --pretrained_model outputs/pretrain-multilingual/final_model \
    --output_dir outputs/sft-stage1 --num_epochs 2 --bf16

# Stage 2: 영어
python train.py --mode sft \
    --dataset tatsu-lab/alpaca \
    --pretrained_model outputs/sft-stage1/final_model \
    --output_dir outputs/sft-bilingual --num_epochs 2 --bf16
```

**예상 성능**: 한국어 + 영어 모두 가능한 바이링궐 모델

---

## 6. 빠른 참조표

### Config 필수 여부

| 데이터셋 | Config 필요? | 예시 |
|---------|-----------|------|
| wikipedia | ✅ | `--dataset_config 20220301.ko` |
| allenai/c4 | ✅ | `--dataset_config en` |
| wikitext | ✅ | `--dataset_config wikitext-2-raw-v1` |
| bigcode/the-stack | ✅ | `--dataset_config data/python` |
| bookcorpus | ❌ | (생략) |
| tatsu-lab/alpaca | ❌ | (생략) |
| BCCard/BCCard-Finance-Kor-QnA | ❌ | (생략) |
| nlpai-lab/kullm-v2 | ❌ | (생략) |
| GAIR/lima | ❌ | (생략) |

### 데이터셋 크기 비교

| 데이터셋 | 크기 | 학습 시간 (A100 × 4) | 용도 |
|---------|------|-------------------|------|
| WikiText-2 | 4MB | 10분 | 빠른 테스트 |
| Wikipedia (ko) | ~1GB | ~1일 | 한국어 사전학습 |
| Wikipedia (en) | ~20GB | ~3일 | 영어 사전학습 |
| C4 (en) | ~300GB | ~수주 | 대규모 사전학습 |
| The Pile | ~800GB | ~수개월 | SOTA 사전학습 |

---

## 7. 문제 해결

### Config 에러

```bash
# ❌ ValueError: Config name is missing
python train.py --dataset wikipedia --output_dir outputs/

# ✅ 해결: Config 추가
python train.py --dataset wikipedia --dataset_config 20220301.ko --output_dir outputs/
```

### Config 확인

```bash
# 잘 모를 때 이 명령어로 확인
python check_dataset.py <dataset_name>
```

---

## 8. 더 알아보기

- **USER_GUIDE.md**: 완전한 학습 가이드
- **QUICKSTART.md**: 10분 빠른 시작
- **ARCHITECTURE.md**: 아키텍처 상세
- **examples/bccard_example.md**: BCCard 실전 예제

---

**🎉 이제 원하는 데이터셋을 선택하여 학습을 시작하세요!**

```bash
# 추천: 한국어 범용 모델
python train.py --mode pretrain --dataset wikipedia --dataset_config 20220301.ko --bf16
```
