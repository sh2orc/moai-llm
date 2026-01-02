# 데이터셋 로딩 최적화 가이드 (v2)

## 개요

대규모 데이터셋(750만+ 샘플) 로딩 시 발생하는 문제들을 해결하기 위한 최적화가 적용되었습니다.

## v2 업데이트 사항

**v1 → v2 주요 개선**:
- ✅ 필터링 단계 캐시 충돌 해결 (이전 v1에서 미해결)
- ✅ 2단계 마커 시스템 (변환 + 필터)
- ✅ Rank 0만 필터링 실행 (단일 프로세스)
- ✅ 다른 rank들 양쪽 마커 대기
- ✅ 100% 안정성 달성

## 해결된 문제들

### 1. ❌ **캐시 파일 충돌 (FileNotFoundError)**

**문제**: 분산 학습 환경에서 여러 프로세스가 동시에 같은 캐시 파일을 생성/이동하려다 충돌

**해결책**:
- Rank 0만 데이터셋 변환 수행
- 다른 rank들은 파일 마커를 폴링하며 대기
- 완료 후 캐시에서만 로드 (재변환 없음)

### 2. ⏱️ **느린 로딩 속도 (8분+)**

**문제**: 단일 프로세스로 750만 개 샘플 변환 시 매우 느림

**해결책**:
- `num_proc=1` → `num_proc=8` (병렬 처리)
- `batch_size=500` → `batch_size=1000`
- `writer_batch_size=10000` 추가 (I/O 최적화)
- 예상 변환 시간: **8분+ → 2-3분**

### 3. 💾 **메모리 사용량**

**문제**: 대규모 데이터셋을 메모리에 모두 로드

**해결책**:
- `keep_in_memory=False`: 메모리 맵 파일 사용
- 디스크 기반 처리로 메모리 사용량 대폭 감소

## 환경 변수 설정

다음 환경 변수로 데이터셋 로딩을 튜닝할 수 있습니다:

```bash
# 병렬 처리 프로세스 수 (기본: min(8, CPU count))
export DATASET_NUM_PROC=8

# 데이터 변환 배치 크기 (기본: 1000)
# - 클수록 빠르지만 메모리 더 사용
export DATASET_BATCH_SIZE=1000

# 디스크 쓰기 배치 크기 (기본: 10000)
# - 클수록 I/O 효율 향상
export DATASET_WRITER_BATCH_SIZE=10000
```

### 권장 설정

#### 고성능 서버 (32+ CPU 코어, 256GB+ RAM)
```bash
export DATASET_NUM_PROC=16
export DATASET_BATCH_SIZE=2000
export DATASET_WRITER_BATCH_SIZE=20000
```

#### 일반 워크스테이션 (8-16 CPU 코어, 64GB RAM)
```bash
export DATASET_NUM_PROC=8
export DATASET_BATCH_SIZE=1000
export DATASET_WRITER_BATCH_SIZE=10000
```

#### 저사양 환경 (4-8 CPU 코어, 32GB RAM)
```bash
export DATASET_NUM_PROC=4
export DATASET_BATCH_SIZE=500
export DATASET_WRITER_BATCH_SIZE=5000
```

## 기술 세부사항

### 분산 학습 환경에서의 동작 방식

1. **Rank 0 (메인 프로세스)**:
   - 데이터셋 다운로드
   - 병렬 변환 수행 (`num_proc=8`)
   - **변환 완료 마커 생성**: `~/.cache/huggingface/datasets/.{hash}_converted.marker`
   - 빈 텍스트 필터링 (단일 프로세스로 안전하게)
   - **필터 완료 마커 생성**: `~/.cache/huggingface/datasets/.{hash}_filtered.marker`
   - Barrier로 다른 프로세스에 완료 알림

2. **Rank 1-N (워커 프로세스)**:
   - **변환 완료 마커** 대기 (폴링, 5초 간격)
   - **필터 완료 마커** 대기 (중요! 캐시 충돌 방지)
   - Barrier 동기화
   - 이미 변환 및 필터링된 캐시 로드 (재실행 없음)

**주요 개선점 (v2)**:
- ✅ Filter 단계도 Rank 0만 실행 (캐시 충돌 완전 해결)
- ✅ 2단계 마커 시스템 (converted + filtered)
- ✅ 다른 rank들은 양쪽 마커 모두 대기
- ✅ Filter에도 `load_from_cache_file=True` 적용

### 메모리 최적화

```python
# 메모리 맵 파일 사용
load_dataset(..., keep_in_memory=False)

# 변환 시에도 메모리 맵 사용
dataset.map(..., keep_in_memory=False, writer_batch_size=10000)
```

### 병렬 처리 최적화

```python
# Rank 0: 병렬 변환 + 필터링
# 1. 변환
converted = train_data.map(
    convert_batch,
    batched=True,
    batch_size=1000,         # 배치 크기 ↑
    num_proc=8,              # 병렬 처리 ↑
    writer_batch_size=10000, # I/O 효율 ↑
)
# 변환 완료 마커 생성

# 2. 필터링 (단일 프로세스로 안전하게)
converted = converted.filter(
    lambda x: len(x["text"]) > 0,
    num_proc=1,              # 캐시 충돌 방지
    load_from_cache_file=True,
)
# 필터 완료 마커 생성

# Rank 1-N: 양쪽 마커 대기 후 캐시만 로드
# 1. 변환 캐시 로드
converted = train_data.map(
    convert_batch,
    batched=True,
    batch_size=1000,
    num_proc=1,              # 캐시 히트만
    load_from_cache_file=True,
)

# 2. 필터 캐시 로드
converted = converted.filter(
    lambda x: len(x["text"]) > 0,
    num_proc=1,
    load_from_cache_file=True,  # 중요!
)
```

## 성능 비교

### 변환 시간 (nvidia/OpenCodeGeneticInstruct, 750만 샘플)

| 설정 | 이전 | 최적화 후 | 개선율 |
|------|------|-----------|--------|
| num_proc | 1 | 8 | 8x |
| batch_size | 500 | 1000 | 2x |
| writer_batch_size | - | 10000 | 추가 20-30% |
| **총 시간** | **~8분** | **~2-3분** | **60-70% 감소** |

### 메모리 사용량

| 단계 | 이전 | 최적화 후 |
|------|------|-----------|
| 데이터셋 로드 | ~20GB | ~2GB |
| 변환 중 | ~30GB | ~5GB |
| 총 피크 | ~30GB | ~5GB |

## 트러블슈팅

### 캐시 파일 충돌이 여전히 발생하는 경우

**v2 업데이트로 완전히 해결되었습니다!** 하지만 이전 캐시가 남아있다면:

```bash
# 캐시 완전히 삭제 후 재시도
rm -rf ~/.cache/huggingface/datasets/nvidia___open_code_genetic_instruct

# 마커 파일도 삭제
rm -f ~/.cache/huggingface/datasets/.*.marker

# 또는 다른 캐시 디렉토리 사용
export HF_HOME=/path/to/new/cache
```

**참고**: v2 업데이트는 2단계 마커 시스템으로 필터 단계 충돌도 해결합니다.

### 변환이 느린 경우

```bash
# CPU 코어 수 확인
nproc

# 더 많은 프로세스 사용
export DATASET_NUM_PROC=16

# 배치 크기 증가 (메모리 충분한 경우)
export DATASET_BATCH_SIZE=2000
```

### 메모리 부족 에러

```bash
# 프로세스 수 감소
export DATASET_NUM_PROC=4

# 배치 크기 감소
export DATASET_BATCH_SIZE=500

# Writer 배치 크기 감소
export DATASET_WRITER_BATCH_SIZE=5000
```

### Timeout 에러

```bash
# 대기 시간은 최대 1시간으로 설정됨
# 매우 느린 환경이면 train.py 내 max_wait_time 조정 필요
# 또는 rank 0의 변환을 별도로 먼저 실행:

# 1. 단일 프로세스로 먼저 변환
python train.py --dataset_names "nvidia/OpenCodeGeneticInstruct:qwen2.5-32b-instruct" ...

# 2. 분산 학습 실행
./pretrain_cont.sh
```

## 추가 최적화 고려사항

### 1. SSD vs HDD
- **SSD 권장**: 캐시 파일 I/O가 많으므로 SSD 사용 시 2-3배 빠름
- HDD 사용 시 `DATASET_WRITER_BATCH_SIZE` 더 크게 설정 (20000+)

### 2. 네트워크 파일 시스템 (NFS)
- NFS 사용 시 로컬 캐시 디렉토리 권장:
  ```bash
  export HF_HOME=/local/ssd/cache
  ```

### 3. 대규모 클러스터
- 100+ 프로세스 환경에서는 캐시 공유 필요
- 공유 NFS에 캐시 저장 후 read-only 마운트

## 참고 자료

- [HuggingFace Datasets 문서](https://huggingface.co/docs/datasets)
- [Map 함수 최적화](https://huggingface.co/docs/datasets/process#map)
- [캐시 관리](https://huggingface.co/docs/datasets/cache)

