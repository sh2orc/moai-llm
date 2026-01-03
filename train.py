"""
MOAI-LLM 학습 스크립트 (통합 버전)

사전학습과 SFT를 하나의 스크립트로 처리합니다.

사용법:

python train.py \
    --mode pretrain \
    --dataset wikimedia/wikipedia \
    --dataset_config 20231101.ko \
    --tokenizer_path tokenizers/moai \
    --model_config configs/model_config_2b.json \
    --output_dir outputs/pretrain-korean-2b \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-6 \
    --max_seq_length 2048 \
    --bf16 \
    --gradient_checkpointing

    
    # 사전학습 - HuggingFace 데이터셋
    python train.py \
        --mode pretrain \
        --dataset wikipedia \
        --dataset_config 20220301.en \
        --output_dir outputs/pretrain

    # 사전학습 - 로컬 txt 파일
    python train.py \
        --mode pretrain \
        --train_file data/pretrain/train.txt \
        --output_dir outputs/pretrain

    # SFT - HuggingFace 데이터셋
    python train.py \
        --mode sft \
        --dataset tatsu-lab/alpaca \
        --output_dir outputs/sft

    # SFT - 로컬 JSON 파일
    python train.py \
        --mode sft \
        --train_file data/sft/alpaca.json \
        --output_dir outputs/sft
"""

# ============================================================================
# Constants and Configuration
# ============================================================================

# Timeout settings (seconds)
IMPORT_SYNC_TIMEOUT = 300  # 5 minutes for import synchronization
TOKENIZATION_TIMEOUT = 7200  # 2 hours for tokenization
DATASET_PROCESSING_TIMEOUT = 3600  # 1 hour for dataset processing
CHECK_INTERVAL = 5  # seconds between checks

# Dataset size thresholds
DATASET_SIZE_LARGE = 5_000_000  # 5M+ samples
DATASET_SIZE_MEDIUM = 1_000_000  # 1M-5M samples

# Tokenization settings
MIN_CHUNK_LENGTH = 128  # Minimum tokens per chunk
WARMUP_STEPS_FIRST_STAGE = 2000  # For first training stage
WARMUP_STEPS_RESUME = 100  # For resumed training

# Default batch sizes (Rust Fast Tokenizer 단일 프로세스 최적화)
BATCH_SIZE_LARGE_DATASET = 5000  # 대규모: Rust 성능 최대 활용
BATCH_SIZE_DEFAULT = 10000  # 기본: 단일 프로세스로 큰 배치
WRITER_BATCH_SIZE = 50000  # 디스크 쓰기 배치

# Default process counts (멀티프로세싱 병렬 처리)
DEFAULT_NUM_PROC = 4  # 4개 프로세스 병렬 처리 (속도와 메모리 균형)
FILTER_NUM_PROC_DIVISOR = 2  # 필터링 프로세스
MAX_FILTER_NUM_PROC = 2  # 최대 필터링 프로세스

# Performance settings
ESTIMATED_TOKENIZATION_SPEED = 10000  # samples/sec (멀티프로세싱)
WARMUP_TEXT_PATTERN = "Hello world " * 100
WARMUP_TEXT_COUNT = 10

# Environment variable keys
ENV_RANK = "RANK"
ENV_WORLD_SIZE = "WORLD_SIZE"
ENV_LOCAL_RANK = "LOCAL_RANK"
ENV_HF_HOME = "HF_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
ENV_DATASET_NUM_PROC = "DATASET_NUM_PROC"
ENV_DATASET_BATCH_SIZE = "DATASET_BATCH_SIZE"
ENV_DATASET_WRITER_BATCH_SIZE = "DATASET_WRITER_BATCH_SIZE"
ENV_TOKENIZERS_PARALLELISM = "TOKENIZERS_PARALLELISM"


# ============================================================================
# Early initialization
# ============================================================================

# ⚠️ 중요: 모든 import 전에 TOKENIZERS_PARALLELISM 설정!
# tokenizers 라이브러리가 import 시점에 이 값을 캐싱하므로 가장 먼저 설정해야 함
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
import time as time_module
from pathlib import Path as PathType

# Check rank early
rank = int(os.environ.get(ENV_RANK, 0))
world_size = int(os.environ.get(ENV_WORLD_SIZE, 1))
is_main = (rank == 0)

# 동기화 마커 파일
import_marker = PathType("/tmp/.moai_import_done")


def _import_all_modules():
    """공통 import 로직 (중복 제거)"""
    import argparse
    import hashlib
    import time
    import gc
    import logging
    from pathlib import Path
    from typing import Optional, Dict, Any

    try:
        import orjson as json
    except ImportError:
        import json

    try:
        import psutil
    except ImportError:
        psutil = None

    import torch
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset, disable_caching
    import datasets
    datasets.config.IN_MEMORY_MAX_SIZE = 0
    from moai_llm.config import MoaiConfig
    from moai_llm.modeling.model import MoaiForCausalLM

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Return all imports as a dict for global namespace injection
    return {
        'argparse': argparse, 'hashlib': hashlib, 'time': time, 'gc': gc,
        'logging': logging, 'Path': Path, 'Optional': Optional, 'Dict': Dict, 'Any': Any,
        'json': json, 'psutil': psutil, 'torch': torch,
        'AutoTokenizer': AutoTokenizer, 'Trainer': Trainer,
        'TrainingArguments': TrainingArguments,
        'DataCollatorForLanguageModeling': DataCollatorForLanguageModeling,
        'load_dataset': load_dataset, 'disable_caching': disable_caching,
        'datasets': datasets, 'MoaiConfig': MoaiConfig,
        'MoaiForCausalLM': MoaiForCausalLM, 'logger': logger,
    }


# Import synchronization
if is_main:
    # Rank 0: 먼저 import
    print(f"[IMPORT] Rank 0: Importing modules (world_size={world_size})...", flush=True)

    # 이전 마커 제거
    if import_marker.exists():
        import_marker.unlink()

    _modules = _import_all_modules()
    globals().update(_modules)

    # 마커 생성 (다른 rank들이 import 시작 가능)
    import_marker.touch()
    print(f"[IMPORT] Rank 0: ✅ All modules imported!", flush=True)
else:
    # 다른 rank들: 마커 대기
    print(f"[IMPORT] Rank {rank}: Waiting for rank 0...", flush=True)

    waited = 0
    while not import_marker.exists() and waited < IMPORT_SYNC_TIMEOUT:
        time_module.sleep(0.5)
        waited += 0.5

    if not import_marker.exists():
        print(f"[IMPORT] Rank {rank}: Timeout waiting for rank 0!", flush=True)
        sys.exit(1)

    # 이제 안전하게 import
    _modules = _import_all_modules()
    globals().update(_modules)

    print(f"[IMPORT] Rank {rank}: ✅ Modules imported!", flush=True)


# ============================================================================
# Utility Functions
# ============================================================================

def get_ddp_info() -> Dict[str, Any]:
    """
    DDP 환경 정보를 반환합니다.

    Returns:
        dict: rank, world_size, is_distributed, is_main_process 포함
    """
    rank = int(os.environ.get(ENV_RANK, -1))
    world_size = int(os.environ.get(ENV_WORLD_SIZE, -1))

    is_distributed = rank >= 0 and world_size > 1

    # torch.distributed로 다시 확인
    if not is_distributed and torch.distributed.is_available():
        try:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                is_distributed = True
        except (RuntimeError, ValueError, AttributeError):
            pass

    return {
        'rank': rank if rank >= 0 else 0,
        'world_size': world_size if world_size > 0 else 1,
        'is_distributed': is_distributed,
        'is_main_process': rank == 0 if rank >= 0 else True,
    }


def ddp_barrier():
    """안전한 DDP barrier 호출"""
    try:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except (RuntimeError, ValueError, AttributeError):
        pass


def get_cache_home() -> str:
    """캐시 홈 디렉토리 반환"""
    return os.environ.get(
        ENV_HF_HOME,
        os.environ.get(ENV_XDG_CACHE_HOME, os.path.expanduser("~/.cache/huggingface"))
    )


def create_cache_path(name: str, suffix: str = "") -> Path:
    """
    캐시 경로 생성

    Args:
        name: 데이터셋 이름 또는 식별자
        suffix: 경로 접미사 (예: "_tokenized", "_final")

    Returns:
        Path: 캐시 경로
    """
    cache_hash = hashlib.md5(name.encode()).hexdigest()[:16]
    cache_home = get_cache_home()
    return Path(cache_home) / "datasets" / f"{cache_hash}{suffix}"


def wait_for_marker(marker_path: Path, timeout: int = TOKENIZATION_TIMEOUT,
                   check_interval: int = CHECK_INTERVAL, rank: int = 0) -> bool:
    """
    마커 파일이 생성될 때까지 대기

    Args:
        marker_path: 마커 파일 경로
        timeout: 최대 대기 시간 (초)
        check_interval: 체크 간격 (초)
        rank: 현재 rank (로깅용)

    Returns:
        bool: 성공 여부
    """
    import time
    waited = 0
    while not marker_path.exists() and waited < timeout:
        time.sleep(check_interval)
        waited += check_interval
        if waited % 60 == 0:  # 1분마다 로그
            logger.info(f"[Rank {rank}] Still waiting... ({waited}s elapsed)")

    return marker_path.exists()


def log_with_rank(msg: str, rank: int = None, is_main: bool = None):
    """
    Rank 정보와 함께 로깅

    Args:
        msg: 로그 메시지
        rank: Rank 번호 (None이면 자동 감지)
        is_main: Main process 여부 (None이면 자동 감지)
    """
    if rank is None or is_main is None:
        ddp_info = get_ddp_info()
        rank = ddp_info['rank']
        is_main = ddp_info['is_main_process']

    if is_main:
        logger.info(msg)
    else:
        logger.info(f"[Rank {rank}] {msg}")


def calculate_optimal_num_proc(total_samples: int, cpu_count: int, available_memory: int = None) -> int:
    """
    최적 프로세스 수 계산 (항상 1 반환 - 메모리 효율)

    num_proc > 1을 사용하면:
    1. 각 프로세스가 전체 데이터셋을 메모리에 로드 → 메모리 폭발
    2. datasets 라이브러리가 TOKENIZERS_PARALLELISM=false 강제 설정
    3. Rust 내부 병렬 처리 비활성화

    num_proc = 1을 사용하면:
    1. 메모리 효율: 데이터셋 1번만 로드
    2. Rust Fast Tokenizer의 내부 멀티스레딩 활용 가능
    3. 전체적으로 더 안정적이고 빠름

    Args:
        total_samples: 총 샘플 수 (사용 안됨)
        cpu_count: CPU 코어 수 (사용 안됨)
        available_memory: 사용 가능한 메모리 (사용 안됨)

    Returns:
        항상 1 (메모리 효율 + Rust 병렬 처리)
    """
    return 1


def get_tokenization_env_config() -> Dict[str, int]:
    """토크나이제이션 관련 환경 변수 설정 반환"""
    cpu_count = os.cpu_count() or 8
    return {
        'num_proc': int(os.getenv(ENV_DATASET_NUM_PROC, min(DEFAULT_NUM_PROC, cpu_count))),
        'batch_size': int(os.getenv(ENV_DATASET_BATCH_SIZE, BATCH_SIZE_DEFAULT)),
        'writer_batch_size': int(os.getenv(ENV_DATASET_WRITER_BATCH_SIZE, WRITER_BATCH_SIZE)),
    }


def get_optimal_num_shards(dataset_size: int, cpu_count: int) -> int:
    """
    데이터셋 크기와 CPU 코어 수에 따른 최적 shard 수 계산

    Args:
        dataset_size: 데이터셋 샘플 수
        cpu_count: CPU 코어 수

    Returns:
        최적 shard 수 (8-64 사이)
    """
    # CPU 코어의 절반을 기준으로, 최소 8, 최대 64
    base_shards = max(8, cpu_count // 2)

    # 대용량 데이터셋은 더 많은 shard 사용
    if dataset_size > DATASET_SIZE_LARGE:
        return min(64, base_shards * 2)
    elif dataset_size > DATASET_SIZE_MEDIUM:
        return min(48, base_shards)
    else:
        return min(32, base_shards)


def get_optimal_prefetch_factor(gpu_memory_gb: float = None, batch_size: int = 4) -> int:
    """
    GPU 메모리에 따른 최적 prefetch factor 계산

    Args:
        gpu_memory_gb: GPU 메모리 크기 (GB), None이면 자동 감지
        batch_size: 배치 크기

    Returns:
        최적 prefetch factor (2-8 사이)
    """
    if gpu_memory_gb is None:
        try:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                gpu_memory_gb = 16  # CPU 모드 기본값
        except:
            gpu_memory_gb = 16

    # GPU 메모리 크기에 따른 최적 prefetch
    # 40GB+ GPU (A100): 8, 24GB (RTX 3090/4090): 6, 16GB: 4, 그 외: 2
    if gpu_memory_gb >= 40:
        return 8
    elif gpu_memory_gb >= 24:
        return 6
    elif gpu_memory_gb >= 16:
        return 4
    else:
        return 2


def load_files_parallel(file_paths: list, max_workers: int = 8) -> list:
    """
    여러 파일을 병렬로 로드

    Args:
        file_paths: 로드할 파일 경로 리스트
        max_workers: 최대 worker 수

    Returns:
        로드된 Dataset 리스트
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import Dataset

    if not file_paths:
        return []

    # 단일 파일은 병렬 처리 불필요
    if len(file_paths) == 1:
        logger.info(f"  Loading file: {file_paths[0]}")
        file_data = _load_single_file(file_paths[0])
        logger.info(f"    → {len(file_data):,} samples")
        return [Dataset.from_list(file_data)]

    # 병렬 로딩
    logger.info(f"🚀 Loading {len(file_paths)} files in parallel (workers={max_workers})...")
    datasets_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 파일 로딩 작업 제출
        future_to_file = {executor.submit(_load_single_file, f): f for f in file_paths}

        # 완료된 작업 처리
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_data = future.result()
                logger.info(f"  ✓ Loaded {file_path}: {len(file_data):,} samples")
                datasets_list.append(Dataset.from_list(file_data))
            except Exception as e:
                logger.error(f"  ✗ Failed to load {file_path}: {e}")

    logger.info(f"✅ Loaded {len(datasets_list)}/{len(file_paths)} files successfully")
    return datasets_list


def get_cache_version_key(tokenizer, additional_info: str = "") -> str:
    """
    토크나이저 버전을 포함한 캐시 버전 키 생성

    Args:
        tokenizer: 토크나이저 객체
        additional_info: 추가 정보 (예: 설정값)

    Returns:
        캐시 버전 키 (8자리 해시)
    """
    # 토크나이저 버전 정보 수집
    version_info = []

    # 1. 토크나이저 vocab 크기
    version_info.append(f"vocab_{tokenizer.vocab_size}")

    # 2. 토크나이저 타입
    tokenizer_type = type(tokenizer).__name__
    version_info.append(f"type_{tokenizer_type}")

    # 3. 특수 토큰
    special_tokens = {
        'bos': tokenizer.bos_token_id,
        'eos': tokenizer.eos_token_id,
        'pad': tokenizer.pad_token_id,
        'unk': tokenizer.unk_token_id,
    }
    version_info.append(f"tokens_{special_tokens}")

    # 4. 추가 정보
    if additional_info:
        version_info.append(additional_info)

    # 해시 생성
    version_string = "_".join(str(v) for v in version_info)
    cache_version = hashlib.md5(version_string.encode()).hexdigest()[:8]

    return cache_version


# ============================================================================
# Sequence Concatenation for Pretraining
# ============================================================================

def concatenate_sequences(
    tokenized_sequences: list,
    max_seq_length: int,
    eos_token_id: int,
) -> list:
    """
    여러 시퀀스를 연결하여 max_seq_length 청크로 분할합니다.
    각 원본 시퀀스 끝에 EOS 토큰을 삽입하여 문서 경계를 표시합니다.

    Args:
        tokenized_sequences: 토큰화된 시퀀스 리스트 (각각 input_ids 포함)
        max_seq_length: 최대 시퀀스 길이
        eos_token_id: EOS 토큰 ID

    Returns:
        연결 후 max_seq_length로 분할된 시퀀스 리스트
    """
    import numpy as np

    # 1. 총 길이 추정 (over-estimate to avoid reallocation)
    estimated_len = sum(len(seq["input_ids"]) + 1 for seq in tokenized_sequences)
    logger.info(f"📦 Concatenating {len(tokenized_sequences):,} sequences (~{estimated_len:,} tokens)")

    # 2. 한 번의 루프로 numpy 배열 구축 (최적화)
    all_tokens = np.empty(estimated_len, dtype=np.int32)
    offset = 0

    for seq in tokenized_sequences:
        input_ids = seq["input_ids"]
        seq_len = len(input_ids)

        if seq_len == 0:
            continue

        # 시퀀스 복사 및 EOS 추가를 한 번에
        all_tokens[offset:offset + seq_len] = input_ids
        offset += seq_len

        # EOS 추가 (필요한 경우만)
        if input_ids[-1] != eos_token_id:
            all_tokens[offset] = eos_token_id
            offset += 1

    # 실제 사용된 길이로 자르기
    all_tokens = all_tokens[:offset]

    # 3. max_seq_length 청크로 분할
    num_chunks = (len(all_tokens) + max_seq_length - 1) // max_seq_length
    chunks = []

    for i in range(num_chunks):
        start = i * max_seq_length
        end = min(start + max_seq_length, len(all_tokens))
        chunk_len = end - start

        # 마지막 청크가 너무 짧으면 버림
        if chunk_len < MIN_CHUNK_LENGTH:
            logger.info(f"  Dropping short final chunk of {chunk_len} tokens")
            continue

        # numpy 배열을 직접 사용 (메모리 복사 최소화)
        chunks.append({
            "input_ids": all_tokens[start:end].copy(),  # numpy array
            "attention_mask": np.ones(chunk_len, dtype=np.int8),  # int8로 메모리 절약
        })

    logger.info(f"✓ Created {len(chunks):,} chunks of max {max_seq_length} tokens each")
    return chunks


# ============================================================================
# Optimized Tokenization Function
# ============================================================================

def tokenize_dataset(
    dataset,
    tokenizer,
    text_column: str = "text",
    max_seq_length: int = 2048,
    packing: bool = False,
    num_proc: int = None,
):
    """
    datasets.map() 기반 토크나이징 (안정적)

    Args:
        dataset: HuggingFace Dataset 객체
        tokenizer: 토크나이저
        text_column: 텍스트 컬럼 이름
        max_seq_length: 최대 시퀀스 길이
        packing: True면 truncation 없이 토큰화
        num_proc: 프로세스 수 (None이면 1)

    Returns:
        토큰화된 Dataset 객체
    """
    total_samples = len(dataset)

    # num_proc 설정
    if num_proc is None:
        num_proc = 4  # 4개 프로세스로 병렬 처리

    batch_size = 10000

    logger.info(f"🔤 Tokenization config:")
    logger.info(f"   Samples: {total_samples:,}")
    logger.info(f"   Processes: {num_proc}")
    logger.info(f"   Batch size: {batch_size:,}")
    logger.info(f"   Mode: {'packing' if packing else 'truncation'}")

    start_time = time.time()

    # 토크나이징 함수
    def batch_tokenize(examples):
        if packing:
            return tokenizer(
                examples[text_column],
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
        else:
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )

    # datasets.map() 사용
    tokenized = dataset.map(
        batch_tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing",
    )

    elapsed = time.time() - start_time
    speed = total_samples / elapsed if elapsed > 0 else 0
    logger.info(f"✅ Tokenization completed:")
    logger.info(f"   Time: {elapsed/60:.1f} min")
    logger.info(f"   Speed: {speed:,.0f} samples/sec")
    logger.info(f"   Output: {len(tokenized):,} samples")

    return tokenized


# ============================================================================
# 데이터셋 로드 및 변환
# ============================================================================

def _load_single_file(file_path: str) -> list:
    """단일 파일을 로드하여 텍스트 리스트로 반환"""
    formatted_data = []
    
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        with open(file_path, 'rb') as f:  # Binary mode for orjson
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.loads(f.read())  # orjson uses loads() not load()
        
        for item in data:
            text = _convert_to_text(item)
            if text:
                formatted_data.append({"text": text})
    else:
        # txt 파일
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    formatted_data.append({"text": line})
    
    return formatted_data


def _load_hf_dataset(dataset_name: str, dataset_config: Optional[str] = None):
    """
    단일 HuggingFace 데이터셋을 로드하여 텍스트 리스트로 반환

    데이터셋 이름에 config를 포함할 수 있음:
        - "dataset_name:config_name" 형식 지원
        - 예: "maywell/korean_textbooks:claude_evol"

    DDP 환경에서는 rank 0만 데이터셋을 다운로드하고, 다른 프로세스는 대기합니다.
    """
    # DDP 환경 정보 가져오기
    ddp_info = get_ddp_info()
    is_distributed = ddp_info['is_distributed']
    is_main_process = ddp_info['is_main_process']
    current_rank = ddp_info['rank']
    
    # dataset_name:config_name 형식 파싱
    if ":" in dataset_name:
        dataset_name, config_from_name = dataset_name.split(":", 1)
        if not dataset_config:
            dataset_config = config_from_name
    
    logger.info(f"  Loading HuggingFace: {dataset_name}")
    
    # 메모리 최적화 옵션 - 메모리 맵 파일 사용 (메모리 절약)
    load_kwargs = {
        "keep_in_memory": False,  # 디스크에 메모리 맵 파일로 유지
    }
    if dataset_config:
        load_kwargs["name"] = dataset_config
    
    # DDP 환경에서는 rank 0만 데이터셋을 다운로드 및 변환
    # 다른 rank들은 최종 변환 결과만 로드
    if is_distributed:
        # 캐시 경로 생성
        cache_key = f"{dataset_name}_{dataset_config}" if dataset_config else dataset_name
        dataset_save_path = create_cache_path(cache_key, "_final")
        filter_marker_path = Path(str(dataset_save_path).replace("_final", ".filtered.marker"))

        # 이미 처리된 데이터셋이 있으면 모든 rank가 로드 (재시작 시 안전)
        if dataset_save_path.exists() and filter_marker_path.exists():
            logger.info(f"    [Rank {current_rank}] ✅ Using existing processed dataset from: {dataset_save_path}")
            from datasets import Dataset
            load_start = time.time()
            converted = Dataset.load_from_disk(str(dataset_save_path))
            load_time = time.time() - load_start
            logger.info(f"    [Rank {current_rank}] Loaded {len(converted):,} samples in {load_time:.1f}s")

            # barrier 동기화
            ddp_barrier()

            # 변환 결과 반환 (나머지 로직 건너뛰기)
            return converted

        # barrier 동기화
        ddp_barrier()
        
        if is_main_process:
            # rank 0만 데이터셋 다운로드 및 변환
            logger.info(f"    [Rank 0] Downloading dataset...")
            if dataset_config:
                logger.info(f"    Config: {dataset_config}")
            raw_dataset = load_dataset(dataset_name, **load_kwargs)
            logger.info(f"    [Rank 0] Dataset download completed")
            
            # train split 사용
            train_data = raw_dataset.get("train", raw_dataset)
        else:
            # 다른 프로세스는 나중에 최종 결과만 로드 (여기서는 아무것도 안 함)
            logger.info(f"    [Rank {current_rank}] Waiting for rank 0 to complete processing...")
            train_data = None  # 나중에 캐시에서 로드
    else:
        # 단일 프로세스 환경
        if dataset_config:
            logger.info(f"    Config: {dataset_config}")
        raw_dataset = load_dataset(dataset_name, **load_kwargs)
        
        # train split 사용
        train_data = raw_dataset.get("train", raw_dataset)
    
    # dataset.map()으로 빠르게 변환
    def convert_batch(examples):
        texts = []
        # 각 컬럼을 개별 딕셔너리로 재구성
        keys = list(examples.keys())
        num_examples = len(examples[keys[0]]) if keys else 0
        
        for i in range(num_examples):
            item = {k: examples[k][i] for k in keys}
            text = _convert_to_text(item)
            texts.append(text if text else "")
        
        return {"text": texts}
    
    # 배치 처리로 변환 (병렬 처리 유지, 메모리 효율적)
    env_config = get_tokenization_env_config()
    dataset_num_proc = env_config['num_proc']
    dataset_batch_size = env_config['batch_size']
    dataset_writer_batch_size = env_config['writer_batch_size']

    # DDP 환경에서는 rank 0만 변환하고 다른 프로세스는 캐시만 로드
    if is_distributed:
        # 캐시 완료 마커 파일 경로 생성
        cache_key = f"{dataset_name}_{dataset_config}" if dataset_config else dataset_name
        cache_marker = Path(str(create_cache_path(cache_key, "")).replace(cache_key[:16], f".{cache_key[:16]}_converted.marker"))
        
        if is_main_process:
            # rank 0만 데이터셋 변환 (멀티프로세스로 빠르게)
            logger.info(f"    [Rank 0] Converting dataset with {dataset_num_proc} processes "
                       f"(batch_size={dataset_batch_size}, writer_batch_size={dataset_writer_batch_size})...")
            converted = train_data.map(
                convert_batch,
                batched=True,
                batch_size=dataset_batch_size,
                num_proc=dataset_num_proc,
                remove_columns=train_data.column_names,
                load_from_cache_file=True,
                writer_batch_size=dataset_writer_batch_size,
                keep_in_memory=False,  # 메모리 맵 파일 사용
                desc=f"Converting {dataset_name}",
            )

            # dataset.map() 후 TOKENIZERS_PARALLELISM 복원
            os.environ[ENV_TOKENIZERS_PARALLELISM] = "true"

            # 변환 완료 마커 생성 (filter 전에!)
            cache_marker.parent.mkdir(parents=True, exist_ok=True)
            cache_marker.touch()
            logger.info(f"    [Rank 0] Created conversion marker: {cache_marker}")

            # 빈 텍스트 필터링 (병렬 처리로 빠르게)
            filter_num_proc = min(dataset_num_proc // FILTER_NUM_PROC_DIVISOR, MAX_FILTER_NUM_PROC)
            logger.info(f"    [Rank 0] Filtering empty texts with {filter_num_proc} processes...")
            converted = converted.filter(
                lambda x: len(x["text"]) > 0,
                num_proc=filter_num_proc,
                writer_batch_size=dataset_writer_batch_size,
                keep_in_memory=False,
                load_from_cache_file=True,
            )

            # filter() 후 TOKENIZERS_PARALLELISM 복원
            os.environ[ENV_TOKENIZERS_PARALLELISM] = "true"

            logger.info(f"    [Rank 0] Conversion completed: {len(converted):,} samples")

            # 최종 결과를 디스크에 저장 (다른 rank들이 안전하게 로드할 수 있도록)
            if not dataset_save_path.exists():
                logger.info(f"    [Rank 0] Saving final dataset to: {dataset_save_path}")
                save_start = time.time()
                num_shards = get_optimal_num_shards(len(converted), os.cpu_count() or 8)
                converted.save_to_disk(
                    str(dataset_save_path),
                    num_shards=num_shards,
                )
                save_time = time.time() - save_start
                logger.info(f"    [Rank 0] Dataset saved in {save_time:.1f}s (shards={num_shards})")
            else:
                logger.info(f"    [Rank 0] Dataset already saved at: {dataset_save_path}")

            # 필터 완료 마커 생성
            filter_marker_path.touch()
            logger.info(f"    [Rank 0] Created filter marker: {filter_marker_path}")

            # 변환 완료 후 barrier
            ddp_barrier()
                
        else:
            # 다른 프로세스는 필터 마커 대기 후 최종 결과만 로드!
            logger.info(f"    [Rank {current_rank}] Waiting for rank 0 to complete all processing...")

            if not wait_for_marker(filter_marker_path, DATASET_PROCESSING_TIMEOUT, CHECK_INTERVAL, current_rank):
                raise TimeoutError(f"Rank {current_rank}: Dataset processing timeout after {DATASET_PROCESSING_TIMEOUT}s")

            logger.info(f"    [Rank {current_rank}] Processing complete, loading final result from cache...")

            # barrier 동기화
            ddp_barrier()

            # rank 0이 저장한 최종 데이터셋을 직접 로드
            logger.info(f"    [Rank {current_rank}] Loading final dataset from: {dataset_save_path}")

            # 파일이 완전히 준비될 때까지 짧은 대기 (파일 시스템 동기화)
            for attempt in range(60):  # 최대 30초 대기
                if dataset_save_path.exists() and (dataset_save_path / "dataset_info.json").exists():
                    break
                time.sleep(0.5)
            else:
                logger.warning(f"    [Rank {current_rank}] Dataset files not fully ready, proceeding anyway...")

            from datasets import Dataset
            load_start = time.time()
            converted = Dataset.load_from_disk(str(dataset_save_path))
            load_time = time.time() - load_start

            logger.info(f"    [Rank {current_rank}] Loaded from disk in {load_time:.1f}s: {len(converted):,} samples")
            
    else:
        # 단일 프로세스 환경
        logger.info(f"    Converting dataset with {dataset_num_proc} processes...")
        converted = train_data.map(
            convert_batch,
            batched=True,
            batch_size=dataset_batch_size,
            num_proc=dataset_num_proc,
            remove_columns=train_data.column_names,
            load_from_cache_file=True,
            writer_batch_size=dataset_writer_batch_size,
            keep_in_memory=False,
            desc=f"Converting {dataset_name}",
        )

        # dataset.map() 후 TOKENIZERS_PARALLELISM 복원
        os.environ[ENV_TOKENIZERS_PARALLELISM] = "true"

        logger.info(f"    Filtering empty texts...")
        filter_num_proc = min(dataset_num_proc // FILTER_NUM_PROC_DIVISOR, MAX_FILTER_NUM_PROC)
        converted = converted.filter(
            lambda x: len(x["text"]) > 0,
            num_proc=filter_num_proc,
            load_from_cache_file=True,
            writer_batch_size=dataset_writer_batch_size,
            keep_in_memory=False,
        )

        # filter() 후 TOKENIZERS_PARALLELISM 복원
        os.environ[ENV_TOKENIZERS_PARALLELISM] = "true"
    
    # Dataset 객체를 그대로 반환 (메모리 효율적)
    # 리스트 변환을 피하고 Dataset을 직접 사용하여 메모리 사용량 최소화
    logger.info(f"    → {len(converted):,} samples")
    return converted  # Dataset 객체 반환


def load_pretrain_dataset(
    dataset_names: Optional[list] = None,
    dataset_config: Optional[str] = None,
    train_files: Optional[list] = None,
    text_column: str = "text",
):
    """
    사전학습용 데이터셋 로드 (여러 파일/데이터셋 지원)

    Args:
        dataset_names: HuggingFace 데이터셋 이름 리스트 (예: ["wikipedia", "alpaca"])
        dataset_config: 데이터셋 설정 (첫 번째 데이터셋에만 적용)
        train_files: 로컬 파일 경로 리스트 (txt 또는 json)
        text_column: 텍스트 컬럼 이름
    
    지원 포맷:
        - txt 파일: 각 줄이 하나의 문서
        - json 파일: instruction/output, input/output, messages, conversations 등
        - HuggingFace 데이터셋: 위 형식 자동 감지
    """
    logger.info("📚 Loading pretrain dataset...")
    
    from datasets import Dataset, concatenate_datasets
    
    datasets_list = []

    # 로컬 파일 로드 (병렬 처리)
    if train_files:
        if isinstance(train_files, str):
            train_files = [train_files]

        # 파일 병렬 로딩 사용
        loaded_datasets = load_files_parallel(train_files, max_workers=8)
        datasets_list.extend(loaded_datasets)
    
    # HuggingFace 데이터셋 로드 (Dataset 객체를 그대로 사용)
    if dataset_names:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        for i, ds_name in enumerate(dataset_names):
            # 첫 번째 데이터셋에만 config 적용
            config = dataset_config if i == 0 else None
            ds_data = _load_hf_dataset(ds_name, config)
            # Dataset 객체를 그대로 추가 (리스트 변환 없음)
            if isinstance(ds_data, Dataset):
                datasets_list.append(ds_data)
            else:
                # 호환성을 위해 리스트인 경우만 변환
                datasets_list.append(Dataset.from_list(ds_data))
    
    if not datasets_list:
        raise ValueError("Either dataset_names or train_files must be provided")
    
    # 여러 Dataset을 메모리 효율적으로 결합
    if len(datasets_list) == 1:
        combined_dataset = datasets_list[0]
    else:
        logger.info(f"  Concatenating {len(datasets_list)} datasets...")
        combined_dataset = concatenate_datasets(datasets_list)
    
    logger.info(f"  Total: {len(combined_dataset):,} samples")
    
    dataset = {"train": combined_dataset}
    text_column = "text"

    logger.info(f"✓ Dataset loaded: {len(dataset['train'])} samples")
    return dataset, text_column


def _convert_to_text(item: dict) -> Optional[str]:
    """
    다양한 데이터 형식을 순수 텍스트로 변환 (Foundation Model pretrain용)
    
    특수 토큰 없이 모든 컬럼을 하나의 연속된 텍스트로 합칩니다.
    이것은 Next Token Prediction을 위한 Foundation Model 학습 방식입니다.
    
    지원 형식:
        - {"text": "..."}: 그대로 사용
        - {"input": "...", "output": "..."}: 순수 텍스트로 합침
        - {"instruction": "...", "output": "..."}: 순수 텍스트로 합침
        - {"messages": [...]}: 모든 메시지 순수 텍스트로 합침
        - {"conversations": [...]}: 모든 대화 순수 텍스트로 합침
    """
    # 안전한 문자열 추출 함수
    def safe_str(val) -> str:
        return (val or "").strip()
    
    # text 필드가 있으면 그대로 사용
    if "text" in item and item["text"]:
        return item["text"]
    
    # input/output 포맷 → 순수 텍스트
    if "input" in item and "output" in item:
        inp = safe_str(item["input"])
        out = safe_str(item["output"])
        if not out:
            return None
        return f"{inp}\n\n{out}" if inp else out
    
    # instruction/output 포맷 (Alpaca) → 순수 텍스트
    if "instruction" in item and "output" in item:
        inst = safe_str(item["instruction"])
        out = safe_str(item["output"])
        # input 필드도 있으면 합침
        inp = safe_str(item.get("input"))
        if inp:
            inst = f"{inst}\n{inp}" if inst else inp
        if not inst and not out:
            return None
        return f"{inst}\n\n{out}" if inst else out
    
    # messages 포맷 (OpenAI Chat) → 순수 텍스트
    if "messages" in item and item["messages"]:
        texts = []
        for msg in item["messages"]:
            if msg:
                content = safe_str(msg.get("content"))
                if content:
                    texts.append(content)
        return "\n\n".join(texts) if texts else None
    
    # conversations 포맷 (ShareGPT) → 순수 텍스트
    if "conversations" in item and item["conversations"]:
        texts = []
        for conv in item["conversations"]:
            if conv:
                value = safe_str(conv.get("value"))
                if value:
                    texts.append(value)
        return "\n\n".join(texts) if texts else None
    
    # DeepSeek R1 스타일 (input/content/reasoning_content) → 순수 텍스트
    if "input" in item and "content" in item:
        parts = []
        inp = safe_str(item.get("input"))
        reasoning = safe_str(item.get("reasoning_content"))
        content = safe_str(item.get("content"))
        if inp:
            parts.append(inp)
        if reasoning:
            parts.append(reasoning)
        if content:
            parts.append(content)
        return "\n\n".join(parts) if parts else None
    
    # prompt/response 포맷 → 순수 텍스트
    if "prompt" in item and "response" in item:
        prompt = safe_str(item["prompt"])
        response = safe_str(item["response"])
        if not response:
            return None
        return f"{prompt}\n\n{response}" if prompt else response
    
    # question/answer 포맷 → 순수 텍스트
    if "question" in item and "answer" in item:
        question = safe_str(item["question"])
        answer = safe_str(item["answer"])
        if not answer:
            return None
        return f"{question}\n\n{answer}" if question else answer
    
    # prompt/completion 포맷 → 순수 텍스트
    if "prompt" in item and "completion" in item:
        prompt = safe_str(item["prompt"])
        completion = safe_str(item["completion"])
        if not completion:
            return None
        return f"{prompt}\n\n{completion}" if prompt else completion
    
    # 알 수 없는 형식
    logger.warning(f"Unknown format, skipping: {list(item.keys())}")
    return None


def load_sft_dataset(
    dataset_names: Optional[list] = None,
    train_files: Optional[list] = None,
):
    """
    SFT용 데이터셋 로드 및 포맷 변환 (여러 파일/데이터셋 지원)

    지원 포맷:
    - Alpaca: {"instruction": "...", "output": "..."}
    - Chat: {"messages": [{"role": "user", "content": "..."}]}
    - ShareGPT: {"conversations": [{"from": "human", "value": "..."}]}
    """
    logger.info("📚 Loading SFT dataset...")
    
    all_data = []

    # 로컬 파일 로드
    if train_files:
        if isinstance(train_files, str):
            train_files = [train_files]
        
        for file_path in train_files:
            logger.info(f"  Loading file: {file_path}")
            file_data = _load_single_file(file_path)
            logger.info(f"    → {len(file_data):,} samples")
            all_data.extend(file_data)
    
    # HuggingFace 데이터셋 로드
    if dataset_names:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        for ds_name in dataset_names:
            ds_data = _load_hf_dataset(ds_name)
            all_data.extend(ds_data)
    
    if not all_data:
        raise ValueError("Either dataset_names or train_files must be provided")
    
    logger.info(f"  Total: {len(all_data):,} samples")

    # Dataset으로 변환
    from datasets import Dataset
    dataset = {"train": Dataset.from_list(all_data)}

    logger.info(f"✓ SFT dataset loaded: {len(dataset['train'])} samples")
    return dataset, "text"


# ============================================================================
# 모델 및 토크나이저 초기화
# ============================================================================

def setup_model_and_tokenizer(
    tokenizer_path: str,
    model_config: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    use_flash_attention: bool = False,
    use_compile: bool = False,
    use_bf16: bool = False,
    use_fp16: bool = False,
):
    """모델과 토크나이저 초기화"""

    # dtype 결정
    if use_bf16:
        dtype = torch.bfloat16
        dtype_str = "bfloat16"
    elif use_fp16:
        dtype = torch.float16
        dtype_str = "float16"
    else:
        dtype = torch.float32
        dtype_str = "float32"

    # 토크나이저
    logger.info(f"📝 Loading tokenizer from: {tokenizer_path}")
    import time
    tok_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,  # Rust 기반 고속 토크나이저 강제 사용
    )
    tok_time = time.time() - tok_start
    logger.info(f"✓ Tokenizer loaded in {tok_time:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델
    if pretrained_model:
        logger.info(f"🔄 Loading pretrained model: {pretrained_model}")
        logger.info(f"  (This may take 20-30s for 8 GPUs...)")
        model_start = time.time()
        model = MoaiForCausalLM.from_pretrained(pretrained_model, dtype=dtype)
        model_time = time.time() - model_start
        logger.info(f"✓ Model loaded in {model_time:.1f}s")
        logger.info(f"  Model dtype: {dtype_str}")
    else:
        logger.info("🆕 Creating new model from config")
        if model_config:
            config = MoaiConfig.from_json_file(model_config)
        else:
            config = MoaiConfig()
        
        # Flash Attention 설정
        if use_flash_attention:
            try:
                import flash_attn
                config.use_flash_attention = True
                logger.info("⚡ Flash Attention 2 enabled")
            except ImportError:
                logger.warning("⚠️ flash-attn not installed, using standard attention")
        
        # 새 모델 생성 시 dtype 지정
        config.dtype = dtype
        model = MoaiForCausalLM(config)
        model = model.to(dtype)
        logger.info(f"  Model dtype: {dtype_str}")
    
    # torch.compile 적용 (PyTorch 2.0+)
    # Note: mode="default" is more stable with DDP than "reduce-overhead"
    if use_compile:
        try:
            logger.info("🔧 Compiling model with torch.compile (mode=default)...")
            model = torch.compile(model, mode="default", dynamic=True)
            logger.info("✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}")

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    return model, tokenizer


# ============================================================================
# 학습
# ============================================================================

# ============================================================================
# Data Source Preparation
# ============================================================================
def prepare_data_sources(args) -> list:
    """
    args에서 모든 데이터 소스를 추출하여 리스트로 반환

    Args:
        args: 학습 인자

    Returns:
        [("hf", "dataset_name"), ("file", "path.jsonl"), ...]
    """
    all_sources = []

    if args.dataset:
        for ds in args.dataset:
            all_sources.append(("hf", ds))

    if args.train_file:
        for f in args.train_file:
            all_sources.append(("file", f))

    return all_sources


def calculate_cache_paths(source: tuple, tokenizer, args, idx: int) -> tuple:
    """
    캐시 경로 계산 (모든 rank에서 동일하게 계산)

    Args:
        source: ("hf", "dataset_name") 또는 ("file", "path")
        tokenizer: 토크나이저
        args: 학습 인자
        idx: 소스 인덱스 (캐시 키 생성용)

    Returns:
        (src_name, tokenized_cache_path, tokenized_marker)
    """
    src_type, src_name = source
    cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    cache_version = get_cache_version_key(
        tokenizer,
        additional_info=f"packing_{args.packing}_maxlen_{args.max_seq_length}_seq_{idx}"
    )
    dataset_hash = hashlib.md5(f"{src_name}_{cache_version}".encode()).hexdigest()[:16]
    tokenized_cache_path = Path(cache_home) / "datasets" / f"{dataset_hash}_tokenized"
    tokenized_marker = Path(cache_home) / "datasets" / f".{dataset_hash}_tokenized.marker"

    return src_name, tokenized_cache_path, tokenized_marker


def load_dataset_info_from_cache(
    source: tuple,
    tokenizer,
    args,
    idx: int,
) -> dict:
    """
    캐시에서 데이터셋 정보 로드 (Non-main ranks용)

    Args:
        source: ("hf", "dataset_name") 또는 ("file", "path")
        tokenizer: 토크나이저
        args: 학습 인자
        idx: 소스 인덱스

    Returns:
        {'name': str, 'cache_path': Path, 'num_samples': int}
    """
    import gc
    from datasets import Dataset as HFDataset

    ddp_info = get_ddp_info()
    rank = ddp_info['rank']

    # 캐시 경로 계산
    src_name, tokenized_cache_path, tokenized_marker = calculate_cache_paths(
        source, tokenizer, args, idx
    )

    # 마커 파일 대기
    logger.info(f"[Rank {rank}] Waiting for tokenization to complete...")
    if not wait_for_marker(tokenized_marker, TOKENIZATION_TIMEOUT, CHECK_INTERVAL, rank):
        raise TimeoutError(f"Rank {rank}: Tokenization timeout after {TOKENIZATION_TIMEOUT}s")

    logger.info(f"[Rank {rank}] ✅ Tokenization completed, loading info...")

    # 샘플 수 확인을 위해 dataset 로드
    tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
    num_samples = len(tokenized_dataset)
    del tokenized_dataset
    gc.collect()

    return {
        'name': src_name,
        'cache_path': tokenized_cache_path,
        'num_samples': num_samples,
    }


def tokenize_single_source(
    source: tuple,
    tokenizer,
    args,
    idx: int,
) -> dict:
    """
    단일 데이터 소스 토크나이징 (Rank 0 전용)

    Args:
        source: ("hf", "dataset_name") 또는 ("file", "path")
        tokenizer: 토크나이저
        args: 학습 인자
        idx: 소스 인덱스 (캐시 키 생성용)

    Returns:
        {'name': str, 'cache_path': Path, 'num_samples': int}
    """
    import gc
    from datasets import Dataset as HFDataset

    src_type, src_name = source

    # 캐시 경로 계산
    src_name, tokenized_cache_path, tokenized_marker = calculate_cache_paths(
        source, tokenizer, args, idx
    )

    logger.info(f"📦 [{idx+1}] Dataset: {src_name}")

    # 캐시 확인
    if tokenized_cache_path.exists() and tokenized_marker.exists():
        logger.info(f"  ✅ Loading cached tokenized dataset")
        tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
        num_samples = len(tokenized_dataset)
        logger.info(f"  ✅ Loaded {num_samples:,} samples from cache")
    else:
        # 데이터셋 로드
        logger.info(f"  📚 Loading dataset...")
        if src_type == "hf":
            dataset, text_column = load_pretrain_dataset(
                dataset_names=[src_name],
                dataset_config=args.dataset_config if idx == 0 else None,
                train_files=None,
                text_column=args.text_column,
            )
        else:
            dataset, text_column = load_pretrain_dataset(
                dataset_names=None,
                dataset_config=None,
                train_files=[src_name],
                text_column=args.text_column,
            )

        # 토크나이징
        logger.info(f"  🔤 Tokenizing dataset...")
        tokenized_ds = tokenize_dataset(
            dataset=dataset["train"],
            tokenizer=tokenizer,
            text_column=text_column,
            max_seq_length=args.max_seq_length,
            packing=args.packing,
        )

        # Packing (선택적) - PyArrow 스트리밍 기반 (파일 페이징)
        if args.packing:
            import tempfile
            import shutil

            logger.info(f"  📦 Packing sequences (PyArrow streaming for memory efficiency)...")

            total_samples = len(tokenized_ds)
            logger.info(f"     Total samples: {total_samples:,}")

            # 1. 임시 디렉토리에 Arrow 포맷으로 저장 (디스크 페이징 시작)
            temp_dir = Path(tempfile.mkdtemp(prefix="moai_packing_"))
            try:
                logger.info(f"     Saving to temporary Arrow files...")
                tokenized_ds.save_to_disk(str(temp_dir / "tokenized"))
                del tokenized_ds
                gc.collect()

                # 2. 스트리밍 방식으로 청크 단위 packing
                STREAM_BATCH_SIZE = 500000  # 50만 샘플씩 스트리밍
                all_packed_chunks = []

                # Arrow 파일에서 스트리밍 로드
                from datasets import load_from_disk
                dataset_on_disk = load_from_disk(str(temp_dir / "tokenized"))

                num_batches = (total_samples + STREAM_BATCH_SIZE - 1) // STREAM_BATCH_SIZE
                logger.info(f"     Processing {num_batches} batches of {STREAM_BATCH_SIZE:,} samples each")

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * STREAM_BATCH_SIZE
                    end_idx = min(start_idx + STREAM_BATCH_SIZE, total_samples)

                    logger.info(f"     Batch {batch_idx+1}/{num_batches}: Loading {start_idx:,} - {end_idx:,}")

                    # Arrow에서 배치 로드 (메모리 효율적)
                    batch_data = dataset_on_disk.select(range(start_idx, end_idx))
                    tokenized_list = [{"input_ids": ids} for ids in batch_data["input_ids"]]
                    del batch_data
                    gc.collect()

                    logger.info(f"     Batch {batch_idx+1}/{num_batches}: Packing...")
                    packed_batch = concatenate_sequences(
                        tokenized_sequences=tokenized_list,
                        max_seq_length=args.max_seq_length,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    del tokenized_list
                    gc.collect()

                    all_packed_chunks.extend(packed_batch)
                    del packed_batch
                    gc.collect()

                    logger.info(f"     ✓ Batch {batch_idx+1}/{num_batches} complete ({len(all_packed_chunks):,} chunks so far)")

                del dataset_on_disk
                gc.collect()

                logger.info(f"  ✓ Total packed chunks: {len(all_packed_chunks):,}")
                tokenized_dataset = HFDataset.from_list(all_packed_chunks)
                del all_packed_chunks
                gc.collect()

            finally:
                # 임시 파일 삭제
                logger.info(f"     Cleaning up temporary files...")
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            tokenized_dataset = tokenized_ds

        # 저장
        logger.info(f"  💾 Saving tokenized dataset...")
        num_shards = get_optimal_num_shards(len(tokenized_dataset), os.cpu_count() or 8)
        tokenized_dataset.save_to_disk(str(tokenized_cache_path), num_shards=num_shards)
        tokenized_marker.touch()
        num_samples = len(tokenized_dataset)
        logger.info(f"  ✅ Tokenized: {num_samples:,} samples (shards={num_shards})")

        # 메모리 해제
        del dataset
        del tokenized_dataset
        gc.collect()

    return {
        'name': src_name,
        'cache_path': tokenized_cache_path,
        'num_samples': num_samples,
    }


# ============================================================================
# Train Function (단일 데이터셋 학습)
# ============================================================================
def train_single_dataset(
    args,
    dataset_info: dict,
    tokenizer,
    pretrained_model_path: str = None,
    is_first_stage: bool = True,
    stage_name: str = None,
):
    """
    단일 데이터셋에 대한 학습 수행

    Args:
        args: 학습 인자
        dataset_info: {'name': str, 'cache_path': Path, 'num_samples': int}
        tokenizer: 토크나이저
        pretrained_model_path: 사전학습 모델 경로 (Sequential mode에서 이전 stage 체크포인트)
        is_first_stage: 첫 번째 stage인지 여부 (warmup_steps 결정)
        stage_name: Stage 이름 (로깅용)

    Returns:
        checkpoint_path: 저장된 체크포인트 경로
    """
    from datasets import Dataset as HFDataset
    import gc

    ddp_info = get_ddp_info()
    is_main_process = ddp_info['is_main_process']

    if is_main_process:
        logger.info("="*80)
        if stage_name:
            logger.info(f"🚀 Training: {stage_name}")
        else:
            logger.info(f"🚀 Training: {dataset_info['name']}")
        logger.info(f"   Samples: {dataset_info['num_samples']:,}")
        logger.info("="*80)

    # ============================================================================
    # 1. 모델 로드
    # ============================================================================
    if is_main_process:
        logger.info(f"⏳ Loading model from: {pretrained_model_path or 'scratch'}")

    model, _ = setup_model_and_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_config=args.model_config,
        pretrained_model=pretrained_model_path,
        use_flash_attention=args.flash_attention,
        use_compile=args.compile,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
    )
    if is_main_process:
        logger.info(f"✓ Model loaded")

    # ============================================================================
    # 2. 토크나이징된 데이터셋 로드 (모든 rank)
    # ============================================================================
    if is_main_process:
        logger.info(f"📥 Loading tokenized dataset from cache...")

    tokenized_dataset = HFDataset.load_from_disk(str(dataset_info['cache_path']))

    if is_main_process:
        logger.info(f"✓ Loaded {len(tokenized_dataset):,} samples")

    # ============================================================================
    # 3. Data Collator
    # ============================================================================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ============================================================================
    # 4. Training Arguments
    # ============================================================================
    output_dir = args.output_dir if stage_name is None else f"{args.output_dir}/{stage_name}"
    optimal_prefetch = get_optimal_prefetch_factor(batch_size=args.batch_size)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=WARMUP_STEPS_FIRST_STAGE if is_first_stage else WARMUP_STEPS_RESUME,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        save_safetensors=True,
        # I/O 최적화
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=optimal_prefetch,
        dataloader_persistent_workers=True,
        dataloader_drop_last=True,
        # 옵티마이저 및 정밀도
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        tf32=True,
        # 배치 최적화
        group_by_length=not getattr(args, 'packing', False),
        max_grad_norm=1.0,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )

    # ============================================================================
    # 5. Trainer 생성 및 학습
    # ============================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    if is_main_process:
        logger.info(f"🏃 Starting training...")
    trainer.train()

    # ============================================================================
    # 6. 체크포인트 저장
    # ============================================================================
    checkpoint_path = f"{output_dir}/checkpoint"
    trainer.save_model(checkpoint_path)

    # DDP barrier
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # dtype 유지하며 저장 (rank 0만)
    if is_main_process:
        model_dtype = next(model.parameters()).dtype
        if model_dtype in (torch.bfloat16, torch.float16):
            logger.info(f"💾 Re-saving model in {model_dtype} format...")
            model.save_pretrained(checkpoint_path, torch_dtype=model_dtype, safe_serialization=True)
        logger.info(f"✅ Training completed: {checkpoint_path}")

    # ============================================================================
    # 7. 메모리 해제
    # ============================================================================
    del model
    del tokenized_dataset
    del trainer
    gc.collect()

    try:
        torch.cuda.empty_cache()
    except:
        pass

    # DDP barrier
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return checkpoint_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MOAI-LLM Training")

    # 필수 인자
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "sft"], help="Training mode")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")

    # 데이터셋 (여러 데이터셋 또는 여러 파일 지원)
    parser.add_argument("--dataset", type=str, nargs="*", help="HuggingFace dataset names")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--train_file", type=str, nargs="*", help="Training data files (JSONL, JSON, Parquet, etc.)")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name for dataset")

    # 모델 설정
    parser.add_argument("--model_config", type=str, help="Model config file (JSON)")
    parser.add_argument("--pretrained_model", type=str, help="Pretrained model path")

    # 학습 설정
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for full)")

    # 최적화
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    # Packing (Pretrain/SFT 둘 다 지원)
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing/concatenation. "
             "Concatenates all sequences with EOS tokens and chunks into max_seq_length. "
             "Works for both pretrain and SFT modes."
    )
    
    # Sequential 모드 (메모리 절약)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process datasets sequentially one by one to save memory. "
             "Each dataset is loaded, trained, then freed before the next."
    )
    
    # Tokenize only 모드 (DDP 전에 토큰화만 수행)
    parser.add_argument(
        "--tokenize_only",
        action="store_true",
        help="Only tokenize datasets and exit (no training). "
             "Use this to pre-tokenize before running torchrun."
    )

    # Skip tokenization 모드 (이미 토큰화된 데이터 사용)
    parser.add_argument(
        "--skip_tokenization",
        action="store_true",
        help="Skip tokenization and load pre-tokenized datasets from cache. "
             "Use this when you've already run tokenize_datasets.py."
    )

    # 로깅
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # Resume from checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from. "
             "Use this to continue training with different datasets."
    )

    # 기타
    parser.add_argument("--num_proc", type=int, default=48, help="Number of processes for tokenization (default: 48 for high-performance CPUs)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    # 추가 최적화 옵션
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use Flash Attention 2 for faster training (requires flash-attn package)"
    )
    parser.add_argument(
        "--compile",
        action="store_true", 
        help="Use torch.compile for faster training (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache processed datasets for faster subsequent runs"
    )
    
    # Logging 옵션
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging (default: tensorboard)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="moai-llm",
        help="W&B project name (default: moai-llm)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )

    args = parser.parse_args()

    # 검증
    if not args.dataset and not args.train_file:
        parser.error("Either --dataset or --train_file must be provided")

    # DDP 환경 정보
    ddp_info = get_ddp_info()
    rank = ddp_info['rank']
    world_size = ddp_info['world_size']
    is_main_process = ddp_info['is_main_process']

    if is_main_process:
        logger.info("="*80)
        logger.info(f"🚀 Starting {args.mode.upper()} training")
        logger.info(f"🌐 Environment: {world_size} GPU(s), Rank {rank}")
        logger.info("="*80)

    # ============================================================================
    # STEP 1: 토크나이저 로드
    # ============================================================================
    if is_main_process:
        logger.info("📝 Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
    )

    # Fast Tokenizer 강제 체크
    if not tokenizer.is_fast:
        raise ValueError(
            f"❌ Fast Tokenizer not available! "
            f"Current tokenizer: {type(tokenizer).__name__}\n"
            f"Please ensure you're using a tokenizer that supports Fast mode (Rust-based)."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        logger.info("✅ Using Fast Tokenizer (Rust-based)")
        logger.info(f"   Tokenizer type: {type(tokenizer).__name__}")

        # Warmup: Rust 토크나이저 초기화
        logger.info("   Warming up tokenizer...")
        warmup_start = time.time()
        for _ in range(WARMUP_TEXT_COUNT):
            _ = tokenizer(WARMUP_TEXT_PATTERN, truncation=False, padding=False)
        warmup_time = time.time() - warmup_start
        logger.info(f"   Warmup completed in {warmup_time:.2f}s")

    # ============================================================================
    # STEP 2: 데이터 소스 준비
    # ============================================================================
    all_sources = prepare_data_sources(args)

    if is_main_process:
        logger.info(f"📋 Total datasets: {len(all_sources)}")
        for i, (src_type, src_name) in enumerate(all_sources):
            logger.info(f"  {i+1}. [{src_type}] {src_name}")

    # Tokenize-only 모드 체크 (early return)
    if args.tokenize_only:
        if is_main_process:
            logger.info("="*80)
            logger.info("🔥 Tokenize-Only Mode")
            logger.info("="*80)

        # 모든 소스 토크나이징
        for idx, source in enumerate(all_sources):
            # Rank 0만 토크나이징 실행
            if is_main_process:
                dataset_info = tokenize_single_source(source, tokenizer, args, idx)

            # 모든 rank 동기화
            ddp_barrier()

            # Non-main ranks는 캐시에서 정보 로드
            if not is_main_process:
                dataset_info = load_dataset_info_from_cache(source, tokenizer, args, idx)

        if is_main_process:
            logger.info("="*80)
            logger.info("✅ All datasets tokenized!")
            logger.info("="*80)
        return

    # ============================================================================
    # STEP 3: W&B 초기화 (선택적)
    # ============================================================================
    if args.use_wandb:
        try:
            import wandb
            if is_main_process:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config=vars(args),
                )
                logger.info(f"📊 W&B initialized: {args.wandb_project}")
        except ImportError:
            logger.warning("⚠️ wandb not installed")
            args.use_wandb = False

    # ============================================================================
    # STEP 4: 각 소스마다 토크나이징 → 학습 반복
    # ============================================================================
    if is_main_process:
        logger.info("="*80)
        logger.info("🎯 Starting Training Pipeline")
        logger.info(f"   Total stages: {len(all_sources)}")
        logger.info("="*80)

    current_checkpoint = args.pretrained_model

    for idx, source in enumerate(all_sources):
        if is_main_process:
            logger.info("")
            logger.info("="*80)
            logger.info(f"📊 Stage {idx+1}/{len(all_sources)}: {source[1]}")
            logger.info("="*80)

        # 1. 데이터셋 정보 로드 (토크나이징 또는 캐시)
        if args.skip_tokenization:
            # Skip tokenization 모드: 모든 rank가 캐시에서 직접 로드
            if is_main_process:
                logger.info("📥 Step 1: Loading from cache (skip_tokenization=True)...")

            # 캐시에서 정보 로드 (barrier 없이 각 rank가 독립적으로 로드)
            src_name, tokenized_cache_path, tokenized_marker = calculate_cache_paths(
                source, tokenizer, args, idx
            )

            # 마커 파일 확인
            if not tokenized_marker.exists():
                if is_main_process:
                    logger.error(f"❌ Tokenized cache not found for {src_name}")
                    logger.error(f"   Expected marker: {tokenized_marker}")
                    logger.error(f"   Please run tokenize_datasets.py first!")
                raise FileNotFoundError(f"Tokenized cache not found for {src_name}")

            # 샘플 수 확인을 위해 dataset 로드
            from datasets import Dataset as HFDataset
            import gc
            tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
            num_samples = len(tokenized_dataset)
            del tokenized_dataset
            gc.collect()

            dataset_info = {
                'name': src_name,
                'cache_path': tokenized_cache_path,
                'num_samples': num_samples,
            }

            if is_main_process:
                logger.info(f"✅ Loaded from cache: {num_samples:,} samples")
        else:
            # 일반 모드: Rank 0이 토크나이징, 다른 ranks는 대기 후 로드
            if is_main_process:
                logger.info("🔤 Step 1: Tokenizing...")
                dataset_info = tokenize_single_source(source, tokenizer, args, idx)

            # Barrier (모든 rank 동기화)
            if is_main_process:
                logger.info("⏳ Step 2: Synchronizing all ranks...")
            ddp_barrier()
            if is_main_process:
                logger.info("✅ All ranks synchronized!")

            # Non-main ranks는 캐시에서 정보 로드
            if not is_main_process:
                dataset_info = load_dataset_info_from_cache(source, tokenizer, args, idx)

        # 2. 학습 (모든 rank)
        if is_main_process:
            step_num = 2 if args.skip_tokenization else 4
            logger.info(f"🏋️ Step {step_num}: Training...")
        checkpoint_path = train_single_dataset(
            args=args,
            dataset_info=dataset_info,
            tokenizer=tokenizer,
            pretrained_model_path=current_checkpoint,
            is_first_stage=(idx == 0),
            stage_name=f"stage_{idx+1}",
        )

        # 다음 stage를 위해 체크포인트 업데이트
        current_checkpoint = checkpoint_path

    # ============================================================================
    # STEP 5: 최종 완료
    # ============================================================================
    if is_main_process:
        logger.info("")
        logger.info("="*80)
        logger.info("🎉 All training completed!")
        logger.info(f"📁 Final model: {current_checkpoint}")
        logger.info(f"📊 Trained on {len(all_sources)} datasets")
        logger.info("="*80)


if __name__ == "__main__":
    # 가장 먼저 출력 (torchrun이 프로세스를 제대로 시작했는지 확인)
    import sys
    import os
    
    # 즉시 출력
    rank = int(os.environ.get(ENV_RANK, -1))
    print(f"[INIT] Rank {rank}: Python script started!", flush=True)

    world_size = int(os.environ.get(ENV_WORLD_SIZE, 1))
    
    if rank == 0:
        print("="*80, flush=True)
        print("🚀 MOAI-LLM Training Starting...", flush=True)
        print(f"🌐 World size: {world_size} GPUs", flush=True)
        print("="*80, flush=True)

    main()

