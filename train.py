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

# Early initialization
import os
import sys
import time as time_module
from pathlib import Path as PathType

# Check rank early
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main = (rank == 0)

# 동기화 마커 파일
import_marker = PathType("/tmp/.moai_import_done")

if is_main:
    # Rank 0: 먼저 import
    print(f"[IMPORT] Rank 0: Importing modules (world_size={world_size})...", flush=True)
    sys.stdout.flush()
    
    # 이전 마커 제거
    if import_marker.exists():
        import_marker.unlink()
    
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
    
    print(f"[IMPORT] Rank 0: Importing torch...", flush=True)
    sys.stdout.flush()
    import torch
    
    print(f"[IMPORT] Rank 0: Importing transformers...", flush=True)
    sys.stdout.flush()
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    
    print(f"[IMPORT] Rank 0: Importing datasets...", flush=True)
    sys.stdout.flush()
    from datasets import load_dataset, disable_caching
    import datasets
    datasets.config.IN_MEMORY_MAX_SIZE = 0
    
    print(f"[IMPORT] Rank 0: Importing moai_llm...", flush=True)
    sys.stdout.flush()
    from moai_llm.config import MoaiConfig
    from moai_llm.modeling.model import MoaiForCausalLM
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # 마커 생성 (다른 rank들이 import 시작 가능)
    import_marker.touch()
    print(f"[IMPORT] Rank 0: ✅ All modules imported!", flush=True)
    sys.stdout.flush()
else:
    # 다른 rank들: 마커 대기
    print(f"[IMPORT] Rank {rank}: Waiting for rank 0...", flush=True)
    sys.stdout.flush()
    
    max_wait = 300  # 5분
    waited = 0
    while not import_marker.exists() and waited < max_wait:
        time_module.sleep(0.5)
        waited += 0.5
    
    if not import_marker.exists():
        print(f"[IMPORT] Rank {rank}: Timeout waiting for rank 0!", flush=True)
        sys.exit(1)
    
    # 이제 안전하게 import
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
    
    print(f"[IMPORT] Rank {rank}: ✅ Modules imported!", flush=True)
    sys.stdout.flush()


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
    
    이렇게 하면 max_seq_length로 잘리더라도 다음 청크에서 
    이어서 EOS까지 온전히 학습할 수 있습니다.
    
    Args:
        tokenized_sequences: 토큰화된 시퀀스 리스트 (각각 input_ids 포함)
        max_seq_length: 최대 시퀀스 길이
        eos_token_id: EOS 토큰 ID
    
    Returns:
        연결 후 max_seq_length로 분할된 시퀀스 리스트
    """
    import numpy as np
    
    # 1. 총 길이 계산 (메모리 미리 할당용)
    total_len = 0
    for seq in tokenized_sequences:
        input_ids = seq["input_ids"]
        total_len += len(input_ids)
        if len(input_ids) > 0 and input_ids[-1] != eos_token_id:
            total_len += 1  # EOS 추가될 예정
    
    logger.info(f"📦 Concatenating {len(tokenized_sequences):,} sequences into ~{total_len:,} tokens")
    
    # 2. numpy 배열로 빠르게 연결 (메모리 효율적)
    all_tokens = np.empty(total_len, dtype=np.int32)
    offset = 0
    
    for seq in tokenized_sequences:
        input_ids = seq["input_ids"]
        seq_len = len(input_ids)
        
        if seq_len == 0:
            continue
            
        # 배열에 복사
        all_tokens[offset:offset + seq_len] = input_ids
        offset += seq_len
        
        # EOS 추가
        if input_ids[-1] != eos_token_id:
            all_tokens[offset] = eos_token_id
            offset += 1
    
    # 실제 사용된 길이로 자르기
    all_tokens = all_tokens[:offset]
    
    # 3. max_seq_length 청크로 분할 (list comprehension으로 빠르게)
    num_chunks = (len(all_tokens) + max_seq_length - 1) // max_seq_length
    chunks = []
    
    for i in range(num_chunks):
        start = i * max_seq_length
        end = min(start + max_seq_length, len(all_tokens))
        chunk = all_tokens[start:end].tolist()
        
        # 마지막 청크가 너무 짧으면 (< 128) 버림
        if len(chunk) < 128:
            logger.info(f"  Dropping short final chunk of {len(chunk)} tokens")
            continue
            
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
        })
    
    logger.info(f"✓ Created {len(chunks):,} chunks of max {max_seq_length} tokens each")
    
    return chunks


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
    # DDP 환경 확인 (환경 변수 사용 - Trainer 초기화 전에도 작동)
    try:
        # 환경 변수로 rank 확인 (torchrun이 설정)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        rank = int(os.environ.get("RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        
        # distributed가 초기화되었는지 확인
        is_distributed = False
        is_main_process = True
        
        # 환경 변수로 distributed 여부 확인
        if rank >= 0 and world_size > 1:
            is_distributed = True
            is_main_process = rank == 0
        elif torch.distributed.is_available():
            # torch.distributed가 초기화되었는지 확인
            try:
                if torch.distributed.is_initialized():
                    is_distributed = True
                    is_main_process = torch.distributed.get_rank() == 0
            except (AttributeError, RuntimeError, ValueError):
                pass
    except (AttributeError, RuntimeError, ValueError):
        is_distributed = False
        is_main_process = True
    
    # rank 변수 보존 (로깅용)
    try:
        current_rank = rank if 'rank' in locals() and rank >= 0 else (
            torch.distributed.get_rank() if is_distributed and torch.distributed.is_initialized() else 0
        )
    except (AttributeError, RuntimeError, ValueError):
        current_rank = 0
    
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
        # Path import (함수 내부에서 사용하기 위해)
        from pathlib import Path as PathLib
        # 먼저 최종 데이터셋이 이미 존재하는지 확인
        cache_home = os.environ.get("HF_HOME", os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache/huggingface")))
        cache_hash = hashlib.md5(f"{dataset_name}_{dataset_config}".encode()).hexdigest()[:16]
        dataset_save_path = PathLib(cache_home) / "datasets" / f"{cache_hash}_final"
        filter_marker_path = PathLib(cache_home) / "datasets" / f".{cache_hash}_filtered.marker"
        
        # 이미 처리된 데이터셋이 있으면 모든 rank가 로드 (재시작 시 안전)
        if dataset_save_path.exists() and filter_marker_path.exists():
            logger.info(f"    [Rank {current_rank}] ✅ Using existing processed dataset from: {dataset_save_path}")
            from datasets import Dataset
            import time
            load_start = time.time()
            converted = Dataset.load_from_disk(str(dataset_save_path))
            load_time = time.time() - load_start
            logger.info(f"    [Rank {current_rank}] Loaded {len(converted):,} samples in {load_time:.1f}s")
            
            # barrier 동기화
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                pass
            
            # 변환 결과 반환 (나머지 로직 건너뛰기)
            return converted
        
        # barrier는 distributed가 완전히 초기화된 후에만 사용
        try:
            if torch.distributed.is_initialized():
                # 모든 프로세스가 동기화 지점에 도달할 때까지 대기
                torch.distributed.barrier()
        except (RuntimeError, ValueError, AttributeError):
            # barrier 실패 시 환경 변수만으로 동기화 (rank 0만 다운로드)
            pass
        
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
    # 환경 변수로 조정 가능한 최적화 파라미터
    # 높은 num_proc = 각 프로세스가 독립적으로 토크나이저 실행 → 빠름!
    dataset_num_proc = int(os.getenv("DATASET_NUM_PROC", min(48, os.cpu_count() or 8)))
    dataset_batch_size = int(os.getenv("DATASET_BATCH_SIZE", 1000))
    dataset_writer_batch_size = int(os.getenv("DATASET_WRITER_BATCH_SIZE", 10000))
    
    # DDP 환경에서는 rank 0만 변환하고 다른 프로세스는 캐시만 로드
    if is_distributed:
        # 캐시 완료 마커 파일 경로 생성
        cache_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        config_str = f"{dataset_name}_{dataset_config}" if dataset_config else dataset_name
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        cache_marker = PathLib(cache_home) / "datasets" / f".{cache_hash}_converted.marker"
        
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
            
            # 변환 완료 마커 생성 (filter 전에!)
            cache_marker.parent.mkdir(parents=True, exist_ok=True)
            cache_marker.touch()
            logger.info(f"    [Rank 0] Created conversion marker: {cache_marker}")
            
            # 빈 텍스트 필터링 (병렬 처리로 빠르게)
            filter_num_proc = min(dataset_num_proc // 2, 4)
            logger.info(f"    [Rank 0] Filtering empty texts with {filter_num_proc} processes...")
            converted = converted.filter(
                lambda x: len(x["text"]) > 0, 
                num_proc=filter_num_proc,  # 병렬 처리로 빠르게
                writer_batch_size=dataset_writer_batch_size,
                keep_in_memory=False,
                load_from_cache_file=True,  # 캐시 활용
            )
            
            logger.info(f"    [Rank 0] Conversion completed: {len(converted):,} samples")
            
            # 최종 결과를 디스크에 저장 (다른 rank들이 안전하게 로드할 수 있도록)
            dataset_save_path = PathLib(cache_home) / "datasets" / f"{cache_hash}_final"
            
            # 이미 저장된 파일이 있으면 건너뛰기 (속도 향상)
            if dataset_save_path.exists():
                logger.info(f"    [Rank 0] Dataset already saved at: {dataset_save_path}")
            else:
                logger.info(f"    [Rank 0] Saving final dataset to: {dataset_save_path}")
                import time
                save_start = time.time()
                # num_shards 지정으로 병렬 저장 최적화
                converted.save_to_disk(
                    str(dataset_save_path),
                    num_shards=dataset_num_proc,  # 병렬 저장
                )
                save_time = time.time() - save_start
                logger.info(f"    [Rank 0] Dataset saved in {save_time:.1f}s")
            
            # 필터 완료 마커 생성
            filter_marker = PathLib(str(cache_marker).replace("_converted.marker", "_filtered.marker"))
            filter_marker.touch()
            logger.info(f"    [Rank 0] Created filter marker: {filter_marker}")
            
            # 변환 완료 후 barrier
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                import time
                time.sleep(1)
                
        else:
            # 다른 프로세스는 필터 마커 대기 후 최종 결과만 로드!
            import time
            max_wait_time = 3600  # 최대 1시간 대기
            check_interval = 5
            
            # 필터 완료 마커 대기 (변환 마커는 건너뛰고 바로 필터 마커만 확인)
            filter_marker = PathLib(str(cache_marker).replace("_converted.marker", "_filtered.marker"))
            logger.info(f"    [Rank {current_rank}] Waiting for rank 0 to complete all processing...")
            waited = 0
            while not filter_marker.exists() and waited < max_wait_time:
                time.sleep(check_interval)
                waited += check_interval
                if waited % 60 == 0:  # 1분마다 로그
                    logger.info(f"    [Rank {current_rank}] Still waiting... ({waited}s elapsed)")
            
            if not filter_marker.exists():
                raise TimeoutError(f"Rank {current_rank}: Dataset processing timeout after {max_wait_time}s")
            
            logger.info(f"    [Rank {current_rank}] Processing complete, loading final result from cache...")
            
            # barrier 동기화
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                time.sleep(2)
            
            # rank 0이 저장한 최종 데이터셋을 직접 로드 (캐시 충돌 없음!)
            dataset_save_path = PathLib(cache_home) / "datasets" / f"{cache_hash}_final"
            logger.info(f"    [Rank {current_rank}] Loading final dataset from: {dataset_save_path}")
            
            # 파일이 완전히 준비될 때까지 짧은 대기 (파일 시스템 동기화)
            max_attempts = 60  # 최대 60번 시도 (30초)
            for attempt in range(max_attempts):
                if dataset_save_path.exists() and (dataset_save_path / "dataset_info.json").exists():
                    break
                time.sleep(0.5)
            else:
                logger.warning(f"    [Rank {current_rank}] Dataset files not fully ready, proceeding anyway...")
            
            from datasets import Dataset
            import time
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
        
        logger.info(f"    Filtering empty texts...")
        filter_num_proc = min(dataset_num_proc // 2, 4)
        converted = converted.filter(
            lambda x: len(x["text"]) > 0, 
            num_proc=filter_num_proc,
            load_from_cache_file=True,  # 캐시 사용
            writer_batch_size=dataset_writer_batch_size,
            keep_in_memory=False,
        )
    
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

    # 로컬 파일 로드
    if train_files:
        if isinstance(train_files, str):
            train_files = [train_files]
        
        for file_path in train_files:
            logger.info(f"  Loading file: {file_path}")
            file_data = _load_single_file(file_path)
            logger.info(f"    → {len(file_data):,} samples")
            datasets_list.append(Dataset.from_list(file_data))
    
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

def train_sequential(args):
    """
    데이터셋을 순차적으로 처리하는 학습 함수 (메모리 절약)
    
    ⚡ 최적화된 순서:
    1. 모든 데이터셋을 먼저 토큰화 (DDP 전, multiprocessing 사용!)
    2. DDP 초기화 및 모델 로드
    3. 각 데이터셋으로 순차 학습 (이미 토큰화된 데이터 사용)
    4. 체크포인트 저장 및 메모리 해제
    """
    import gc
    import sys
    
    # DDP 환경 정보
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"🌐 Environment: {world_size} GPU(s), Sequential Mode")
        logger.info(f"⚡ Strategy: Pre-tokenize all datasets, then train sequentially")
    sys.stdout.flush()
    
    # ========================================================================
    # STEP 0: 데이터 소스 리스트 준비
    # ========================================================================
    dataset_names = args.dataset if args.dataset else []
    train_files = args.train_file if args.train_file else []
    
    all_sources = []
    for ds in dataset_names:
        all_sources.append(("hf", ds))
    for f in train_files:
        all_sources.append(("file", f))
    
    if is_main_process:
        logger.info(f"📋 Sequential Mode: Processing {len(all_sources)} datasets")
        for i, (src_type, src_name) in enumerate(all_sources):
            logger.info(f"  {i+1}. [{src_type}] {src_name}")
    sys.stdout.flush()
    
    # ========================================================================
    # STEP 1: 토크나이저 로드 (DDP 전!)
    # ========================================================================
    if is_main_process:
        logger.info("📝 [Rank 0] Loading tokenizer...")
    sys.stdout.flush()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_main_process:
        if not tokenizer.is_fast:
            logger.warning("⚠️ WARNING: Using slow tokenizer!")
        else:
            logger.info("✅ [Rank 0] Using Fast Tokenizer (Rust-based)")
    sys.stdout.flush()
    
    # ========================================================================
    # STEP 2: 모든 데이터셋을 먼저 토큰화 (DDP 전! Rank 0만 실행!)
    # ========================================================================
    tokenized_datasets_info = []  # 각 데이터셋의 정보 저장
    
    # ⚡ Rank 0만 토큰화 수행, 다른 Rank는 완전히 대기
    if is_main_process:
        logger.info("="*80)
        logger.info("⚡ STEP 2: Pre-tokenizing all datasets (Rank 0 only, before DDP)")
        logger.info("="*80)
        sys.stdout.flush()
        
        for idx, (src_type, src_name) in enumerate(all_sources):
            logger.info(f"")
            logger.info(f"📦 [{idx+1}/{len(all_sources)}] Dataset: {src_name}")
            sys.stdout.flush()
            
            # 데이터셋 로드
            logger.info(f"  Loading dataset...")
            
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
            
            # 토큰화 캐시 경로
            cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            dataset_hash = hashlib.md5(f"{src_name}_seq_{idx}".encode()).hexdigest()[:16]
            tokenized_cache_path = Path(cache_home) / "datasets" / f"{dataset_hash}_tokenized"
            tokenized_marker = Path(cache_home) / "datasets" / f".{dataset_hash}_tokenized.marker"
            
            from datasets import Dataset as HFDataset
            
            if tokenized_cache_path.exists() and tokenized_marker.exists():
                # 캐시가 있으면 로드만
                logger.info(f"  ✅ Loading cached tokenized dataset")
                tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
                logger.info(f"  ✅ Loaded {len(tokenized_dataset):,} samples")
            else:
                # 캐시가 없으면 토큰화
                logger.info(f"  🔤 Tokenizing with BATCH ITERATOR...")
                
                if args.packing:
                    import time
                    
                    train_data = dataset["train"]
                    total_samples = len(train_data)
                    batch_size = 50000  # 5만 개씩 배치
                    
                    logger.info(f"  ⚡ Batch Iterator Tokenization")
                    logger.info(f"     Total samples: {total_samples:,}")
                    logger.info(f"     Batch size: {batch_size:,}")
                    sys.stdout.flush()
                    
                    all_input_ids = []
                    start_time = time.time()
                    samples_done = 0
                    
                    # iter()를 사용하여 배치 단위로 빠르게 순회
                    for batch in train_data.iter(batch_size=batch_size):
                        texts = batch[text_column]
                        
                        # 배치 토크나이징 (Fast Tokenizer 내부 병렬 처리)
                        tokenized = tokenizer(
                            texts,
                            truncation=False,
                            padding=False,
                            add_special_tokens=True,
                        )
                        
                        all_input_ids.extend(tokenized["input_ids"])
                        samples_done += len(texts)
                        
                        # 진행률 출력 (10만 개마다)
                        if samples_done % 100000 == 0 or samples_done == total_samples:
                            elapsed = time.time() - start_time
                            samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
                            eta = (total_samples - samples_done) / samples_per_sec if samples_per_sec > 0 else 0
                            
                            logger.info(f"  📦 Progress: {samples_done:,}/{total_samples:,} "
                                       f"({100*samples_done/total_samples:.1f}%) "
                                       f"[{samples_per_sec:.0f} samples/s, ETA: {eta/60:.1f}min]")
                            sys.stdout.flush()
                    
                    total_time = time.time() - start_time
                    logger.info(f"  ✅ Tokenization completed in {total_time/60:.1f} minutes")
                    logger.info(f"     Average speed: {total_samples/total_time:.0f} samples/s")
                    sys.stdout.flush()
                    
                    # Packing
                    logger.info(f"  📦 Packing sequences...")
                    sys.stdout.flush()
                    tokenized_list = [{"input_ids": ids} for ids in all_input_ids]
                    del all_input_ids
                    gc.collect()
                    
                    concatenated_chunks = concatenate_sequences(
                        tokenized_sequences=tokenized_list,
                        max_seq_length=args.max_seq_length,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    del tokenized_list
                    gc.collect()
                    
                    tokenized_dataset = HFDataset.from_list(concatenated_chunks)
                    del concatenated_chunks
                    gc.collect()
                else:
                    def tokenize_function(examples):
                        return tokenizer(
                            examples[text_column],
                            truncation=True,
                            max_length=args.max_seq_length,
                            padding=False,
                            return_special_tokens_mask=True,
                        )
                    
                    # ⚡ 최적화: num_proc=1 + Fast Tokenizer 내부 병렬화 (가장 빠름!)
                    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Fast Tokenizer 병렬화 활성화
                    import multiprocessing
                    cpu_count = multiprocessing.cpu_count()
                    
                    # 최적 프로세스 수 계산
                    optimal_num_proc = int(os.getenv("DATASET_NUM_PROC", min(48, multiprocessing.cpu_count())))
                    
                    logger.info(f"  ⚡ Parallel Tokenization: {optimal_num_proc} processes ({cpu_count} CPUs)")
                    logger.info(f"     Strategy: Each process runs tokenizer independently → FAST!")
                    
                    tokenized_dataset = dataset["train"].map(
                        tokenize_function,
                        batched=True,
                        batch_size=5000,
                        num_proc=optimal_num_proc,  # ⚡ 48개 프로세스 동시 실행!
                        remove_columns=dataset["train"].column_names,
                        load_from_cache_file=False,
                        writer_batch_size=100000,
                        keep_in_memory=False,
                        desc=f"Tokenizing {src_name} (num_proc={optimal_num_proc})",
                    )
                
                # 캐시 저장
                logger.info(f"  💾 Saving tokenized dataset...")
                tokenized_dataset.save_to_disk(str(tokenized_cache_path), num_shards=8)
                tokenized_marker.touch()
                logger.info(f"  ✅ Tokenized: {len(tokenized_dataset):,} samples")
            
            # 정보 저장
            tokenized_datasets_info.append({
                'name': src_name,
                'cache_path': tokenized_cache_path,
                'num_samples': len(tokenized_dataset),
            })
            
            # 메모리 해제
            del dataset
            del tokenized_dataset
            gc.collect()
        
        logger.info("="*80)
        logger.info("✅ All datasets pre-tokenized!")
        logger.info("="*80)
        sys.stdout.flush()
    else:
        # 다른 Rank들은 Rank 0이 모든 토큰화를 완료할 때까지 대기
        logger.info(f"[Rank {rank}] Waiting for rank 0 to complete all tokenization...")
        sys.stdout.flush()
        
        # 마지막 데이터셋의 마커를 기다림
        import time as time_module
        cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        last_src_name = all_sources[-1][1]
        last_idx = len(all_sources) - 1
        dataset_hash = hashlib.md5(f"{last_src_name}_seq_{last_idx}".encode()).hexdigest()[:16]
        last_marker = Path(cache_home) / "datasets" / f".{dataset_hash}_tokenized.marker"
        
        max_wait = 7200
        waited = 0
        while not last_marker.exists() and waited < max_wait:
            time_module.sleep(10)
            waited += 10
            if waited % 60 == 0:
                logger.info(f"[Rank {rank}] Still waiting... ({waited}s)")
        
        if not last_marker.exists():
            raise TimeoutError(f"Rank {rank}: Tokenizing timeout after {max_wait}s")
        
        logger.info(f"[Rank {rank}] ✅ Rank 0 completed tokenization! Loading datasets...")
        sys.stdout.flush()
        
        # 토큰화된 데이터셋 정보 로드
        from datasets import Dataset as HFDataset
        for idx, (src_type, src_name) in enumerate(all_sources):
            cache_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            dataset_hash = hashlib.md5(f"{src_name}_seq_{idx}".encode()).hexdigest()[:16]
            tokenized_cache_path = Path(cache_home) / "datasets" / f"{dataset_hash}_tokenized"
            
            tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
            
            tokenized_datasets_info.append({
                'name': src_name,
                'cache_path': tokenized_cache_path,
                'num_samples': len(tokenized_dataset),
            })
            
            del tokenized_dataset
            gc.collect()
        
        logger.info(f"[Rank {rank}] ✅ All dataset info loaded!")
        sys.stdout.flush()
    
    # ========================================================================
    # Barrier: 모든 Rank 동기화 (토큰화 완료 후)
    # ========================================================================
    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info(f"[Rank {rank}] Synchronizing with other ranks...")
            sys.stdout.flush()
            dist.barrier()
            logger.info(f"[Rank {rank}] ✅ All ranks synchronized!")
            sys.stdout.flush()
    
    # ========================================================================
    # Tokenize-only 모드: 여기서 종료
    # ========================================================================
    if hasattr(args, '_tokenize_only') and args._tokenize_only:
        logger.info("="*80)
        logger.info("✅ Tokenization completed! Exiting (tokenize-only mode)")
        logger.info("="*80)
        return
    
    # ========================================================================
    # STEP 3: W&B 초기화 (선택적)
    # ========================================================================
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
            logger.warning("⚠️ wandb not installed, falling back to tensorboard")
            args.use_wandb = False
    
    # ========================================================================
    # STEP 4: 순차 학습 (각 토큰화된 데이터셋으로)
    # ========================================================================
    if is_main_process:
        logger.info("="*80)
        logger.info("🎯 STEP 4: Sequential Training")
        logger.info("="*80)
    sys.stdout.flush()
    
    current_checkpoint = args.pretrained_model
    
    for idx, dataset_info in enumerate(tokenized_datasets_info):
        if is_main_process:
            logger.info("")
            logger.info("="*80)
            logger.info(f"🚀 Training [{idx+1}/{len(tokenized_datasets_info)}]: {dataset_info['name']}")
            logger.info(f"   Samples: {dataset_info['num_samples']:,}")
            logger.info("="*80)
        sys.stdout.flush()
        
        # 모델 로드 (토크나이저는 재사용)
        if is_main_process:
            if idx == 0:
                logger.info(f"⏳ Loading model from: {current_checkpoint or 'scratch'}")
            else:
                logger.info(f"⏳ Resuming from: {current_checkpoint}")
        
        model, _ = setup_model_and_tokenizer(
            tokenizer_path=args.tokenizer_path,
            model_config=args.model_config,
            pretrained_model=current_checkpoint,
            use_flash_attention=args.flash_attention,
            use_compile=args.compile,
            use_bf16=args.bf16,
            use_fp16=args.fp16,
        )
        if is_main_process:
            logger.info(f"✓ Model loaded")
        
        # 토큰화된 데이터셋 로드
        if is_main_process:
            logger.info(f"📥 Loading tokenized dataset...")
        
        from datasets import Dataset as HFDataset
        tokenized_dataset = HFDataset.load_from_disk(str(dataset_info['cache_path']))
        
        if is_main_process:
            logger.info(f"✓ Loaded {len(tokenized_dataset):,} samples")
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training Arguments
        stage_output_dir = f"{args.output_dir}/stage_{idx+1}"
        
        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps if idx == 0 else 100,
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
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            dataloader_drop_last=True,
            optim="adamw_torch_fused",
            ddp_find_unused_parameters=False,
            tf32=True,
            group_by_length=False,
            max_grad_norm=1.0,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        if is_main_process:
            logger.info(f"🏃 Starting training...")
        trainer.train()
        
        # 체크포인트 저장
        checkpoint_path = f"{stage_output_dir}/checkpoint"
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
            logger.info(f"✅ Stage {idx+1} completed: {checkpoint_path}")
        
        # 다음 라운드를 위해 체크포인트 경로 업데이트
        current_checkpoint = checkpoint_path
        
        # 메모리 해제
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
    
    # 최종 완료
    if is_main_process:
        logger.info("="*80)
        logger.info("🎉 Sequential training completed!")
        logger.info(f"📁 Final model: {current_checkpoint}")
        logger.info("="*80)


# ============================================================================
# Main Train Function (Concatenated Mode)
# ============================================================================
def train(args):
    """메인 학습 함수"""
    
    import sys
    sys.stdout.flush()  # 즉시 출력
    
    logger.info("="*80)
    logger.info(f"🚀 Starting {args.mode.upper()} training")
    logger.info("="*80)
    sys.stdout.flush()
    
    # W&B 초기화 (사용하는 경우)
    if args.use_wandb:
        try:
            import wandb
            # DDP 환경에서는 rank 0만 초기화
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config=vars(args),
                )
                logger.info(f"📊 W&B initialized: {args.wandb_project}")
        except ImportError:
            logger.warning("⚠️ wandb not installed, falling back to tensorboard")
            args.use_wandb = False
    
    # Sequential 모드: 각 데이터셋을 순차적으로 처리
    if args.sequential and args.dataset and len(args.dataset) > 1:
        logger.info("📦 Sequential mode: Processing datasets one by one")
        train_sequential(args)
        return

    # ============================================================================
    # DDP 환경 확인 (STEP 0 전에 먼저 확인)
    # ============================================================================
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"🌐 Environment: {world_size} GPU(s), Rank {rank}")
    sys.stdout.flush()

    # ============================================================================
    # STEP 0: 토크나이저만 먼저 로드 (DDP 전!)
    # ============================================================================
    if is_main_process:
        logger.info("📝 [Rank 0] Loading tokenizer (before DDP)...")
    sys.stdout.flush()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_main_process:
        if not tokenizer.is_fast:
            logger.warning("⚠️ WARNING: Using slow tokenizer! This will be very slow.")
        else:
            logger.info("✅ [Rank 0] Using Fast Tokenizer (Rust-based)")
    sys.stdout.flush()

    # ============================================================================
    # STEP 1: 데이터셋 로드 및 토크나이징 (DDP 전! multiprocessing 사용 가능!)
    # ============================================================================
    # DDP 환경 확인 (환경 변수 사용)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info("📚 [Rank 0] Loading datasets (may take 2-5 minutes for large datasets)...")
        logger.info("⚡ Rank 0 will process data, others will load from cache!")
        sys.stdout.flush()
    else:
        logger.info(f"📚 [Rank {rank}] Waiting for rank 0 to complete data processing...")
        sys.stdout.flush()
    
    # 데이터셋 로드
    import time
    load_start = time.time()
    
    if args.mode == "pretrain":
        if is_main_process:
            logger.info(f"[Rank 0] Loading {len(args.dataset) if args.dataset else 0} datasets...")
        dataset, text_column = load_pretrain_dataset(
            dataset_names=args.dataset,
            dataset_config=args.dataset_config,
            train_files=args.train_file,
            text_column=args.text_column,
        )
    else:  # sft
        dataset, text_column = load_sft_dataset(
            dataset_names=args.dataset,
            train_files=args.train_file,
        )
    
    load_time = time.time() - load_start
    if is_main_process:
        logger.info(f"✅ [Rank 0] Dataset loaded in {load_time:.1f}s: {len(dataset['train']):,} samples")
    else:
        logger.info(f"✅ [Rank {rank}] Dataset loaded in {load_time:.1f}s: {len(dataset['train']):,} samples")

    # 토크나이징 캐시 경로
    cache_home = os.environ.get("HF_HOME", os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache/huggingface")))
    dataset_names_str = "_".join(args.dataset) if args.dataset else "local"
    dataset_hash = hashlib.md5(dataset_names_str.encode()).hexdigest()[:16]
    tokenized_cache_path = Path(cache_home) / "datasets" / f"{dataset_hash}_tokenized"
    tokenized_marker = Path(cache_home) / "datasets" / f".{dataset_hash}_tokenized.marker"

    # Rank 0만 토큰화 수행
    if is_main_process:
        logger.info("🔤 [Rank 0] Tokenizing dataset...")
        
        # 토크나이저 워밍업
        logger.info("🔥 Warming up tokenizer...")
        warmup_texts = ["Hello world " * 100] * 10
        _ = tokenizer(warmup_texts, truncation=False, padding=False)
        logger.info("✅ Tokenizer warmed up")

    # 캐시 확인 및 로드
    from datasets import Dataset as HFDataset
    
    if tokenized_cache_path.exists() and tokenized_marker.exists():
        # 캐시가 있으면 모든 rank가 로드
        if is_main_process:
            logger.info(f"✅ [Rank 0] Loading cached tokenized dataset from: {tokenized_cache_path}")
        else:
            logger.info(f"✅ [Rank {rank}] Loading cached tokenized dataset from: {tokenized_cache_path}")
        tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
        if is_main_process:
            logger.info(f"✅ [Rank 0] Loaded {len(tokenized_dataset):,} samples from cache")
        else:
            logger.info(f"✅ [Rank {rank}] Loaded {len(tokenized_dataset):,} samples from cache")
    elif is_main_process:
        # Packing 모드: 시퀀스 연결 방식 사용
        if args.packing:
            logger.info(f"📦 Using sequence concatenation (packing mode)")
            
            # 배치 토큰화
            def batch_tokenize(examples):
                return tokenizer(
                    examples[text_column],
                    truncation=False,
                    padding=False,
                    add_special_tokens=True,
                )
            
            # Multiprocessing 사용 (DDP 전이므로 자유롭게!)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # datasets가 multiprocessing 시 강제
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            optimal_num_proc = min(32, max(16, cpu_count // 6))
            
            logger.info(f"⚡ Multiprocessing tokenization: {optimal_num_proc} processes")
            logger.info(f"   CPU cores: {cpu_count}, batch_size=50000")
            logger.info(f"   Expected time: {len(dataset['train']) / (optimal_num_proc * 7000) / 60:.1f} minutes")
            
            tokenized_ds = dataset["train"].map(
                batch_tokenize,
                batched=True,
                batch_size=50000,
                num_proc=optimal_num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                writer_batch_size=100000,
                keep_in_memory=False,
                desc="Tokenizing",
            )
            
            logger.info("📦 Packing sequences...")
            tokenized_list = [{"input_ids": ids} for ids in tokenized_ds["input_ids"]]
            del tokenized_ds
            
            concatenated_chunks = concatenate_sequences(
                tokenized_sequences=tokenized_list,
                max_seq_length=args.max_seq_length,
                eos_token_id=tokenizer.eos_token_id,
            )
            del tokenized_list
            
            from datasets import Dataset as HFDataset
            tokenized_dataset = HFDataset.from_list(concatenated_chunks)
            del concatenated_chunks
            
        else:
            # 일반 mode: truncation
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column],
                    truncation=True,
                    max_length=args.max_seq_length,
                    padding=False,
                    return_special_tokens_mask=True,
                )
            
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            optimal_num_proc = min(32, max(16, cpu_count // 6))
            
            logger.info(f"⚡ Multiprocessing tokenization: {optimal_num_proc} processes")
            logger.info(f"   CPU cores: {cpu_count}, batch_size=50000")
            
            tokenized_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                batch_size=50000,
                num_proc=optimal_num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                writer_batch_size=100000,
                keep_in_memory=False,
                desc="Tokenizing",
            )
        
        # 캐시 저장 (rank 0만)
        logger.info(f"💾 [Rank 0] Saving tokenized dataset to: {tokenized_cache_path}")
        tokenized_dataset.save_to_disk(str(tokenized_cache_path), num_shards=8)
        tokenized_marker.touch()
        logger.info(f"✅ [Rank 0] Tokenized and saved: {len(tokenized_dataset):,} samples")
    else:
        # 다른 rank들은 마커 대기 후 로드
        import time
        max_wait = 7200  # 최대 2시간
        waited = 0
        while not tokenized_marker.exists() and waited < max_wait:
            time.sleep(5)
            waited += 5
            if waited % 60 == 0:
                logger.info(f"  [Rank {rank}] Still waiting for rank 0... ({waited}s)")
        
        if not tokenized_marker.exists():
            raise TimeoutError(f"Rank {rank}: Tokenizing timeout after {max_wait}s")
        
        logger.info(f"📥 [Rank {rank}] Loading tokenized dataset from: {tokenized_cache_path}")
        tokenized_dataset = HFDataset.load_from_disk(str(tokenized_cache_path))
        logger.info(f"✅ [Rank {rank}] Loaded {len(tokenized_dataset):,} samples")
    
    if is_main_process:
        logger.info(f"✅ Tokenization complete: {len(tokenized_dataset):,} samples ready for training")
    else:
        logger.info(f"✅ [Rank {rank}] Ready for training with {len(tokenized_dataset):,} samples")
    
    # ============================================================================
    # STEP 2: DDP 초기화 및 모델 로드
    # ============================================================================
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        logger.info(f"🌐 Distributed Training: Rank {rank}/{world_size}")
        logger.info(f"⏳ Initializing DDP environment...")
    else:
        logger.info(f"💻 Single GPU Training")
    
    logger.info("⏳ Loading model...")
    model, _ = setup_model_and_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_config=args.model_config,
        pretrained_model=args.pretrained_model,
        use_flash_attention=args.flash_attention,
        use_compile=args.compile,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
    )
    
    # ============================================================================
    # STEP 3: 학습 시작
    # ============================================================================
    logger.info("🚀 Starting training with pre-tokenized data...")
    
    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # 5. Training Arguments (최적화 옵션 포함)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        save_safetensors=True,
        ddp_find_unused_parameters=False,
        # 추가 최적화 옵션
        dataloader_pin_memory=True,  # GPU 전송 속도 향상
        dataloader_prefetch_factor=4,  # 미리 배치 로드
        dataloader_drop_last=True,  # 불완전 배치 제거 (속도↑)
        optim="adamw_torch_fused",  # Fused Adam (faster)
        tf32=True,  # TF32 사용 (Ampere GPU)
        group_by_length=False,  # 길이별 그룹핑 비활성화 (packing 사용시)
        max_grad_norm=1.0,  # 그래디언트 클리핑
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. 학습 시작
    logger.info("="*80)
    logger.info("🎯 Training configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Packing: {args.packing}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max steps: {args.max_steps if args.max_steps > 0 else 'Full epoch'}")
    logger.info(f"  Logging: {'wandb' if args.use_wandb else 'tensorboard'}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from: {args.resume_from_checkpoint}")
    logger.info("="*80)

    logger.info("🏃 Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. 모델 저장
    logger.info("💾 Saving final model...")
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    
    # 모델 dtype 확인 및 bf16/fp16으로 명시적 저장
    model_dtype = next(model.parameters()).dtype
    if model_dtype in (torch.bfloat16, torch.float16):
        logger.info(f"💾 Re-saving model in {model_dtype} format for compatibility...")
        model.save_pretrained(str(final_path), torch_dtype=model_dtype, safe_serialization=True)
    
    tokenizer.save_pretrained(str(final_path))
    
    logger.info("="*80)
    logger.info(f"✅ Training completed!")
    logger.info(f"📁 Model saved to: {final_path}")
    logger.info(f"📊 Model dtype: {model_dtype}")
    logger.info("="*80)


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

    # Tokenize only 모드: DDP 없이 토큰화만 수행
    if args.tokenize_only:
        print("="*80)
        print("🔥 Tokenize-Only Mode: Pre-tokenizing datasets (no DDP)")
        print("="*80)
        
        # Sequential이 필요
        if not args.sequential:
            args.sequential = True
            print("⚡ Automatically enabling --sequential mode for tokenization")
        
        # 핵심: 단일 프로세스로 Fast Tokenizer 사용
        # num_proc=1 → datasets가 TOKENIZERS_PARALLELISM=false 설정 안함
        # TOKENIZERS_PARALLELISM=true → Fast Tokenizer 내부 병렬화 활성화
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["RAYON_NUM_THREADS"] = str(os.cpu_count() or 96)
        os.environ["DATASET_NUM_PROC"] = "1"  # 핵심! 단일 프로세스로 변환
        print(f"⚡ DATASET_NUM_PROC=1 + TOKENIZERS_PARALLELISM=true + RAYON_NUM_THREADS={os.cpu_count()}")
        
        # train_sequential 호출 (토큰화 부분만 실행됨)
        print("🚀 Calling train_sequential for tokenization...")
        
        # DDP 환경 변수 제거 (단일 프로세스로 실행)
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        
        # tokenization만 수행하고 training은 스킵하도록 플래그 설정
        args._tokenize_only = True
        
        train_sequential(args)
        
        print("="*80)
        print("✅ Tokenization completed! Now run torchrun for training.")
        print("="*80)
        return

    # 학습 시작
    train(args)


if __name__ == "__main__":
    # 가장 먼저 출력 (torchrun이 프로세스를 제대로 시작했는지 확인)
    import sys
    import os
    
    # 즉시 출력
    rank = int(os.environ.get("RANK", -1))
    print(f"[INIT] Rank {rank}: Python script started!", flush=True)
    sys.stdout.flush()
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank == 0:
        print("="*80, flush=True)
        print("🚀 MOAI-LLM Training Starting...", flush=True)
        print(f"🌐 World size: {world_size} GPUs", flush=True)
        print("="*80, flush=True)
    
    sys.stdout.flush()
    main()

