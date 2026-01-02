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

import argparse
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
try:
    import orjson as json  # Rust-based, 10-50x faster
except ImportError:
    import json  # Fallback to standard json

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, disable_caching
import datasets

# Enable memory-efficient settings for large datasets
datasets.config.IN_MEMORY_MAX_SIZE = 0  # Force memory mapping (no in-memory)

from moai_llm.config import MoaiConfig
from moai_llm.modeling.model import MoaiForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


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
    
    # DDP 환경에서는 rank 0만 데이터셋을 다운로드 및 변환
    if is_distributed:
        # barrier는 distributed가 완전히 초기화된 후에만 사용
        try:
            if torch.distributed.is_initialized():
                # 모든 프로세스가 동기화 지점에 도달할 때까지 대기
                torch.distributed.barrier()
        except (RuntimeError, ValueError, AttributeError):
            # barrier 실패 시 환경 변수만으로 동기화 (rank 0만 다운로드)
            pass
        
        if is_main_process:
            # rank 0만 데이터셋 다운로드
            logger.info(f"    [Rank 0] Downloading dataset...")
            if dataset_config:
                logger.info(f"    Config: {dataset_config}")
                raw_dataset = load_dataset(dataset_name, dataset_config)
            else:
                raw_dataset = load_dataset(dataset_name)
            logger.info(f"    [Rank 0] Dataset download completed")
            
            # 다운로드 완료 후 barrier (가능한 경우)
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                pass
        else:
            # 다른 프로세스는 rank 0이 다운로드 완료할 때까지 대기
            logger.info(f"    [Rank {current_rank}] Waiting for dataset download...")
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                # barrier 실패 시 짧은 대기 후 진행 (rank 0이 다운로드 완료했을 것으로 가정)
                import time
                time.sleep(2)
            
            # barrier 통과 후 캐시된 데이터셋 로드 (다운로드 없이)
            logger.info(f"    [Rank {current_rank}] Loading cached dataset...")
            if dataset_config:
                raw_dataset = load_dataset(dataset_name, dataset_config, download_mode="reuse_cache_if_exists")
            else:
                raw_dataset = load_dataset(dataset_name, download_mode="reuse_cache_if_exists")
    else:
        # 단일 프로세스 환경
        if dataset_config:
            logger.info(f"    Config: {dataset_config}")
            raw_dataset = load_dataset(dataset_name, dataset_config)
        else:
            raw_dataset = load_dataset(dataset_name)
    
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
    # DDP 환경에서는 rank 0만 변환하고 다른 프로세스는 대기하여 캐시 파일 충돌 방지
    if is_distributed:
        # 모든 프로세스가 변환 시작 전에 동기화
        try:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except (RuntimeError, ValueError, AttributeError):
            pass
        
        if is_main_process:
            # rank 0만 데이터셋 변환 (캐시 파일 쓰기)
            # 단일 프로세스로 변환하여 캐시 파일 충돌 방지
            logger.info(f"    [Rank 0] Converting dataset...")
            converted = train_data.map(
                convert_batch,
                batched=True,
                batch_size=500,
                num_proc=1,  # 단일 프로세스로 변환 (캐시 파일 충돌 방지)
                remove_columns=train_data.column_names,
                load_from_cache_file=True,
                desc=f"Converting {dataset_name}",
            )
            logger.info(f"    [Rank 0] Dataset conversion completed")
            
            # 변환 완료 후 barrier (다른 프로세스가 진행하도록)
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                import time
                time.sleep(1)
        else:
            # 다른 프로세스는 rank 0이 변환 완료할 때까지 대기
            logger.info(f"    [Rank {current_rank}] Waiting for dataset conversion...")
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except (RuntimeError, ValueError, AttributeError):
                import time
                # rank 0이 변환을 완료할 충분한 시간 대기
                time.sleep(10)
            
            # barrier 통과 후 캐시된 데이터셋 로드 (변환 없이, 읽기 전용)
            logger.info(f"    [Rank {current_rank}] Loading cached converted dataset...")
            # 캐시가 완성되었는지 확인하기 위해 짧은 대기
            import time
            time.sleep(2)
            
            converted = train_data.map(
                convert_batch,
                batched=True,
                batch_size=500,
                num_proc=1,  # 단일 프로세스로 캐시 읽기만
                remove_columns=train_data.column_names,
                load_from_cache_file=True,  # 캐시에서만 로드
                desc=f"Loading cached {dataset_name}",
            )
    else:
        # 단일 프로세스 환경
        converted = train_data.map(
            convert_batch,
            batched=True,
            batch_size=500,
            num_proc=min(4, os.cpu_count() or 2),
            remove_columns=train_data.column_names,
            load_from_cache_file=True,
            desc=f"Converting {dataset_name}",
        )
    
    # 빈 텍스트 필터링 (단일 프로세스로 변경 - 버퍼 공간 부족 방지)
    # 병렬 필터링은 프로세스 간 통신 버퍼를 많이 사용하므로 단일 프로세스 사용
    # TOKENIZERS_PARALLELISM=true가 설정되어 있어 Rust 레벨에서 병렬화됨
    converted = converted.filter(lambda x: len(x["text"]) > 0, num_proc=1)
    
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델
    if pretrained_model:
        logger.info(f"🔄 Loading pretrained model: {pretrained_model}")
        model = MoaiForCausalLM.from_pretrained(pretrained_model, dtype=dtype)
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
    
    각 데이터셋에 대해:
    1. 해당 데이터셋만 로드
    2. 토큰화 및 학습
    3. 체크포인트 저장
    4. 메모리 해제
    5. 다음 데이터셋으로 (이전 체크포인트에서 resume)
    """
    import gc
    
    dataset_names = args.dataset if args.dataset else []
    train_files = args.train_file if args.train_file else []
    
    # 모든 데이터 소스 리스트
    all_sources = []
    for ds in dataset_names:
        all_sources.append(("hf", ds))
    for f in train_files:
        all_sources.append(("file", f))
    
    logger.info(f"📋 Processing {len(all_sources)} datasets sequentially:")
    for i, (src_type, src_name) in enumerate(all_sources):
        logger.info(f"  {i+1}. [{src_type}] {src_name}")
    
    current_checkpoint = args.pretrained_model
    
    for idx, (src_type, src_name) in enumerate(all_sources):
        logger.info("="*80)
        logger.info(f"🔄 [{idx+1}/{len(all_sources)}] Processing: {src_name}")
        logger.info("="*80)
        
        # 1. 모델 및 토크나이저 로드
        model, tokenizer = setup_model_and_tokenizer(
            tokenizer_path=args.tokenizer_path,
            model_config=args.model_config,
            pretrained_model=current_checkpoint,
            use_flash_attention=args.flash_attention,
            use_compile=args.compile,
            use_bf16=args.bf16,
            use_fp16=args.fp16,
        )
        
        # 2. 해당 데이터셋만 로드
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
        
        # 3. 토큰화
        logger.info("🔤 Tokenizing dataset...")
        
        if args.packing:
            logger.info(f"📦 Using sequence concatenation (packing mode)")
            
            # 배치 토큰화 (빠름)
            def batch_tokenize(examples):
                return tokenizer(
                    examples[text_column],
                    truncation=False,
                    padding=False,
                    add_special_tokens=True,
                )
            
            logger.info("  Batch tokenizing...")
            # 버퍼 공간 부족 방지를 위해 num_proc 제한
            # TOKENIZERS_PARALLELISM=true가 설정되어 있어 Rust 레벨에서 병렬화됨
            safe_num_proc = min(args.num_proc, 4)  # 최대 4개 프로세스로 제한
            tokenized_ds = dataset["train"].map(
                batch_tokenize,
                batched=True,
                batch_size=1000,
                num_proc=safe_num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=True,  # Use cache to avoid re-tokenizing
                desc="Tokenizing",
            )
            
            # input_ids 리스트로 변환
            tokenized_list = [{"input_ids": ids} for ids in tokenized_ds["input_ids"]]
            del tokenized_ds
            gc.collect()
            
            concatenated_chunks = concatenate_sequences(
                tokenized_sequences=tokenized_list,
                max_seq_length=args.max_seq_length,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            from datasets import Dataset as HFDataset
            tokenized_dataset = HFDataset.from_list(concatenated_chunks)
            
            # 메모리 해제
            del tokenized_list
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

            # 버퍼 공간 부족 방지를 위해 num_proc 제한
            safe_num_proc = min(args.num_proc, 4)  # 최대 4개 프로세스로 제한
            tokenized_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                num_proc=safe_num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=True,  # Use cache to avoid re-tokenizing
                desc="Tokenizing",
            )
        
        # 원본 데이터셋 메모리 해제
        del dataset
        gc.collect()
        
        logger.info(f"✓ Tokenized {len(tokenized_dataset)} samples")
        
        # 4. 학습
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 출력 디렉토리 (각 데이터셋별)
        stage_output_dir = f"{args.output_dir}/stage_{idx+1}"
        
        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps if idx == 0 else 100,  # 첫 번째만 full warmup
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="none",
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            # 추가 최적화 옵션
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4,
            dataloader_drop_last=True,  # 불완전 배치 제거 (속도↑)
            optim="adamw_torch_fused",  # Fused Adam (faster than 8-bit)
            ddp_find_unused_parameters=False,
            tf32=True,
            group_by_length=False,
            max_grad_norm=1.0,  # 그래디언트 클리핑
            gradient_checkpointing_kwargs={"use_reentrant": False},  # 최신 방식
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,  # tokenizer deprecated in v5.0
        )
        
        logger.info(f"🏃 Training on dataset {idx+1}/{len(all_sources)}...")
        trainer.train()
        
        # 5. 체크포인트 저장 (DDP 환경에서 동기화 필요)
        checkpoint_path = f"{stage_output_dir}/checkpoint"
        trainer.save_model(checkpoint_path)
        
        # DDP 환경에서 모든 rank가 저장 완료될 때까지 대기
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # 저장 확인 및 dtype 유지 (rank 0에서)
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process:
            # 모델 dtype 확인 및 bf16/fp16으로 재저장
            model_dtype = next(model.parameters()).dtype
            if model_dtype in (torch.bfloat16, torch.float16):
                logger.info(f"💾 Re-saving model in {model_dtype} format...")
                model.save_pretrained(checkpoint_path, torch_dtype=model_dtype, safe_serialization=True)
            
            saved_files = list(Path(checkpoint_path).glob("*.safetensors")) + \
                         list(Path(checkpoint_path).glob("*.bin"))
            if saved_files:
                logger.info(f"💾 Saved checkpoint to: {checkpoint_path}")
            else:
                logger.warning(f"⚠️  No model files found in: {checkpoint_path}")
        
        # 다음 라운드를 위해 체크포인트 경로 업데이트
        current_checkpoint = checkpoint_path
        
        # 6. 메모리 해제
        del model
        del tokenizer
        del tokenized_dataset
        del trainer
        gc.collect()
        
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        # DDP barrier (다음 단계 시작 전 동기화)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        logger.info(f"✅ Completed dataset {idx+1}/{len(all_sources)}")
    
    # 최종 모델 저장
    logger.info("="*80)
    logger.info("🎯 Sequential training completed!")
    logger.info(f"📁 Final model: {current_checkpoint}")
    logger.info("="*80)


def train(args):
    """메인 학습 함수"""

    logger.info("="*80)
    logger.info(f"🚀 Starting {args.mode.upper()} training")
    logger.info("="*80)

    # Sequential 모드: 각 데이터셋을 순차적으로 처리
    if args.sequential and args.dataset and len(args.dataset) > 1:
        logger.info("📦 Sequential mode: Processing datasets one by one")
        train_sequential(args)
        return

    # 1. 모델 및 토크나이저 로드
    model, tokenizer = setup_model_and_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_config=args.model_config,
        pretrained_model=args.pretrained_model,
        use_flash_attention=args.flash_attention,
        use_compile=args.compile,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
    )

    # 2. 데이터셋 로드
    if args.mode == "pretrain":
        dataset, text_column = load_pretrain_dataset(
            dataset_names=args.dataset,  # 여러 데이터셋 지원
            dataset_config=args.dataset_config,
            train_files=args.train_file,  # 여러 파일 지원
            text_column=args.text_column,
        )
    else:  # sft
        dataset, text_column = load_sft_dataset(
            dataset_names=args.dataset,  # 여러 데이터셋 지원
            train_files=args.train_file,  # 여러 파일 지원
        )

    # 3. 토큰화
    logger.info("🔤 Tokenizing dataset...")

    # Packing 모드: 시퀀스 연결 방식 사용 (pretrain/sft 둘 다 지원)
    if args.packing:
        logger.info(f"📦 Using sequence concatenation (packing mode) for {args.mode}")
        
        # 배치 토큰화 (빠름)
        def batch_tokenize(examples):
            return tokenizer(
                examples[text_column],
                truncation=False,  # 연결할 것이므로 truncation 안함
                padding=False,
                add_special_tokens=True,
            )
        
        logger.info("  Batch tokenizing...")
        # 버퍼 공간 부족 방지를 위해 num_proc 제한
        safe_num_proc = min(args.num_proc, 4)  # 최대 4개 프로세스로 제한
        tokenized_ds = dataset["train"].map(
            batch_tokenize,
            batched=True,
            batch_size=1000,
            num_proc=safe_num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,  # Use cache to avoid re-tokenizing
            desc="Tokenizing",
        )
        
        # input_ids 리스트로 변환
        tokenized_list = [{"input_ids": ids} for ids in tokenized_ds["input_ids"]]
        del tokenized_ds
        
        # 시퀀스 연결 및 청킹
        concatenated_chunks = concatenate_sequences(
            tokenized_sequences=tokenized_list,
            max_seq_length=args.max_seq_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        del tokenized_list
        
        # Dataset으로 변환
        from datasets import Dataset as HFDataset
        tokenized_dataset = HFDataset.from_list(concatenated_chunks)
        
        del concatenated_chunks
        
    else:
        # 기존 방식: 개별 샘플 토큰화 with truncation
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
                return_special_tokens_mask=True,
            )

        # 버퍼 공간 부족 방지를 위해 num_proc 제한
        safe_num_proc = min(args.num_proc, 4)  # 최대 4개 프로세스로 제한
        tokenized_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            num_proc=safe_num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,  # Use cache to avoid re-tokenizing
            desc="Tokenizing",
        )

    logger.info(f"✓ Tokenized {len(tokenized_dataset)} samples")

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
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        # 추가 최적화 옵션
        dataloader_pin_memory=True,  # GPU 전송 속도 향상
        dataloader_prefetch_factor=4,  # 미리 배치 로드 (증가)
        dataloader_drop_last=True,  # 불완전 배치 제거 (속도↑)
        optim="adamw_torch_fused",  # Fused Adam (faster than 8-bit)
        ddp_find_unused_parameters=False,  # DDP 최적화
        tf32=True,  # TF32 사용 (Ampere GPU)
        group_by_length=False,  # 길이별 그룹핑 비활성화 (packing 사용시)
        max_grad_norm=1.0,  # 그래디언트 클리핑
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 최신 방식
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
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
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from: {args.resume_from_checkpoint}")
    logger.info("="*80)

    logger.info("🏃 Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. 모델 저장
    logger.info("💾 Saving model...")
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    # 모델 dtype 확인 및 bf16/fp16으로 저장
    model_dtype = next(model.parameters()).dtype
    if model_dtype in (torch.bfloat16, torch.float16):
        logger.info(f"💾 Saving model in {model_dtype} format...")
        model.save_pretrained(str(final_path), torch_dtype=model_dtype, safe_serialization=True)

    logger.info("="*80)
    logger.info(f"✅ Training completed!")
    logger.info(f"📁 Model saved to: {final_path}")
    logger.info("="*80)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MOAI-LLM Training")

    # 모드
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "sft"],
        help="Training mode: pretrain or sft"
    )

    # 데이터 (여러 파일/데이터셋 지원)
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',  # 여러 데이터셋 지원
        help="HuggingFace dataset name(s). Multiple datasets can be specified."
    )
    parser.add_argument("--dataset_config", type=str, help="Dataset config/subset (for single dataset)")
    parser.add_argument(
        "--train_file",
        type=str,
        nargs='+',  # 여러 파일 지원
        help="Local train file(s) (txt or json). Multiple files can be specified."
    )
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")

    # 모델
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizers/",
        help="Tokenizer path"
    )
    parser.add_argument("--model_config", type=str, help="Model config JSON file")
    parser.add_argument("--pretrained_model", type=str, help="Pretrained model path (for SFT)")

    # 학습 설정
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
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
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for tokenization")
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

    args = parser.parse_args()

    # 검증
    if not args.dataset and not args.train_file:
        parser.error("Either --dataset or --train_file must be provided")

    # 학습 시작
    train(args)


if __name__ == "__main__":
    main()
