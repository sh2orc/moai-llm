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
from datasets import load_dataset

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


def _load_hf_dataset(dataset_name: str, dataset_config: Optional[str] = None) -> list:
    """
    단일 HuggingFace 데이터셋을 로드하여 텍스트 리스트로 반환
    
    데이터셋 이름에 config를 포함할 수 있음:
        - "dataset_name:config_name" 형식 지원
        - 예: "maywell/korean_textbooks:claude_evol"
    """
    # dataset_name:config_name 형식 파싱
    if ":" in dataset_name:
        dataset_name, config_from_name = dataset_name.split(":", 1)
        if not dataset_config:
            dataset_config = config_from_name
    
    logger.info(f"  Loading HuggingFace: {dataset_name}")
    
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
    
    # 배치 처리로 변환 (빠름)
    converted = train_data.map(
        convert_batch,
        batched=True,
        batch_size=5000,  # Increased for less overhead
        num_proc=min(16, os.cpu_count() or 4),
        remove_columns=train_data.column_names,
        load_from_cache_file=False,
        desc=f"Converting {dataset_name}",
    )
    
    # 빈 텍스트 필터링
    converted = converted.filter(lambda x: len(x["text"]) > 0, num_proc=4)
    
    # 리스트로 변환
    formatted_data = [{"text": t} for t in converted["text"]]
    
    logger.info(f"    → {len(formatted_data):,} samples")
    return formatted_data


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
        
        for i, ds_name in enumerate(dataset_names):
            # 첫 번째 데이터셋에만 config 적용
            config = dataset_config if i == 0 else None
            ds_data = _load_hf_dataset(ds_name, config)
            all_data.extend(ds_data)
    
    if not all_data:
        raise ValueError("Either dataset_names or train_files must be provided")
    
    logger.info(f"  Total: {len(all_data):,} samples")
    
    from datasets import Dataset
    dataset = {"train": Dataset.from_list(all_data)}
    text_column = "text"

    logger.info(f"✓ Dataset loaded: {len(dataset['train'])} samples")
    return dataset, text_column


def _convert_to_text(item: dict) -> Optional[str]:
    """
    다양한 데이터 형식을 텍스트로 변환 (pretrain용)
    
    지원 형식:
        - {"text": "..."}: 그대로 사용
        - {"input": "...", "output": "..."}: Chat 형식으로 변환
        - {"instruction": "...", "output": "..."}: Chat 형식으로 변환
        - {"messages": [...]}: Chat 형식으로 변환
        - {"conversations": [...]}: Chat 형식으로 변환
    """
    # text 필드가 있으면 그대로 사용
    if "text" in item:
        return item["text"]
    
    # input/output 포맷
    if "input" in item and "output" in item:
        text = f"<|im_start|>user\n{item['input']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        return text
    
    # instruction/output 포맷 (Alpaca)
    if "instruction" in item and "output" in item:
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        return text
    
    # messages 포맷 (OpenAI Chat)
    if "messages" in item:
        text = ""
        for msg in item["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text.strip()
    
    # conversations 포맷 (ShareGPT)
    if "conversations" in item:
        text = ""
        for conv in item["conversations"]:
            role = "user" if conv.get("from") == "human" else "assistant"
            value = conv.get("value", "")
            text += f"<|im_start|>{role}\n{value}<|im_end|>\n"
        return text.strip()
    
    # DeepSeek R1 스타일 (input/content/reasoning_content)
    if "input" in item and "content" in item:
        text = f"<|im_start|>user\n{item['input']}<|im_end|>\n"
        # reasoning_content가 있으면 먼저 추가
        if item.get("reasoning_content"):
            text += f"<|im_start|>assistant\n<think>\n{item['reasoning_content']}\n</think>\n{item['content']}<|im_end|>"
        else:
            text += f"<|im_start|>assistant\n{item['content']}<|im_end|>"
        return text
    
    # prompt/response 포맷
    if "prompt" in item and "response" in item:
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['response']}<|im_end|>"
        return text
    
    # question/answer 포맷
    if "question" in item and "answer" in item:
        text = f"<|im_start|>user\n{item['question']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['answer']}<|im_end|>"
        return text
    
    # prompt/completion 포맷
    if "prompt" in item and "completion" in item:
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{item['completion']}<|im_end|>"
        return text
    
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
        model = MoaiForCausalLM.from_pretrained(pretrained_model, torch_dtype=dtype)
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
        config.torch_dtype = dtype
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
            tokenized_ds = dataset["train"].map(
                batch_tokenize,
                batched=True,
                batch_size=5000,  # Increased for less overhead
                num_proc=args.num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,  # Skip cache check
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

            tokenized_dataset = dataset["train"].map(
                tokenize_function,
                batched=True,
                batch_size=5000,
                num_proc=args.num_proc,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
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
            tokenizer=tokenizer,
        )
        
        logger.info(f"🏃 Training on dataset {idx+1}/{len(all_sources)}...")
        trainer.train()
        
        # 5. 체크포인트 저장
        checkpoint_path = f"{stage_output_dir}/checkpoint"
        trainer.save_model(checkpoint_path)
        logger.info(f"💾 Saved checkpoint to: {checkpoint_path}")
        
        # 다음 라운드를 위해 체크포인트 경로 업데이트
        current_checkpoint = checkpoint_path
        
        # 6. 메모리 해제
        del model
        del tokenizer
        del tokenized_dataset
        del trainer
        gc.collect()
        
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
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
        tokenized_ds = dataset["train"].map(
            batch_tokenize,
            batched=True,
            batch_size=5000,  # Increased for less overhead
            num_proc=args.num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,  # Skip cache check
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

        tokenized_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            batch_size=5000,
            num_proc=args.num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
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
