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
import json

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
from moai_llm.data import SequencePacker

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
    # 모든 시퀀스를 하나로 연결 (각 끝에 EOS 추가)
    all_tokens = []
    
    for seq in tokenized_sequences:
        input_ids = seq["input_ids"]
        
        # 이미 EOS로 끝나지 않는 경우에만 EOS 추가
        if len(input_ids) > 0 and input_ids[-1] != eos_token_id:
            input_ids = input_ids + [eos_token_id]
        
        all_tokens.extend(input_ids)
    
    logger.info(f"📦 Concatenating {len(tokenized_sequences)} sequences into {len(all_tokens):,} tokens")
    
    # max_seq_length 청크로 분할
    chunks = []
    for i in range(0, len(all_tokens), max_seq_length):
        chunk = all_tokens[i:i + max_seq_length]
        
        # 마지막 청크가 너무 짧으면 (< 128) 버림
        if len(chunk) < 128:
            logger.info(f"  Dropping short final chunk of {len(chunk)} tokens")
            continue
            
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
        })
    
    logger.info(f"✓ Created {len(chunks)} chunks of max {max_seq_length} tokens each")
    
    return chunks


# ============================================================================
# 데이터셋 로드 및 변환
# ============================================================================

def _load_single_file(file_path: str) -> list:
    """단일 파일을 로드하여 텍스트 리스트로 반환"""
    formatted_data = []
    
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        
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
    """단일 HuggingFace 데이터셋을 로드하여 텍스트 리스트로 반환"""
    logger.info(f"  Loading HuggingFace: {dataset_name}")
    
    if dataset_config:
        logger.info(f"    Config: {dataset_config}")
        raw_dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
    else:
        raw_dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    formatted_data = []
    
    # train split 사용
    train_data = raw_dataset.get("train", raw_dataset)
    if hasattr(train_data, "__iter__"):
        for item in train_data:
            # dict가 아닌 경우 (예: IterableDataset)
            if not isinstance(item, dict):
                continue
            
            text = _convert_to_text(item)
            if text:
                formatted_data.append({"text": text})
    
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
):
    """모델과 토크나이저 초기화"""

    # 토크나이저
    logger.info(f"📝 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델
    if pretrained_model:
        logger.info(f"🔄 Loading pretrained model: {pretrained_model}")
        model = MoaiForCausalLM.from_pretrained(pretrained_model)
    else:
        logger.info("🆕 Creating new model from config")
        if model_config:
            config = MoaiConfig.from_json_file(model_config)
        else:
            config = MoaiConfig()
        model = MoaiForCausalLM(config)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    return model, tokenizer


# ============================================================================
# 학습
# ============================================================================

def train(args):
    """메인 학습 함수"""

    logger.info("="*80)
    logger.info(f"🚀 Starting {args.mode.upper()} training")
    logger.info("="*80)

    # 1. 모델 및 토크나이저 로드
    model, tokenizer = setup_model_and_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_config=args.model_config,
        pretrained_model=args.pretrained_model,
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
        
        # 각 샘플 토큰화 (truncation 없이)
        tokenized_list = []
        for i, text in enumerate(dataset["train"][text_column]):
            tokens = tokenizer(
                text,
                truncation=False,  # 연결할 것이므로 truncation 안함
                padding=False,
                add_special_tokens=True,
            )
            tokenized_list.append(tokens)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Tokenized {i + 1:,} / {len(dataset['train']):,} samples...")
        
        # 시퀀스 연결 및 청킹
        concatenated_chunks = concatenate_sequences(
            tokenized_sequences=tokenized_list,
            max_seq_length=args.max_seq_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Dataset으로 변환
        from datasets import Dataset as HFDataset
        tokenized_dataset = HFDataset.from_list(concatenated_chunks)
        
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
            num_proc=args.num_proc,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )

    logger.info(f"✓ Tokenized {len(tokenized_dataset)} samples")

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # 5. Training Arguments
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
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    args = parser.parse_args()

    # 검증
    if not args.dataset and not args.train_file:
        parser.error("Either --dataset or --train_file must be provided")

    # 학습 시작
    train(args)


if __name__ == "__main__":
    main()
