"""
MOAI-LLM 학습 스크립트 (통합 버전)

사전학습과 SFT를 하나의 스크립트로 처리합니다.

사용법:
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 데이터셋 로드 및 변환
# ============================================================================

def load_pretrain_dataset(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    train_file: Optional[str] = None,
    text_column: str = "text",
):
    """
    사전학습용 데이터셋 로드

    Args:
        dataset_name: HuggingFace 데이터셋 이름 (예: "wikipedia")
        dataset_config: 데이터셋 설정 (예: "20220301.en")
        train_file: 로컬 txt 파일 경로
        text_column: 텍스트 컬럼 이름
    """
    logger.info("📚 Loading pretrain dataset...")

    if train_file:
        # 로컬 txt 파일
        logger.info(f"  From local file: {train_file}")
        dataset = load_dataset("text", data_files={"train": train_file})
        text_column = "text"
    elif dataset_name:
        # HuggingFace 데이터셋
        logger.info(f"  From HuggingFace: {dataset_name}")
        if dataset_config:
            logger.info(f"  Config: {dataset_config}")
            dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

    logger.info(f"✓ Dataset loaded: {len(dataset['train'])} samples")
    return dataset, text_column


def load_sft_dataset(
    dataset_name: Optional[str] = None,
    train_file: Optional[str] = None,
):
    """
    SFT용 데이터셋 로드 및 포맷 변환

    지원 포맷:
    - Alpaca: {"instruction": "...", "output": "..."}
    - Chat: {"messages": [{"role": "user", "content": "..."}]}
    - ShareGPT: {"conversations": [{"from": "human", "value": "..."}]}
    """
    logger.info("📚 Loading SFT dataset...")

    if train_file:
        # 로컬 JSON 파일
        logger.info(f"  From local file: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 포맷 자동 감지 및 변환
        formatted_data = []

        for item in data:
            # input/output 포맷 (BCCard 등)
            if "input" in item and "output" in item:
                text = f"<|im_start|>user\n{item['input']}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                formatted_data.append({"text": text})

            # Alpaca 포맷
            elif "instruction" in item and "output" in item:
                text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                formatted_data.append({"text": text})

            # Chat 포맷
            elif "messages" in item:
                text = ""
                for msg in item["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                formatted_data.append({"text": text})

            # ShareGPT 포맷
            elif "conversations" in item:
                text = ""
                for conv in item["conversations"]:
                    role = "user" if conv["from"] == "human" else "assistant"
                    text += f"<|im_start|>{role}\n{conv['value']}<|im_end|>\n"
                formatted_data.append({"text": text})

        # Dataset으로 변환
        from datasets import Dataset
        dataset = {"train": Dataset.from_list(formatted_data)}

    elif dataset_name:
        # HuggingFace 데이터셋 (자동 변환)
        logger.info(f"  From HuggingFace: {dataset_name}")
        raw_dataset = load_dataset(dataset_name, trust_remote_code=True)

        # 포맷 변환
        formatted_data = []
        for item in raw_dataset["train"]:
            # input/output 포맷
            if "input" in item and "output" in item:
                text = f"<|im_start|>user\n{item['input']}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                formatted_data.append({"text": text})

            # instruction/output 포맷
            elif "instruction" in item and "output" in item:
                text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
                text += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
                formatted_data.append({"text": text})

            # messages 포맷
            elif "messages" in item:
                text = ""
                for msg in item["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                formatted_data.append({"text": text})

            # conversations 포맷
            elif "conversations" in item:
                text = ""
                for conv in item["conversations"]:
                    role = "user" if conv["from"] == "human" else "assistant"
                    text += f"<|im_start|>{role}\n{conv['value']}<|im_end|>\n"
                formatted_data.append({"text": text})

        # Dataset으로 변환
        from datasets import Dataset
        dataset = {"train": Dataset.from_list(formatted_data)}
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

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
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            train_file=args.train_file,
            text_column=args.text_column,
        )
    else:  # sft
        dataset, text_column = load_sft_dataset(
            dataset_name=args.dataset,
            train_file=args.train_file,
        )

    # 3. 토큰화
    logger.info("🔤 Tokenizing dataset...")

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
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max steps: {args.max_steps if args.max_steps > 0 else 'Full epoch'}")
    logger.info("="*80)

    logger.info("🏃 Starting training...")
    trainer.train()

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

    # 데이터
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, help="Dataset config/subset")
    parser.add_argument("--train_file", type=str, help="Local train file (txt or json)")
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

    # 로깅
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)

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
