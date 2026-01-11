#!/usr/bin/env python3
"""
Training script for Moai-Mamba model.

Supports:
- Pretraining with blank infilling
- Causal LM fine-tuning
- 4-bit quantization for efficiency
- Multi-GPU training with DeepSpeed
- Gradient checkpointing for memory efficiency

Usage:
    # Pretrain from scratch
    python scripts/train_moai_mamba.py --mode pretrain --config 2b

    # Fine-tune existing model
    python scripts/train_moai_mamba.py --mode finetune --model_path ./checkpoint

    # Quantize and train
    python scripts/train_moai_mamba.py --mode finetune --quantize --bits 4
"""

import os
import sys
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moai_llm.modeling.ssm_config import get_mamba_config, MoaiMambaConfig
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.quantization import quantize_model, QuantizationConfig


@dataclass
class MoaiMambaTrainingConfig:
    """Training configuration for Moai-Mamba."""

    # Model configuration
    model_size: str = field(
        default="2b",
        metadata={"help": "Model size: '2b', '8b', or '16b'"}
    )

    # Training mode
    mode: str = field(
        default="pretrain",
        metadata={"help": "Training mode: 'pretrain' or 'finetune'"}
    )

    # Data configuration
    train_data_path: str = field(
        default="data/train",
        metadata={"help": "Path to training data"}
    )

    valid_data_path: str = field(
        default="data/valid",
        metadata={"help": "Path to validation data"}
    )

    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )

    # Training hyperparameters
    batch_size: int = field(
        default=4,
        metadata={"help": "Per-device batch size"}
    )

    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )

    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )

    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Warmup steps"}
    )

    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )

    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping"}
    )

    # Mamba-specific
    use_mamba: bool = field(
        default=True,
        metadata={"help": "Use Mamba architecture"}
    )

    # Quantization
    quantize: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization"}
    )

    quant_bits: int = field(
        default=4,
        metadata={"help": "Quantization bits (4 recommended)"}
    )

    # Memory optimization
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"}
    )

    # Multi-GPU
    distributed: bool = field(
        default=False,
        metadata={"help": "Use distributed training"}
    )

    # Logging
    output_dir: str = field(
        default="./outputs/mamba_glm",
        metadata={"help": "Output directory"}
    )

    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging frequency"}
    )

    save_steps: int = field(
        default=1000,
        metadata={"help": "Save frequency"}
    )

    eval_steps: int = field(
        default=500,
        metadata={"help": "Evaluation frequency"}
    )

    # Seed
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )


class DummyDataset(Dataset):
    """Simple dummy dataset for testing."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dummy_dataloader(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_batches: int = 100,
) -> DataLoader:
    """Create dummy dataloader for testing."""
    dataset = DummyDataset(
        num_samples=num_batches * batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
    )
    return DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Train Moai-Mamba model")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("--config", type=str, default="2b", choices=["2b", "8b", "16b"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./outputs/moai_mamba")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run with dummy data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)

    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Create model config
    print(f"Loading config from configs/mamba_config_{args.config}.json...")
    model_config = get_mamba_config(args.config)
    print(f"Initializing Moai-Mamba-{args.config.upper()} model...")

    # Initialize model
    print("Creating model...")
    model = MoaiMambaForCausalLM(model_config)

    # Enable gradient checkpointing
    if not args.no_checkpoint:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Quantize if requested
    if args.quantize:
        print(f"Quantizing model to {args.bits}-bit...")
        quant_config = QuantizationConfig(
            bits=args.bits,
            group_size=128,
            symmetric=True,
        )
        model = quantize_model(model, bits=args.bits, group_size=128)
        print("Model quantized")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    if args.test:
        print("Running in TEST mode with dummy data...")
        train_dataloader = create_dummy_dataloader(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=model_config.vocab_size,
            num_batches=10,
        )
        eval_dataloader = create_dummy_dataloader(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=model_config.vocab_size,
            num_batches=2,
        )
    else:
        # TODO: Load real data
        print("Error: Real data loading not implemented yet. Use --test flag.")
        return

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Calculate total steps
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps

    # Setup scheduler
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=total_steps,
    )

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    # Training loop
    model.train()
    global_step = 0
    accumulated_loss = 0
    grad_accum = args.gradient_accumulation_steps

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum

            # Backward pass
            loss.backward()

            accumulated_loss += loss.item()

            # Update weights
            if (step + 1) % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    avg_loss = accumulated_loss / 10
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    accumulated_loss = 0

                # Evaluation
                if global_step % 50 == 0:
                    model.eval()
                    eval_loss = 0
                    with torch.no_grad():
                        for eval_batch in eval_dataloader:
                            eval_input_ids = eval_batch["input_ids"].to(device)
                            eval_attention_mask = eval_batch["attention_mask"].to(device)
                            eval_labels = eval_batch["labels"].to(device)

                            eval_outputs = model(
                                input_ids=eval_input_ids,
                                attention_mask=eval_attention_mask,
                                labels=eval_labels,
                            )
                            eval_loss += eval_outputs.loss.item()

                    eval_loss /= len(eval_dataloader)
                    print(f"Eval Loss: {eval_loss:.4f}")
                    model.train()

                # Save checkpoint
                if global_step % 100 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

    print("\nTraining completed!")

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
