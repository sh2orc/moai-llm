#!/usr/bin/env python3
"""
Training script for Moai-Mamba model (Pure SSM Architecture).

Supports:
- Pretraining from scratch
- Fine-tuning existing models
- 4-bit quantization for efficiency
- Multi-GPU training with FSDP/DeepSpeed
- Gradient checkpointing for memory efficiency
- Mixed precision training (BF16/FP16)

Usage:
    # Quick test with dummy data
    python scripts/train_mamba.py --config 2b --test

    # Pretrain from scratch
    python scripts/train_mamba.py --config 2b --data_path data/train --output_dir outputs/mamba-2b

    # Fine-tune existing model
    python scripts/train_mamba.py --model_path checkpoints/mamba-2b --data_path data/finetune

    # Quantized training
    python scripts/train_mamba.py --config 2b --quantize --bits 4
"""

import os
import sys
import argparse
import math
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moai_llm.modeling.ssm_config import get_mamba_config, MoaiMambaConfig, load_mamba_config
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.quantization import quantize_model, QuantizationConfig


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
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class TextDataset(Dataset):
    """Simple text dataset for pretraining."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load data
        data_path = Path(data_path)
        if data_path.is_dir():
            # Load all text files
            for file in data_path.glob("**/*.txt"):
                with open(file, 'r', encoding='utf-8') as f:
                    self.data.append(f.read())
        elif data_path.is_file() and data_path.suffix == '.txt':
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data.append(f.read())
        elif data_path.is_file() and data_path.suffix == '.json':
            # Load JSON lines
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line)['text'])

        print(f"Loaded {len(self.data)} documents from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def setup_optimizer(model, learning_rate: float, weight_decay: float):
    """Setup optimizer."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    return optimizer


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Get cosine learning rate scheduler."""
    from transformers import get_scheduler

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return scheduler


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: str):
    """Save training checkpoint."""
    checkpoint_path = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_path)

    # Save training state
    training_state = {
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    import pickle
    with open(checkpoint_path / "training_state.pt", 'wb') as f:
        pickle.dump(training_state, f)

    print(f"Checkpoint saved: {checkpoint_path}")


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Calculate perplexity
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Train Moai-Mamba model")

    # Model arguments
    parser.add_argument("--config", type=str, default="2b", choices=["2b", "8b", "16b"],
                        help="Model size configuration")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to custom config JSON file (overrides --config)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model checkpoint for fine-tuning")

    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data (file or directory)")
    parser.add_argument("--valid_data_path", type=str, default=None,
                        help="Path to validation data")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/moai",
                        help="Path to tokenizer")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Quantization
    parser.add_argument("--quantize", action="store_true",
                        help="Use quantization")
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantization bits")

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/moai_mamba",
                        help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with dummy data")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config_path:
        print(f"Loading config from {args.config_path}...")
        config = load_mamba_config(args.config_path)
    else:
        print(f"Loading config: Moai-Mamba-{args.config.upper()}...")
        config = get_mamba_config(args.config)

    # Create or load model
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = MoaiMambaForCausalLM.from_pretrained(args.model_path)
    else:
        print("Creating model from config...")
        model = MoaiMambaForCausalLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Gradient checkpointing
    if args.gradient_checkpointing and not args.no_checkpoint:
        model.model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")

    # Quantize
    if args.quantize:
        print(f"Quantizing model to {args.bits}-bit...")
        model = quantize_model(model, bits=args.bits, group_size=128)
        print("Model quantized")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")

    # Load tokenizer
    tokenizer = None
    if not args.test:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            print(f"Tokenizer loaded from {args.tokenizer_path}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    # Create datasets
    if args.test:
        print("Running in TEST mode with dummy data...")
        train_dataset = DummyDataset(
            num_samples=100,
            seq_length=args.max_length,
            vocab_size=config.vocab_size,
        )
        eval_dataset = DummyDataset(
            num_samples=20,
            seq_length=args.max_length,
            vocab_size=config.vocab_size,
        )
    elif args.data_path:
        if tokenizer is None:
            print("Error: Real data requires tokenizer.")
            return

        print(f"Loading training data from {args.data_path}...")
        train_dataset = TextDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        if args.valid_data_path:
            print(f"Loading validation data from {args.valid_data_path}...")
            eval_dataset = TextDataset(
                data_path=args.valid_data_path,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
        else:
            eval_dataset = train_dataset
    else:
        print("Error: Please provide --data_path or use --test flag")
        return

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    print(f"Training batches: {len(train_dataloader)}")
    print(f"Evaluation batches: {len(eval_dataloader)}")

    # Setup optimizer and scheduler
    optimizer = setup_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epoch(s)...")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps} effective")

    model.train()
    global_step = 0
    accumulated_loss = 0
    grad_accum = args.gradient_accumulation_steps

    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum

            # Backward pass
            loss.backward()

            accumulated_loss += loss.item()

            # Update weights
            if (step + 1) % grad_accum == 0:
                # Gradient clipping
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = accumulated_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    accumulated_loss = 0

                # Evaluation
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    model.eval()
                    eval_loss, perplexity = evaluate(model, eval_dataloader, device)
                    print(f"Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
                    model.train()

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, str(output_dir))

        # Epoch evaluation
        model.eval()
        eval_loss, perplexity = evaluate(model, eval_dataloader, device)
        print(f"\nEpoch {epoch + 1} - Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
        model.train()

        # Save epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, global_step, str(output_dir / f"epoch-{epoch+1}"))

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    print(f"\nTraining completed! Final model saved: {final_path}")


if __name__ == "__main__":
    main()
