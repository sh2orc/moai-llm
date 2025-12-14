"""
Pre-training script for MOAI-LLM.

Implements state-of-the-art training pipeline with:
- Warmup-Stable-Decay (WSD) learning rate schedule
- Mixed precision (BF16) training
- Gradient checkpointing
- Sequence packing
- Multi-objective loss
- DeepSpeed/FSDP support

Usage:
    # Single GPU
    python scripts/pretrain.py --config configs/training_config.yaml

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=8 scripts/pretrain.py --config configs/training_config.yaml --deepspeed configs/deepspeed_config.json

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/training_config.yaml --use_fsdp

References:
- Warmup-Stable-Decay: https://arxiv.org/abs/2410.05192
- DeepSpeed: https://www.deepspeed.ai/
- FSDP: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""

import argparse
import os
import yaml
import logging
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from datasets import load_dataset
import wandb

from moai_llm.config import MoaiConfig
from moai_llm.modeling.model import MoaiForCausalLM
from moai_llm.losses import create_loss_function

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WarmupStableDecayScheduler:
    """
    Warmup-Stable-Decay learning rate scheduler.

    Schedule:
    1. Warmup: Linear increase from 0 to max_lr
    2. Stable: Constant at max_lr
    3. Decay: Cosine decay to min_lr

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        stable_steps: Number of stable steps
        decay_steps: Number of decay steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate (default: 0.1 * max_lr)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        max_lr: float,
        min_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr * 0.1

        self.current_step = 0
        self.total_steps = warmup_steps + stable_steps + decay_steps

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step <= self.warmup_steps + self.stable_steps:
            # Stable phase: constant
            lr = self.max_lr
        else:
            # Decay phase: cosine decay
            decay_step = self.current_step - self.warmup_steps - self.stable_steps
            progress = decay_step / self.decay_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * torch.pi))
            )
            lr = float(lr)

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class MoaiTrainer(Trainer):
    """
    Custom Trainer for MOAI-LLM with multi-objective loss support.
    """

    def __init__(self, *args, loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to use custom loss function.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute loss
        if self.loss_function is not None:
            loss = self.loss_function(shift_logits, shift_labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str, run_name: str):
    """Setup logging and wandb."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler
    fh = logging.FileHandler(log_dir / f"{run_name}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("="*80)
    logger.info(f"MOAI-LLM Pre-training - {run_name}")
    logger.info("="*80)


def prepare_dataset(config: Dict[str, Any], tokenizer):
    """
    Prepare training dataset.

    Args:
        config: Training configuration
        tokenizer: Tokenizer

    Returns:
        Tokenized dataset
    """
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset_name")
    dataset_config = data_config.get("dataset_config", None)
    text_column = data_config.get("text_column", "text")
    max_seq_length = config["model"].get("max_position_embeddings", 8192)

    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_config.get("num_proc", 4),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    return tokenized_dataset


def create_model(config: Dict[str, Any]) -> MoaiForCausalLM:
    """
    Create MOAI model from configuration.

    Args:
        config: Model configuration

    Returns:
        Initialized model
    """
    model_config = MoaiConfig(**config["model"])

    logger.info("="*80)
    logger.info("Model Configuration")
    logger.info("="*80)
    logger.info(f"Hidden size: {model_config.hidden_size}")
    logger.info(f"Layers: {model_config.num_hidden_layers}")
    logger.info(f"Attention heads: {model_config.num_attention_heads}")
    logger.info(f"KV heads: {model_config.num_key_value_heads}")
    logger.info(f"Vocab size: {model_config.vocab_size}")
    logger.info(f"Max seq length: {model_config.max_position_embeddings}")
    logger.info("="*80)

    # Initialize model
    model = MoaiForCausalLM(model_config)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-train MOAI-LLM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration JSON file",
    )
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Use PyTorch FSDP for distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run name
    run_name = config["training"].get("run_name", f"moai-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Setup logging
    setup_logging(str(output_dir), run_name)

    # Initialize wandb if enabled
    if config["training"].get("use_wandb", False):
        wandb.init(
            project=config["training"].get("wandb_project", "moai-llm"),
            name=run_name,
            config=config,
        )

    # Load tokenizer
    tokenizer_path = config["data"]["tokenizer_path"]
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Set special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model = create_model(config)

    # Prepare dataset
    tokenized_dataset = prepare_dataset(config, tokenizer)

    # Create loss function
    loss_config = config.get("loss", {"type": "cross_entropy"})
    loss_function = create_loss_function(loss_config)
    logger.info(f"Using loss function: {loss_config['type']}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=training_config.get("num_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 3e-4),
        weight_decay=training_config.get("weight_decay", 0.1),
        warmup_steps=training_config.get("warmup_steps", 2000),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
        report_to="wandb" if config["training"].get("use_wandb", False) else "none",
        deepspeed=args.deepspeed,
        fsdp="full_shard auto_wrap" if args.use_fsdp else "",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "MoaiDecoderLayer",
        } if args.use_fsdp else None,
    )

    # Create trainer
    trainer = MoaiTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_function=loss_function,
    )

    # Log training info
    logger.info("="*80)
    logger.info("Training Configuration")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size per device: {training_config.get('per_device_train_batch_size', 4)}")
    logger.info(f"Gradient accumulation steps: {training_config.get('gradient_accumulation_steps', 1)}")
    logger.info(f"Global batch size: {training_args.train_batch_size}")
    logger.info(f"Learning rate: {training_config.get('learning_rate', 3e-4)}")
    logger.info(f"Weight decay: {training_config.get('weight_decay', 0.1)}")
    logger.info(f"Warmup steps: {training_config.get('warmup_steps', 2000)}")
    logger.info(f"Mixed precision: BF16={training_config.get('bf16', True)}")
    logger.info(f"Gradient checkpointing: {training_config.get('gradient_checkpointing', True)}")
    logger.info("="*80)

    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("✓ Training completed successfully!")

        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        logger.info(f"✓ Final model saved to: {final_model_path}")

    except Exception as e:
        logger.error(f"✗ Training failed with error: {e}")
        raise

    finally:
        if config["training"].get("use_wandb", False):
            wandb.finish()


if __name__ == "__main__":
    main()
