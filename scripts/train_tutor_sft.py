#!/usr/bin/env python3
"""
SFT Training Script for Pedagogical Tutor Model

This script fine-tunes a base LLM on expert tutor trajectories that include:
- Explicit thinking traces (<think>...</think>)
- Strategy declarations ([Strategy: X])
- Pedagogical responses

Usage:
    python scripts/train_tutor_sft.py --data data/sessions.jsonl --output ./tutor-sft
    python scripts/train_tutor_sft.py --data data/sessions.jsonl --model Qwen/Qwen2.5-3B-Instruct --output ./tutor-sft

Requirements:
    pip install transformers datasets peft trl accelerate bitsandbytes wandb
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model options (choose based on available VRAM)
MODEL_OPTIONS = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",  # Needs more VRAM
}


def load_sessions(data_path: str) -> List[Dict[str, Any]]:
    """Load session logs from JSONL file"""
    sessions = []
    with open(data_path, "r") as f:
        for line in f:
            sessions.append(json.loads(line))
    logger.info(f"Loaded {len(sessions)} sessions from {data_path}")
    return sessions


def format_turn_for_sft(
    session: Dict[str, Any],
    turn_idx: int,
    context_window: int = 4
) -> Dict[str, str] | None:
    """
    Format a single tutor turn for SFT training.
    
    Returns dict with 'input' and 'output' keys, or None if not a tutor turn.
    """
    turns = session["turns"]
    if turn_idx >= len(turns):
        return None
    
    turn = turns[turn_idx]
    if turn["role"] != "tutor":
        return None
    
    # Build context from previous turns
    context_turns = turns[max(0, turn_idx - context_window):turn_idx]
    
    # Format state
    state_lines = [
        f"<|state|>",
        f"Concept: {session['concept']}",
        f"Mastery: {turn.get('mastery_before', 0.0):.2f}",
        f"Turn: {turn_idx}",
        f"<|/state|>",
    ]
    
    # Format conversation history
    history_lines = ["<|history|>"]
    for t in context_turns:
        role = "Student" if t["role"] == "student" else "Tutor"
        history_lines.append(f"{role}: {t['content'][:200]}")
    history_lines.append("<|/history|>")
    
    # Input is state + history
    input_text = "\n".join(state_lines + history_lines)
    
    # Output is thinking + strategy + response (for tutor turns)
    output_parts = []
    if turn.get("thinking"):
        output_parts.append(f"<think>\n{turn['thinking']}\n</think>")
    if turn.get("strategy"):
        output_parts.append(f"[Strategy: {turn['strategy']}]")
    output_parts.append(turn["content"])
    
    output_text = "\n".join(output_parts)
    
    return {
        "input": input_text,
        "output": output_text,
        "concept": session["concept"],
        "mastery_delta": turn.get("mastery_delta", 0.0),
    }


def create_sft_dataset(sessions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert session logs to SFT training format"""
    examples = []
    
    for session in sessions:
        for i, turn in enumerate(session["turns"]):
            if turn["role"] == "tutor":
                example = format_turn_for_sft(session, i)
                if example:
                    examples.append(example)
    
    logger.info(f"Created {len(examples)} SFT examples")
    return examples


def formatting_prompts_func(examples):
    """Format examples for TRL SFT trainer"""
    texts = []
    for i in range(len(examples["input"])):
        text = f"### Input:\n{examples['input'][i]}\n\n### Response:\n{examples['output'][i]}"
        texts.append(text)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Train pedagogical tutor with SFT")
    parser.add_argument("--data", type=str, required=True, help="Path to sessions.jsonl")
    parser.add_argument("--output", type=str, default="./tutor-sft", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually train, just test data loading")
    
    args = parser.parse_args()
    
    # Load and prepare data
    sessions = load_sessions(args.data)
    sft_examples = create_sft_dataset(sessions)
    
    if len(sft_examples) == 0:
        logger.error("No training examples found!")
        sys.exit(1)
    
    # Show sample
    logger.info(f"Sample training example:")
    logger.info(f"Input: {sft_examples[0]['input'][:300]}...")
    logger.info(f"Output: {sft_examples[0]['output'][:300]}...")
    
    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return
    
    # Import training libraries (lazy import for faster startup)
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install transformers datasets peft trl accelerate bitsandbytes")
        sys.exit(1)
    
    # Setup W&B if requested
    if args.wandb:
        import wandb
        wandb.init(project="tutor-sft", config=vars(args))
    
    # Create dataset
    dataset = Dataset.from_list(sft_examples)
    dataset = dataset.map(
        lambda x: {"text": f"### Input:\n{x['input']}\n\n### Response:\n{x['output']}"},
        remove_columns=["input", "output", "concept", "mastery_delta"]
    )
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="wandb" if args.wandb else "none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    logger.info(f"Model saved to {args.output}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
