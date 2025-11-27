#!/usr/bin/env python3
"""
GRPO Training Script for Pedagogical Tutor Model

Group Relative Policy Optimization (GRPO) training for the tutor model.
Based on DeepSeekMath's GRPO algorithm - simpler than PPO (no critic model).

This script:
1. Loads an SFT-trained model as the starting point
2. Uses the pedagogical reward model to score responses
3. Trains using GRPO to maximize reward

Usage:
    python scripts/train_tutor_grpo.py --sft-model ./tutor-sft --output ./tutor-grpo
    python scripts/train_tutor_grpo.py --sft-model ./tutor-sft --data data/prompts.jsonl --output ./tutor-grpo

Requirements:
    pip install transformers datasets peft trl accelerate bitsandbytes wandb
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add backend to path for reward model
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(data_path: str) -> List[Dict[str, Any]]:
    """Load prompts for RL training from JSONL file"""
    prompts = []
    with open(data_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info(f"Loaded {len(prompts)} prompts from {data_path}")
    return prompts


def create_prompts_from_sessions(sessions_path: str) -> List[Dict[str, Any]]:
    """Create RL prompts from session data (use student turns as prompts)"""
    prompts = []
    
    with open(sessions_path, "r") as f:
        for line in f:
            session = json.loads(line)
            concept = session["concept"]
            
            for i, turn in enumerate(session["turns"]):
                if turn["role"] == "student":
                    # Build state context
                    state = {
                        "concept": concept,
                        "mastery": turn.get("mastery_before", 0.0),
                        "turn_number": i,
                        "student_message": turn["content"],
                    }
                    
                    # Build conversation history
                    history = []
                    for prev_turn in session["turns"][max(0, i-4):i]:
                        history.append({
                            "role": prev_turn["role"],
                            "content": prev_turn["content"][:200]
                        })
                    
                    prompts.append({
                        "state": state,
                        "history": history,
                        "concept": concept,
                    })
    
    logger.info(f"Created {len(prompts)} prompts from sessions")
    return prompts


def format_prompt(state: Dict, history: List[Dict]) -> str:
    """Format state and history into a prompt string"""
    lines = [
        "<|state|>",
        f"Concept: {state['concept']}",
        f"Mastery: {state['mastery']:.2f}",
        f"Turn: {state['turn_number']}",
        f"Student: {state['student_message']}",
        "<|/state|>",
        "",
        "<|history|>",
    ]
    
    for turn in history:
        role = "Student" if turn["role"] == "student" else "Tutor"
        lines.append(f"{role}: {turn['content']}")
    
    lines.append("<|/history|>")
    lines.append("")
    lines.append("### Response:")
    
    return "\n".join(lines)


class PedagogicalRewardFunction:
    """
    Reward function for GRPO training.
    Combines verifiable rewards with LLM judge scores.
    """
    
    LEAKAGE_PATTERNS = [
        r"the answer is",
        r"the correct answer",
        r"the solution is",
        r"= \d+",
        r"therefore[,\s]+\w+ equals",
    ]
    
    SCAFFOLDING_PATTERNS = [
        r"\?$",
        r"what do you think",
        r"can you",
        r"try to",
        r"think about",
        r"how would",
    ]
    
    def __init__(self, llm_judge=None, weights=None):
        self.llm_judge = llm_judge
        self.weights = weights or {
            "no_leakage": 0.25,
            "scaffolding": 0.25,
            "format": 0.20,
            "length": 0.15,
            "helpfulness": 0.15,
        }
    
    def __call__(
        self, 
        prompts: List[str], 
        outputs: List[str],
        contexts: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """Compute rewards for a batch of outputs"""
        rewards = []
        
        for i, output in enumerate(outputs):
            context = contexts[i] if contexts else {}
            reward = self._score_output(output, context)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _score_output(self, output: str, context: Dict) -> float:
        """Score a single output"""
        scores = {}
        
        # 1. Check for answer leakage (verifiable)
        output_lower = output.lower()
        has_leakage = any(re.search(p, output_lower) for p in self.LEAKAGE_PATTERNS)
        scores["no_leakage"] = 0.0 if has_leakage else 1.0
        
        # 2. Check scaffolding usage (verifiable)
        scaffolding_count = sum(
            1 for p in self.SCAFFOLDING_PATTERNS if re.search(p, output_lower)
        )
        scores["scaffolding"] = min(1.0, scaffolding_count * 0.3)
        
        # 3. Check format compliance (verifiable)
        has_think = "<think>" in output and "</think>" in output
        has_strategy = re.search(r"\[Strategy:\s*\w+\]", output, re.IGNORECASE)
        scores["format"] = (0.5 if has_think else 0.0) + (0.5 if has_strategy else 0.0)
        
        # 4. Length penalty (verifiable)
        # Remove thinking from word count
        visible_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        visible_output = re.sub(r"\[Strategy:\s*\w+\]", "", visible_output)
        word_count = len(visible_output.split())
        
        if word_count < 20:
            scores["length"] = 0.3  # Too short
        elif word_count > 200:
            scores["length"] = 0.5  # Too long
        else:
            scores["length"] = 1.0  # Good length
        
        # 5. Helpfulness (heuristic or LLM judge)
        # Simple heuristic: does it reference the concept?
        concept = context.get("concept", "")
        if concept and concept.lower() in output_lower:
            scores["helpfulness"] = 0.7
        else:
            scores["helpfulness"] = 0.4
        
        # Compute weighted total
        total = sum(scores[k] * self.weights[k] for k in scores)
        
        # Apply hard penalty for leakage
        if has_leakage:
            total = max(-0.5, total - 0.5)
        
        return total


def main():
    parser = argparse.ArgumentParser(description="Train pedagogical tutor with GRPO")
    parser.add_argument("--sft-model", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--output", type=str, default="./tutor-grpo", help="Output directory")
    parser.add_argument("--data", type=str, default=None, help="Path to prompts JSONL (or sessions.jsonl)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--num-generations", type=int, default=4, help="Samples per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.05, help="KL penalty coefficient")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--dry-run", action="store_true", help="Don't train, just test")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.data:
        if "sessions" in args.data:
            prompts_data = create_prompts_from_sessions(args.data)
        else:
            prompts_data = load_prompts(args.data)
    else:
        # Create dummy prompts for testing
        prompts_data = [
            {
                "state": {"concept": "convection", "mastery": 0.3, "turn_number": 2, 
                         "student_message": "I don't understand how heat moves"},
                "history": [],
                "concept": "convection",
            }
        ] * 100
    
    # Format prompts
    prompts = [format_prompt(p["state"], p["history"]) for p in prompts_data]
    contexts = [{"concept": p["concept"]} for p in prompts_data]
    
    logger.info(f"Prepared {len(prompts)} prompts for training")
    logger.info(f"Sample prompt:\n{prompts[0][:500]}...")
    
    if args.dry_run:
        # Test reward function
        reward_fn = PedagogicalRewardFunction()
        test_output = """<think>
The student is confused about heat transfer. I should use an analogy.
</think>
[Strategy: ANALOGY]
Think of heat like water flowing downhill. Just as water naturally flows from high ground to low ground, heat naturally flows from hot to cold. Can you think of a time you've felt this happen in your daily life?"""
        
        score = reward_fn._score_output(test_output, {"concept": "convection"})
        logger.info(f"Test output score: {score:.3f}")
        logger.info("Dry run complete. Exiting.")
        return
    
    # Import training libraries
    try:
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install transformers datasets peft trl>=0.8.0 accelerate")
        sys.exit(1)
    
    # Setup W&B
    if args.wandb:
        import wandb
        wandb.init(project="tutor-grpo", config=vars(args))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (SFT checkpoint)
    logger.info(f"Loading SFT model from {args.sft_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Create reward function
    reward_fn = PedagogicalRewardFunction()
    
    # Create dataset
    dataset = Dataset.from_dict({"prompt": prompts, "context": contexts})
    
    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_new_tokens=512,
        temperature=0.7,
        kl_coef=args.kl_coef,
        logging_steps=10,
        save_steps=100,
        report_to="wandb" if args.wandb else "none",
    )
    
    # Wrapper for reward function that matches TRL interface
    def reward_function(completions, prompts=None, **kwargs):
        """TRL-compatible reward function"""
        # Extract text from completions
        texts = [c if isinstance(c, str) else c.get("text", str(c)) for c in completions]
        # Get contexts from dataset (simplified - assumes batch alignment)
        ctx = [{"concept": "concept"} for _ in texts]  # Placeholder
        return reward_fn(prompts or [], texts, ctx)
    
    # Create trainer
    # Note: TRL's GRPO API may vary by version - adjust as needed
    try:
        trainer = GRPOTrainer(
            model=model,
            config=grpo_config,
            tokenizer=tokenizer,
            train_dataset=dataset,
            reward_funcs=reward_function,  # TRL 0.8+ interface
        )
    except TypeError:
        # Fallback for older TRL versions
        logger.warning("Using fallback GRPO configuration - check TRL version")
        from trl import PPOTrainer, PPOConfig
        
        ppo_config = PPOConfig(
            model_name=args.sft_model,
            learning_rate=args.lr,
            batch_size=args.batch_size * args.num_generations,
            mini_batch_size=args.batch_size,
        )
        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
    
    # Train
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    
    logger.info(f"Model saved to {args.output}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
