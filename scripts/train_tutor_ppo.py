#!/usr/bin/env python3
"""
PPO Training Script for Pedagogical Tutor Model

This script trains the Tutor MDP policy using Proximal Policy Optimization (PPO).

Training Pipeline:
1. Load SFT-trained model as initial policy
2. Load trajectory data with rewards
3. Train using PPO to optimize expected reward

Usage:
    python scripts/train_tutor_ppo.py --sft-model ./tutor-sft --data data/trajectories --output ./tutor-ppo
    python scripts/train_tutor_ppo.py --sft-model Qwen/Qwen2.5-3B-Instruct --data data/trajectories --output ./tutor-ppo

Requirements:
    pip install transformers datasets peft trl accelerate bitsandbytes wandb
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class TutorRewardFunction:
    """
    Reward function for PPO training.
    
    Computes reward based on:
    - Pedagogical quality (scaffolding, no answer leakage)
    - Action appropriateness
    - Response quality
    """
    
    # Patterns indicating answer leakage (bad)
    LEAKAGE_PATTERNS = [
        "the answer is",
        "the correct answer",
        "the solution is",
        "= ",
        "therefore it equals",
        "which gives us",
    ]
    
    # Patterns indicating scaffolding (good)
    SCAFFOLDING_PATTERNS = [
        "what do you think",
        "can you",
        "think about",
        "consider",
        "how would",
        "?",
    ]
    
    # Patterns for each action type
    ACTION_PATTERNS = {
        "socratic_question": ["?", "what", "why", "how", "can you"],
        "give_hint": ["think about", "consider", "hint", "clue"],
        "worked_example": ["for example", "step", "let's work through"],
        "use_analogy": ["like", "similar to", "imagine", "think of it as"],
        "explain": ["means", "is", "because", "refers to"],
        "correct": ["actually", "not quite", "the issue is"],
    }
    
    def __init__(
        self,
        weight_no_leakage: float = 0.3,
        weight_scaffolding: float = 0.25,
        weight_format: float = 0.2,
        weight_action_match: float = 0.15,
        weight_conciseness: float = 0.1,
    ):
        self.weights = {
            "no_leakage": weight_no_leakage,
            "scaffolding": weight_scaffolding,
            "format": weight_format,
            "action_match": weight_action_match,
            "conciseness": weight_conciseness,
        }
    
    def __call__(
        self, 
        queries: List[str], 
        responses: List[str],
        **kwargs
    ) -> List[float]:
        """Compute rewards for a batch of responses"""
        rewards = []
        for query, response in zip(queries, responses):
            reward = self._score_response(query, response)
            rewards.append(reward)
        return rewards
    
    def _score_response(self, query: str, response: str) -> float:
        """Score a single response"""
        scores = {}
        response_lower = response.lower()
        
        # 1. Check for answer leakage
        has_leakage = any(p in response_lower for p in self.LEAKAGE_PATTERNS)
        scores["no_leakage"] = 0.0 if has_leakage else 1.0
        
        # 2. Check scaffolding usage
        scaffolding_count = sum(
            1 for p in self.SCAFFOLDING_PATTERNS 
            if p in response_lower
        )
        scores["scaffolding"] = min(1.0, scaffolding_count * 0.25)
        
        # 3. Check format compliance
        import re
        has_think = "<think>" in response and "</think>" in response
        has_action = bool(re.search(r"\[Action:\s*\w+\]", response, re.IGNORECASE))
        scores["format"] = (0.5 if has_think else 0.0) + (0.5 if has_action else 0.0)
        
        # 4. Check action-response alignment
        action_match = re.search(r"\[Action:\s*(\w+)\]", response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).lower()
            patterns = self.ACTION_PATTERNS.get(action, [])
            if patterns:
                matches = sum(1 for p in patterns if p in response_lower)
                scores["action_match"] = min(1.0, matches / len(patterns) + 0.3)
            else:
                scores["action_match"] = 0.5
        else:
            scores["action_match"] = 0.3
        
        # 5. Check conciseness
        # Remove thinking from word count
        visible_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        visible_response = re.sub(r"\[Action:\s*\w+\]", "", visible_response)
        word_count = len(visible_response.split())
        
        if 20 <= word_count <= 150:
            scores["conciseness"] = 1.0
        elif word_count < 20:
            scores["conciseness"] = 0.4  # Too short
        elif word_count <= 250:
            scores["conciseness"] = 0.7  # A bit long
        else:
            scores["conciseness"] = 0.3  # Way too long
        
        # Compute weighted total
        total = sum(scores[k] * self.weights[k] for k in scores)
        
        # Hard penalty for leakage
        if has_leakage:
            total = max(-0.5, total - 0.5)
        
        return total


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_data(data_path: str) -> Dict[str, List]:
    """Load trajectory data from directory or file"""
    data_dir = Path(data_path)
    
    queries = []
    responses = []
    rewards = []
    
    # Load from PPO-formatted files
    if data_dir.is_dir():
        files = list(data_dir.glob("*_ppo.jsonl")) + list(data_dir.glob("*.jsonl"))
    else:
        files = [data_dir]
    
    for f in files:
        with open(f, "r") as fp:
            for line in fp:
                try:
                    item = json.loads(line)
                    queries.append(item.get("query", ""))
                    responses.append(item.get("response", ""))
                    rewards.append(item.get("reward", 0.0))
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(queries)} trajectories from {data_path}")
    
    return {
        "query": queries,
        "response": responses,
        "reward": rewards,
    }


def create_prompts_from_data(data: Dict[str, List]) -> List[str]:
    """Create PPO prompts from loaded data"""
    prompts = []
    for query in data["query"]:
        prompt = f"You are an intelligent tutoring system. Select the best pedagogical action and generate a response.\n\n{query}\n\nRespond with:\n<think>your reasoning</think>\n[Action: ACTION_NAME]\nYour response to the student"
        prompts.append(prompt)
    return prompts


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train pedagogical tutor with PPO")
    parser.add_argument("--sft-model", type=str, required=True, 
                       help="Path to SFT model or base model")
    parser.add_argument("--output", type=str, default="./tutor-ppo", 
                       help="Output directory")
    parser.add_argument("--data", type=str, default=None, 
                       help="Path to trajectory data directory")
    parser.add_argument("--steps", type=int, default=1000, 
                       help="Number of PPO steps")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--mini-batch-size", type=int, default=2, 
                       help="Mini-batch size for PPO")
    parser.add_argument("--lr", type=float, default=1e-6, 
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.05, 
                       help="KL penalty coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2, 
                       help="PPO clip range")
    parser.add_argument("--value-clip-range", type=float, default=0.2, 
                       help="Value function clip range")
    parser.add_argument("--gamma", type=float, default=0.99, 
                       help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, 
                       help="GAE lambda")
    parser.add_argument("--use-4bit", action="store_true", 
                       help="Use 4-bit quantization")
    parser.add_argument("--lora-r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--wandb", action="store_true", 
                       help="Enable W&B logging")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Don't train, just test setup")
    
    args = parser.parse_args()
    
    # Load data
    if args.data:
        data = load_trajectory_data(args.data)
        prompts = create_prompts_from_data(data)
        logger.info(f"Created {len(prompts)} prompts for training")
    else:
        # Create dummy prompts for testing
        prompts = [
            """<|observation|>
CONCEPT: convection
CURRENT_STEP: 1/3
STUDENT_MESSAGE: "I don't understand how heat moves"
STUDENT_INTENT: confusion
CORRECTNESS: na
MASTERY: 0.20 / 0.80
<|/observation|>"""
        ] * 100
        logger.warning("No data provided, using dummy prompts")
    
    if args.dry_run:
        # Test reward function
        reward_fn = TutorRewardFunction()
        test_response = """<think>
The student is confused about heat transfer. Let me use a Socratic question.
</think>
[Action: socratic_question]
What happens when you put your hand near a warm cup of coffee? Where do you feel the warmth coming from?"""
        
        score = reward_fn._score_response(prompts[0], test_response)
        logger.info(f"Test response reward: {score:.3f}")
        
        # Test bad response (leakage)
        bad_response = """The answer is that heat moves through convection by fluid motion."""
        bad_score = reward_fn._score_response(prompts[0], bad_response)
        logger.info(f"Bad response (leakage) reward: {bad_score:.3f}")
        
        logger.info("Dry run complete. Exiting.")
        return
    
    # Import training libraries
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install transformers datasets peft trl accelerate bitsandbytes")
        sys.exit(1)
    
    # Setup W&B
    if args.wandb:
        import wandb
        wandb.init(project="tutor-ppo", config=vars(args))
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation
    
    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Load model with value head
    logger.info(f"Loading model from {args.sft_model}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )
    
    # Load reference model (frozen copy)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )
    
    # LoRA config for efficient training
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # PPO config
    ppo_config = PPOConfig(
        model_name=args.sft_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        max_grad_norm=0.5,
        kl_penalty="kl",
        init_kl_coef=args.kl_coef,
        target_kl=0.1,
        cliprange=args.clip_range,
        cliprange_value=args.value_clip_range,
        gamma=args.gamma,
        lam=args.lam,
        log_with="wandb" if args.wandb else None,
    )
    
    # Create dataset
    dataset = Dataset.from_dict({"query": prompts})
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # Create reward function
    reward_fn = TutorRewardFunction()
    
    # Generation config
    generation_kwargs = {
        "max_new_tokens": 400,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Training loop
    logger.info("Starting PPO training...")
    
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= args.steps:
            break
        
        query_tensors = batch["input_ids"]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        
        # Decode
        queries = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute rewards
        rewards = reward_fn(queries, responses)
        rewards = [torch.tensor(r) for r in rewards]
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log
        if step % 10 == 0:
            avg_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info(f"Step {step}: avg_reward={avg_reward:.3f}")
            
            if args.wandb:
                import wandb
                wandb.log({
                    "step": step,
                    "avg_reward": avg_reward,
                    "ppo/policy_loss": stats["ppo/loss/policy"],
                    "ppo/value_loss": stats["ppo/loss/value"],
                    "ppo/kl": stats.get("objective/kl", 0),
                })
    
    # Save model
    logger.info(f"Saving model to {args.output}")
    ppo_trainer.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    if args.wandb:
        import wandb
        wandb.finish()
    
    logger.info("PPO training complete!")


if __name__ == "__main__":
    main()
