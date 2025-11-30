"""
Trajectory Logger for PPO Training

This module handles:
1. Real-time logging of transitions during tutoring sessions
2. Export to formats suitable for PPO training
3. Statistics and validation of trajectory quality
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from mdp.schemas_v2 import TutorTransition, TutorObservation, TutorAction, TutorReward

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStats:
    """Statistics for a collection of trajectories"""
    total_transitions: int = 0
    total_sessions: int = 0
    
    # Reward stats
    avg_reward: float = 0.0
    min_reward: float = 1.0
    max_reward: float = -1.0
    
    # Mastery stats
    avg_mastery_delta: float = 0.0
    positive_mastery_count: int = 0
    
    # Action distribution
    action_counts: Dict[str, int] = field(default_factory=dict)
    
    # Quality flags
    leakage_count: int = 0
    scaffolding_count: int = 0


class TrajectoryLogger:
    """
    Logs tutoring trajectories for PPO training.
    
    Features:
    - Real-time logging to file
    - Batch export for training
    - Statistics tracking
    - Data validation
    """
    
    def __init__(
        self,
        output_dir: str = "data/trajectories",
        session_id: Optional[str] = None,
        buffer_size: int = 100,
    ):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for trajectory files
            session_id: Optional session ID for file naming
            buffer_size: Number of transitions to buffer before flushing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or f"session_{int(time.time())}"
        self.buffer_size = buffer_size
        
        self.transitions: List[TutorTransition] = []
        self.stats = TrajectoryStats()
        self._sessions_seen = set()
        
        # File paths
        self.jsonl_path = self.output_dir / f"{self.session_id}.jsonl"
        self.ppo_path = self.output_dir / f"{self.session_id}_ppo.jsonl"
        
        logger.info(f"TrajectoryLogger initialized: {self.output_dir}")
    
    def log(self, transition: TutorTransition):
        """Log a single transition"""
        self.transitions.append(transition)
        self._update_stats(transition)
        
        # Track unique sessions
        if transition.session_id:
            self._sessions_seen.add(transition.session_id)
        
        # Flush to file periodically
        if len(self.transitions) >= self.buffer_size:
            self.flush()
        
        logger.debug(f"Logged transition {transition.transition_id}", extra={
            "session": transition.session_id,
            "turn": transition.turn_number,
            "action": transition.action.action.value if transition.action else None,
            "reward": transition.reward.compute_total() if transition.reward else None,
        })
    
    def _update_stats(self, t: TutorTransition):
        """Update running statistics"""
        self.stats.total_transitions += 1
        self.stats.total_sessions = len(self._sessions_seen)
        
        if t.reward:
            total_reward = t.reward.compute_total()
            
            # Update running average
            n = self.stats.total_transitions
            self.stats.avg_reward = (
                (self.stats.avg_reward * (n - 1) + total_reward) / n
            )
            
            self.stats.min_reward = min(self.stats.min_reward, total_reward)
            self.stats.max_reward = max(self.stats.max_reward, total_reward)
            
            # Mastery stats
            if t.reward.mastery_delta > 0:
                self.stats.positive_mastery_count += 1
            self.stats.avg_mastery_delta = (
                (self.stats.avg_mastery_delta * (n - 1) + t.reward.mastery_delta) / n
            )
            
            # Quality flags
            if not t.reward.no_answer_leakage:
                self.stats.leakage_count += 1
            if t.reward.scaffolding_used:
                self.stats.scaffolding_count += 1
        
        if t.action:
            action_name = t.action.action.value
            self.stats.action_counts[action_name] = (
                self.stats.action_counts.get(action_name, 0) + 1
            )
    
    def flush(self):
        """Flush buffer to files"""
        if not self.transitions:
            return
        
        # Write raw transitions
        with open(self.jsonl_path, "a") as f:
            for t in self.transitions:
                f.write(json.dumps(t.to_dict()) + "\n")
        
        # Write PPO format
        with open(self.ppo_path, "a") as f:
            for t in self.transitions:
                f.write(json.dumps(t.to_ppo_format()) + "\n")
        
        logger.info(f"Flushed {len(self.transitions)} transitions to {self.jsonl_path}")
        self.transitions = []
    
    def close(self):
        """Flush remaining and close"""
        self.flush()
        self._write_stats()
        logger.info(f"TrajectoryLogger closed. Stats: {asdict(self.stats)}")
    
    def _write_stats(self):
        """Write statistics file"""
        stats_path = self.output_dir / f"{self.session_id}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)
    
    def get_stats(self) -> TrajectoryStats:
        """Get current statistics"""
        return self.stats
    
    def validate(self) -> Dict[str, Any]:
        """Validate trajectory quality"""
        issues = []
        warnings = []
        
        # Check for answer leakage
        leakage_rate = (
            self.stats.leakage_count / max(1, self.stats.total_transitions)
        )
        if leakage_rate > 0.1:
            issues.append(f"High answer leakage rate: {leakage_rate:.1%}")
        elif leakage_rate > 0.05:
            warnings.append(f"Moderate answer leakage rate: {leakage_rate:.1%}")
        
        # Check scaffolding rate
        scaffolding_rate = (
            self.stats.scaffolding_count / max(1, self.stats.total_transitions)
        )
        if scaffolding_rate < 0.3:
            warnings.append(f"Low scaffolding rate: {scaffolding_rate:.1%}")
        
        # Check mastery improvement
        if self.stats.avg_mastery_delta < 0:
            issues.append(f"Negative average mastery delta: {self.stats.avg_mastery_delta:.3f}")
        
        # Check action diversity
        if len(self.stats.action_counts) < 3:
            warnings.append(f"Low action diversity: {len(self.stats.action_counts)} unique actions")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": asdict(self.stats),
        }


class TrajectoryDataset:
    """
    Dataset wrapper for PPO training.
    
    Loads trajectories from files and provides iteration for training.
    """
    
    def __init__(self, data_dir: str):
        """Load all trajectories from directory"""
        self.data_dir = Path(data_dir)
        self.trajectories: List[Dict[str, Any]] = []
        self._load_all()
    
    def _load_all(self):
        """Load all JSONL files"""
        for jsonl_file in self.data_dir.glob("*_ppo.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    self.trajectories.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.trajectories)} trajectories from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.trajectories[idx]
    
    def get_queries(self) -> List[str]:
        """Get all queries (observations) for PPO"""
        return [t["query"] for t in self.trajectories]
    
    def get_responses(self) -> List[str]:
        """Get all responses (actions) for PPO"""
        return [t["response"] for t in self.trajectories]
    
    def get_rewards(self) -> List[float]:
        """Get all rewards"""
        return [t["reward"] for t in self.trajectories]
    
    def filter_by_reward(self, min_reward: float = 0.0) -> "TrajectoryDataset":
        """Filter trajectories by minimum reward"""
        filtered = TrajectoryDataset.__new__(TrajectoryDataset)
        filtered.data_dir = self.data_dir
        filtered.trajectories = [
            t for t in self.trajectories if t["reward"] >= min_reward
        ]
        logger.info(f"Filtered to {len(filtered)} trajectories with reward >= {min_reward}")
        return filtered
    
    def split(self, train_ratio: float = 0.9) -> tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split into train/eval sets"""
        import random
        
        indices = list(range(len(self.trajectories)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]
        
        train_ds = TrajectoryDataset.__new__(TrajectoryDataset)
        train_ds.data_dir = self.data_dir
        train_ds.trajectories = [self.trajectories[i] for i in train_indices]
        
        eval_ds = TrajectoryDataset.__new__(TrajectoryDataset)
        eval_ds.data_dir = self.data_dir
        eval_ds.trajectories = [self.trajectories[i] for i in eval_indices]
        
        return train_ds, eval_ds
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.trajectories:
            return {"count": 0}
        
        rewards = self.get_rewards()
        
        return {
            "count": len(self.trajectories),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "positive_reward_rate": sum(1 for r in rewards if r > 0) / len(rewards),
        }


def merge_trajectory_files(
    input_dir: str, 
    output_file: str,
    min_reward: Optional[float] = None
) -> Dict[str, Any]:
    """
    Merge multiple trajectory files into one.
    
    Args:
        input_dir: Directory with trajectory files
        output_file: Output JSONL file
        min_reward: Optional minimum reward filter
        
    Returns:
        Statistics about merged data
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    trajectories = []
    stats = defaultdict(int)
    
    for jsonl_file in input_path.glob("*_ppo.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                t = json.loads(line)
                
                # Apply filter
                if min_reward is not None and t["reward"] < min_reward:
                    stats["filtered_out"] += 1
                    continue
                
                trajectories.append(t)
                stats["total"] += 1
    
    # Write merged file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
    
    logger.info(f"Merged {stats['total']} trajectories to {output_path}")
    
    return dict(stats)
