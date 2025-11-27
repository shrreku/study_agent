# StudyAgent RL MDP Analysis & Revised Implementation Plan

## Executive Summary

After analyzing your codebase and reviewing state-of-the-art research on RL for intelligent tutoring systems, I've identified several improvements to your MDP definitions that will enable training a more effective pedagogical model. The key insight from recent research (especially "From Problem-Solving to Teaching Problem-Solving", arXiv:2505.15607) is that **pedagogical quality must be explicitly rewarded** alongside mastery gains, and **explicit reasoning traces** (thinking tags) improve policy learning.

---

## 1. Current Implementation Analysis

### Strengths ✓
1. **Hierarchical Architecture**: Two-level design (Planner + Tutor) aligns with SOTA
2. **Mastery Tracking**: EMA-based mastery model exists (`backend/mdp/mastery.py`)
3. **Input Analysis**: Good foundation for understanding student intent
4. **RAG Integration**: Grounded responses via vector + graph search
5. **Reward Schema**: Existing `tutor_rl.py` has multi-component reward structure

### Gaps Identified ✗
| Component | Current | SOTA Best Practice |
|-----------|---------|-------------------|
| **State** | Dict with plan, step_index, history | Cognitive graph + mastery vector + conversation |
| **Action Space** | Flow control: continue/stay/replan/reply | Pedagogical micro-actions with reasoning trace |
| **Reward** | Implicit mastery EMA only | Composite: mastery_gain + ped_quality + no_leakage |
| **Policy Output** | Raw text response | `<think>` → `[Strategy]` → Text |
| **Training Signal** | None (heuristic policy) | Verifiable rewards + LLM judge |

---

## 2. Revised MDP Definitions

### 2.1 State Space (S)

**Current** (`engine.py`):
```python
tutor_state = {
    "plan": Plan,
    "current_step_index": int,
    "student_response_history": list,
    "awaiting_student_input": bool
}
```

**Proposed** - Structured `TutorState` dataclass:

```python
@dataclass
class TutorState:
    # ===== COGNITIVE CONTEXT =====
    concept_id: str                    # Current focus concept
    concept_prerequisites: List[str]   # From Neo4j graph
    concept_depth: int                 # Distance from root in DAG
    
    # ===== MASTERY STATE =====
    student_mastery: Dict[str, float]  # {concept: 0.0-1.0} for all relevant concepts
    mastery_trajectory: List[float]    # Recent mastery deltas (last 5 turns)
    target_mastery: float              # Session goal (usually 0.8)
    
    # ===== CONVERSATION STATE =====
    turn_count: int
    last_n_exchanges: List[Exchange]   # Sliding window of (student_msg, tutor_response, analysis)
    student_errors: List[str]          # Tracked misconceptions
    hints_given: int                   # Count for current concept
    
    # ===== PLAN STATE =====
    plan: Plan
    plan_step_index: int
    steps_remaining: int
    
    # ===== RAG CONTEXT =====
    retrieved_chunk_ids: List[str]     # For grounding verification
    context_relevance_score: float     # From retrieval
```

**Key Changes**:
1. **Mastery trajectory** (not just current value) - helps detect stuck students
2. **Prerequisites** from knowledge graph - enables prerequisite-guided actions
3. **Error tracking** - enables targeted correction
4. **Hints count** - penalize over-hinting in reward

---

### 2.2 Action Space (A)

**Current** (`policies.py`):
```python
action ∈ {"continue", "stay", "replan", "reply_to_user", "finish"}
```

**Proposed** - Hierarchical action with reasoning trace:

```python
@dataclass
class TutorAction:
    # ===== LEVEL 1: REASONING (Hidden from student) =====
    thinking: str           # Internal deliberation (like DeepSeek-R1's <think> tag)
    
    # ===== LEVEL 2: PEDAGOGICAL STRATEGY =====
    strategy: PedagogicalStrategy  # Enum, see below
    
    # ===== LEVEL 3: CONTENT =====
    response_text: str      # Actual text shown to student
    
    # ===== METADATA =====
    retrieval_query: Optional[str]
    difficulty_level: str   # "introductory" | "intermediate" | "advanced"


class PedagogicalStrategy(Enum):
    # Scaffolding (avoid answer leakage)
    SOCRATIC_QUESTION = "socratic_question"     # Guide via questioning
    HINT = "hint"                                # Partial reveal
    WORKED_EXAMPLE = "worked_example"            # Step-by-step demo
    ANALOGY = "analogy"                          # Connect to known concept
    
    # Direct instruction (when scaffolding failed)
    EXPLAIN = "explain"                          # Clear explanation
    CORRECT_MISCONCEPTION = "correct"            # Address specific error
    SUMMARIZE = "summarize"                      # Consolidate learning
    
    # Assessment
    CONCEPT_CHECK = "concept_check"              # Quick verification question
    CHALLENGE = "challenge"                      # Harder problem
    
    # Flow control
    ADVANCE = "advance"                          # Move to next step
    REPLAN = "replan"                            # Regenerate plan
    CONCLUDE = "conclude"                        # End session
```

**Output Format for Training**:
```
<think>
The student said "I think heat flows from cold to hot" which shows a fundamental 
misconception about the second law of thermodynamics. They have mastery=0.3 on 
this concept. Since they've already seen the definition, I should use an analogy 
to make it intuitive rather than re-explaining.
</think>
[Strategy: ANALOGY]
Think of it like water flowing downhill - it naturally goes from high to low. 
Similarly, heat naturally flows from hot (high energy) to cold (low energy). 
Can you think of an everyday example where you've seen this happen?
```

---

### 2.3 Reward Function (R)

**Current** (`config/tutor_rl.py`):
```python
class RewardWeights:
    stepwise_rubric: float = 0.0
    rubric: float = 0.4
    intent: float = 0.2
    gating: float = 0.2
    grounding: float = 0.15
    style: float = 0.05
```

**Proposed** - Based on arXiv:2505.15607:

```python
@dataclass
class PedagogicalReward:
    """Composite reward for tutor turn"""
    
    # ===== PRIMARY: MASTERY OUTCOME (Verifiable) =====
    mastery_delta: float      # Post-turn mastery - Pre-turn mastery [-1, 1]
    mastery_weight: float = 0.35
    
    # ===== PEDAGOGICAL QUALITY (LLM Judge) =====
    no_answer_leakage: bool   # Did NOT give away the answer
    scaffolding_quality: float  # 0-1: Used hints/questions appropriately
    helpfulness: float        # 0-1: Addressed student's actual need
    tone_quality: float       # 0-1: Encouraging, patient, clear
    ped_weight: float = 0.30
    
    # ===== GROUNDING (Verifiable) =====
    factual_accuracy: float   # 0-1: Claims match retrieved chunks
    no_hallucination: bool    # Hard constraint
    grounding_weight: float = 0.15
    
    # ===== EFFICIENCY (Verifiable) =====
    response_length_penalty: float  # Penalize overly long responses
    hint_efficiency: float    # Fewer hints to achieve mastery = better
    efficiency_weight: float = 0.10
    
    # ===== STRATEGY ALIGNMENT (Verifiable) =====
    strategy_followed: float  # Did response match declared [Strategy]?
    strategy_weight: float = 0.10
    
    def compute_total(self) -> float:
        # Hard constraints (gates)
        if not self.no_hallucination:
            return -1.0  # Severe penalty
        if not self.no_answer_leakage:
            return max(0, self.mastery_delta * 0.3)  # Reduced reward
        
        # Weighted sum
        ped_score = (self.scaffolding_quality + self.helpfulness + self.tone_quality) / 3
        r_mastery = self.mastery_delta * self.mastery_weight
        r_ped = ped_score * self.ped_weight
        r_ground = self.factual_accuracy * self.grounding_weight
        r_eff = (1 - self.response_length_penalty + self.hint_efficiency) / 2 * self.efficiency_weight
        r_strat = self.strategy_followed * self.strategy_weight
        
        return r_mastery + r_ped + r_ground + r_eff + r_strat
```

**Key Insight from Research**: The penalty for answer leakage is crucial. The paper uses:
```
r = r_sol + r_ped * 1{all_judges_accept} - λ * 1{any_judge_rejects}
```
This ensures pedagogical quality is a **hard gate**, not just a soft weight.

---

### 2.4 Policy Network Architecture

**Current**: Rule-based `ConversationalTutorPolicy` in `policies.py`

**Proposed**: Train a small LLM (e.g., Llama-3.2-3B or Qwen2.5-3B) with the following architecture:

```
Input: [State Encoding] + [Conversation History] + [RAG Context]
       ↓
   Thinking Head → <think>...</think>
       ↓
   Strategy Head → [Strategy: X]
       ↓
   Response Head → Text output
```

**Model Selection Rationale**:
- **Llama-3.2-3B** or **Qwen2.5-3B**: Small enough for fast inference, capable enough for tutoring
- **Phi-3-mini (3.8B)**: Good reasoning for size, MIT license
- Train with **LoRA/QLoRA** to reduce memory requirements

---

## 3. Training Pipeline (Revised)

### Phase 1: Data Collection & Instrumentation (Week 1-2)

#### 1.1 Unified Logging Schema
Extend current logging to capture full training signal:

```python
@dataclass
class TutorTurnLog:
    # Identifiers
    session_id: str
    turn_id: int
    timestamp: float
    
    # State (before turn)
    state: TutorState
    
    # Action (what tutor did)
    action: TutorAction
    
    # Reward components
    mastery_before: float
    mastery_after: float
    mastery_delta: float
    
    # LLM Judge scores (computed async)
    judge_scores: Optional[Dict[str, float]]
    
    # Metadata
    model_id: str  # Which model generated this
    prompt_version: str
```

#### 1.2 LLM Client Modes
Update `backend/mdp/llm_client.py`:

```python
class LLMClient:
    def __init__(self, mode: str = "teacher"):
        """
        Modes:
        - teacher: SOTA model (GPT-4o/Claude-3.5) for data generation
        - student: Weaker model for adversarial simulation
        - policy: Local/remote endpoint of model being trained
        - judge: Critic model for reward computation
        """
        self.mode = mode
        self._configure_endpoint()
```

### Phase 2: Expert Trajectory Generation (Week 2-3)

#### 2.1 Teacher Model Prompting
Force explicit reasoning traces in teacher outputs:

```yaml
# Add to prompts/baseline.yaml
tutor_rl:
  teacher_trajectory: |
    You are an expert tutor demonstrating pedagogical best practices.
    
    CRITICAL RULES:
    1. NEVER give the answer directly - use scaffolding
    2. ALWAYS output your thinking in <think>...</think> tags
    3. ALWAYS declare your strategy in [Strategy: X] format
    4. Keep responses under 150 words
    
    Student state:
    - Concept: {{concept}}
    - Mastery: {{mastery}}
    - Recent errors: {{errors}}
    - Last message: "{{student_message}}"
    
    Available strategies: SOCRATIC_QUESTION, HINT, WORKED_EXAMPLE, ANALOGY, 
                         EXPLAIN, CORRECT_MISCONCEPTION, CONCEPT_CHECK
    
    Respond in format:
    <think>
    [Your reasoning about what the student needs]
    </think>
    [Strategy: STRATEGY_NAME]
    [Your response to the student]
```

#### 2.2 Adversarial Student Simulator
Create diverse student personas (`scripts/simulate_session.py`):

```python
STUDENT_PERSONAS = {
    "diligent": {
        "p_correct": 0.7,
        "p_ask_question": 0.3,
        "p_give_up": 0.05,
        "style": "I think the answer is {attempt}. Is that right?"
    },
    "confused": {
        "p_correct": 0.3,
        "p_ask_question": 0.5,
        "p_give_up": 0.1,
        "style": "I'm not sure I understand. Is it because {misconception}?"
    },
    "rusher": {
        "p_correct": 0.5,
        "p_ask_question": 0.1,
        "p_give_up": 0.2,
        "style": "Yeah got it, what's next?"
    },
    "challenger": {
        "p_correct": 0.6,
        "p_ask_question": 0.4,
        "p_give_up": 0.05,
        "style": "But what about {edge_case}? Doesn't that contradict?"
    }
}
```

### Phase 3: SFT Training (Week 3-4)

#### 3.1 Dataset Format
Convert logs to HuggingFace `Dataset`:

```python
def format_for_sft(turn_log: TutorTurnLog) -> dict:
    """Format single turn for SFT training"""
    
    # Encode state as structured text
    state_text = f"""<|state|>
Concept: {turn_log.state.concept_id}
Mastery: {turn_log.state.student_mastery.get(turn_log.state.concept_id, 0):.2f}
Turn: {turn_log.state.turn_count}
Recent errors: {', '.join(turn_log.state.student_errors[-3:])}
Plan step: {turn_log.state.plan_step_index + 1}/{len(turn_log.state.plan.steps)}
<|/state|>
"""
    
    # Encode conversation history
    history_text = format_conversation(turn_log.state.last_n_exchanges)
    
    # Encode RAG context
    context_text = f"<|context|>\n{turn_log.state.retrieved_chunks}\n<|/context|>"
    
    # Target: thinking + strategy + response
    target_text = f"""<think>
{turn_log.action.thinking}
</think>
[Strategy: {turn_log.action.strategy.value}]
{turn_log.action.response_text}"""
    
    return {
        "input": state_text + history_text + context_text,
        "output": target_text,
        "reward": turn_log.compute_reward()
    }
```

#### 3.2 Training Script
```python
# train_tutor_sft.py
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

model_id = "meta-llama/Llama-3.2-3B-Instruct"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

training_args = SFTConfig(
    output_dir="./tutor-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model_id,
    train_dataset=sft_dataset,
    peft_config=lora_config,
    args=training_args,
)
trainer.train()
```

### Phase 4: GRPO Training (Week 5-6)

**Why GRPO over PPO?**
- No critic model needed (saves VRAM)
- Uses group relative advantage (sample K responses, compare within group)
- Proven effective for reasoning (DeepSeek-R1)

#### 4.1 Reward Model Integration
```python
# reward_model.py
class PedagogicalRewardModel:
    def __init__(self):
        self.mastery_model = MasteryModel()
        self.judge_client = LLMClient(mode="judge")
        
    def compute_reward(self, 
                       state: TutorState, 
                       action: TutorAction,
                       student_response: str) -> PedagogicalReward:
        """Compute multi-component reward after tutor turn"""
        
        # 1. Mastery delta (verifiable)
        new_mastery = self._simulate_mastery_update(state, student_response)
        mastery_delta = new_mastery - state.student_mastery[state.concept_id]
        
        # 2. Pedagogical quality (LLM judge)
        judge_result = self._call_ped_judge(state, action, student_response)
        
        # 3. Grounding check (verifiable)
        grounding_score = self._check_grounding(action.response_text, state.retrieved_chunk_ids)
        
        # 4. Strategy alignment (verifiable)
        strategy_score = self._check_strategy_alignment(action.strategy, action.response_text)
        
        return PedagogicalReward(
            mastery_delta=mastery_delta,
            no_answer_leakage=judge_result["no_leakage"],
            scaffolding_quality=judge_result["scaffolding"],
            helpfulness=judge_result["helpfulness"],
            tone_quality=judge_result["tone"],
            factual_accuracy=grounding_score,
            no_hallucination=grounding_score > 0.5,
            strategy_followed=strategy_score,
            hint_efficiency=1.0 - (state.hints_given / 5)
        )
```

#### 4.2 GRPO Training Loop
```python
# train_tutor_grpo.py
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir="./tutor-grpo",
    per_device_train_batch_size=2,
    num_generations=4,  # K samples per prompt
    learning_rate=1e-6,
    kl_coef=0.05,  # KL penalty to reference model
    max_new_tokens=512,
    temperature=0.8,
)

# Custom reward function
def reward_fn(samples, prompts, outputs):
    """Compute rewards for batch of generated responses"""
    rewards = []
    for prompt, output in zip(prompts, outputs):
        state = decode_state_from_prompt(prompt)
        action = parse_action_from_output(output)
        
        # Simulate student response
        student_response = student_simulator.respond(state, action)
        
        # Compute reward
        reward = reward_model.compute_reward(state, action, student_response)
        rewards.append(reward.compute_total())
    
    return torch.tensor(rewards)

trainer = GRPOTrainer(
    model=sft_model,  # Start from SFT checkpoint
    ref_model=sft_model,  # Reference for KL
    config=grpo_config,
    reward_fn=reward_fn,
    train_dataset=prompts_dataset,
)
trainer.train()
```

---

## 4. Concrete Code Changes Required

### 4.1 Schema Updates

**File**: `backend/mdp/schemas.py`
```python
# ADD these new dataclasses
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

class PedagogicalStrategy(Enum):
    SOCRATIC_QUESTION = "socratic_question"
    HINT = "hint"
    WORKED_EXAMPLE = "worked_example"
    ANALOGY = "analogy"
    EXPLAIN = "explain"
    CORRECT_MISCONCEPTION = "correct"
    SUMMARIZE = "summarize"
    CONCEPT_CHECK = "concept_check"
    CHALLENGE = "challenge"
    ADVANCE = "advance"
    REPLAN = "replan"
    CONCLUDE = "conclude"

@dataclass
class Exchange:
    role: str  # "student" | "tutor"
    content: str
    analysis: Optional[Dict] = None
    timestamp: float = 0.0

@dataclass
class TutorState:
    concept_id: str
    concept_prerequisites: List[str] = field(default_factory=list)
    student_mastery: Dict[str, float] = field(default_factory=dict)
    mastery_trajectory: List[float] = field(default_factory=list)
    turn_count: int = 0
    last_n_exchanges: List[Exchange] = field(default_factory=list)
    student_errors: List[str] = field(default_factory=list)
    hints_given: int = 0
    plan: Optional[Plan] = None
    plan_step_index: int = 0
    retrieved_chunk_ids: List[str] = field(default_factory=list)

@dataclass  
class TutorAction:
    thinking: str
    strategy: PedagogicalStrategy
    response_text: str
    retrieval_query: Optional[str] = None
    difficulty_level: str = "intermediate"
```

### 4.2 Response Generator Update

**File**: `backend/mdp/response_generator.py`
- Parse `<think>` and `[Strategy:]` from LLM output
- Validate strategy alignment
- Log structured action

### 4.3 Reward Module

**New File**: `backend/mdp/reward.py`
- Implement `PedagogicalReward` class
- Add LLM judge integration
- Add grounding verification

### 4.4 Training Scripts

**New Files**:
- `scripts/simulate_session.py` - Generate training data
- `scripts/train_tutor_sft.py` - SFT training
- `scripts/train_tutor_grpo.py` - GRPO training
- `scripts/evaluate_tutor.py` - Evaluation metrics

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **Δ Mastery** | Average mastery gain per session | > +0.3 |
| **Answer Leakage Rate** | % turns where answer given directly | < 5% |
| **Scaffolding Rate** | % turns using hints/questions | > 60% |
| **Turns to Mastery** | Avg turns to reach 0.8 mastery | < 10 |

### 5.2 Secondary Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Strategy Accuracy | % correct strategy selection | > 80% |
| Grounding Score | % claims supported by RAG | > 90% |
| Student Satisfaction | Simulated student rating | > 4/5 |
| Adversarial Robustness | Performance vs "confused" persona | > 70% success |

---

## 6. Updated Action Plan

### Week 1-2: Instrumentation
- [ ] Update `TutorState` schema in `schemas.py`
- [ ] Add `PedagogicalStrategy` enum
- [ ] Create `TutorAction` dataclass
- [ ] Update `Orchestrator` to log full `TutorTurnLog`
- [ ] Add `LLMClient` modes (teacher/student/judge/policy)

### Week 2-3: Data Generation
- [ ] Create `scripts/simulate_session.py`
- [ ] Implement student personas
- [ ] Add teacher prompting for reasoning traces
- [ ] Generate 1000 expert trajectories
- [ ] Validate data quality (check mastery deltas, strategy distribution)

### Week 3-4: SFT Training
- [ ] Prepare training `Dockerfile`
- [ ] Write `train_tutor_sft.py`
- [ ] Train on cloud GPU (RunPod/Lambda)
- [ ] Evaluate SFT model on held-out set

### Week 5-6: GRPO Training
- [ ] Implement `PedagogicalRewardModel`
- [ ] Write `train_tutor_grpo.py`
- [ ] Run GRPO training
- [ ] Compare GRPO vs SFT-only

### Week 7: Evaluation & Integration
- [ ] Run full evaluation suite
- [ ] Integrate trained model into production
- [ ] A/B test vs baseline

---

## 7. Key Research References

1. **"From Problem-Solving to Teaching Problem-Solving"** (arXiv:2505.15607)
   - Multi-turn RL for tutoring
   - Composite reward: solve_rate + pedagogical_quality
   - Thinking tags for explicit reasoning

2. **"Towards Goal-oriented Intelligent Tutoring Systems"** (arXiv:2312.10053)
   - Cognitive graph state representation
   - Prerequisite-guided action selection
   - MDP formulation for ITS

3. **"DeepSeekMath: GRPO"** (arXiv:2402.03300)
   - Group Relative Policy Optimization
   - No critic model needed
   - Efficient RL for LLMs

4. **"Get a Head Start"** (AAAI 2024)
   - Off-policy evaluation for ITS
   - Pedagogical policy selection

---

## 8. Summary of Key Changes to Your Plan

| Aspect | Your Original Plan | Revised Recommendation |
|--------|-------------------|----------------------|
| **State** | Basic dict | Structured `TutorState` with mastery trajectory, prerequisites |
| **Action** | Flow control only | Thinking + Strategy + Response (hierarchical) |
| **Reward** | Mastery delta | Composite: mastery + ped_quality + grounding + efficiency |
| **Training** | PPO | **GRPO** (simpler, no critic, proven for reasoning) |
| **Output Format** | Raw text | `<think>` → `[Strategy]` → Text |
| **Evaluation** | Mastery gain only | Add leakage rate, scaffolding rate, strategy accuracy |

The most impactful change is **forcing explicit strategy selection** before response generation - this makes the model's pedagogical reasoning auditable and trainable.
