# SRL Multi-Stage Reasoning: Usage, Dataset Building, and Training Guide

This guide explains how to enable and use the SRL (Supervised Reinforcement Learning) multi-stage reasoning flow in the Tutor Agent, generate datasets, and train an LLM with SRL artifacts.

## Overview

- **Stages**
  - **Classification**: Detects intent, affect, and concept.
  - **Planning (Internal Reasoning)**: Generates a `TutorPlan` with `<think>...</think>` reasoning and intended action.
  - **Execution**: Generates the response guided by the plan (and optionally self-critique).
  - **Self-Critique (Optional)**: Critiques the response before committing.
- **Artifacts**
  - **Plan**: reasoning, intended action, pedagogy focus, retrieval query, assumptions/risks, and confidence.
  - **Critique**: quality, issues, suggestions, and should_revise.
  - **Export**: When enabled, SRL artifacts are included in the observation for RL datasets.

## Components and Prompts

- **Planner**: `backend/agents/tutor/planning.py`
  - Produces `TutorPlan` via `tutor.srl_planning` prompt in `prompts/baseline.yaml`.
- **Responses**: `backend/agents/tutor/responses.py`
  - `generate_explain_response_with_plan` uses `tutor.explain_with_plan` to align response with the plan.
- **Self-Critique**: `backend/agents/tutor/self_critique.py`
  - `SelfCritic.critique_response` uses `tutor.self_critique` prompt.
- **Agent Integration**: `backend/agents/tutor/agent.py`
  - SRL planning integrated after classification.
  - Retrieval guided by `TutorPlan.retrieval_query` and `pedagogy_focus`.
  - Execution uses plan-aware builder for explain; other actions supported too.
  - Optional self-critique stage with logging.
  - SRL artifacts can be exported into the observation.

## Environment Flags

- **Core**
  - `TUTOR_SRL_MODE` (default: false): enable SRL flow.
  - `TUTOR_SRL_PLANNING_ENABLED` (default: true): enable planning stage.
  - `TUTOR_SRL_SELF_CRITIQUE` (default: false): enable critique stage.
  - `TUTOR_SRL_LOG_THINKING` (default: true): log planning trace (truncated).
  - `TUTOR_SRL_LOG_CRITIQUE` (default: true): log critique summary.
  - `TUTOR_RL_EXPORT_REASONING` (default: false): export plan/critique into `observation.srl`.
  - `TUTOR_SRL_FALLBACK_TO_RULES` (default: true): rule-based fallback if LLM planning fails.
- **Optional Features**
  - `TUTOR_MASTERY_REALTIME_UPDATE` and related: real-time mastery updater.
  - `TUTOR_PREREQ_CHECK_ENABLED` (default: true): prerequisite gating.
- See `.env.example` for defaults and hints.

## Enabling SRL

- Set flags (example for development):
```
USE_LLM_MOCK=1
TUTOR_SRL_MODE=1
TUTOR_SRL_PLANNING_ENABLED=1
TUTOR_SRL_SELF_CRITIQUE=1
TUTOR_SRL_LOG_THINKING=1
TUTOR_SRL_LOG_CRITIQUE=1
TUTOR_RL_EXPORT_REASONING=1
PROMPT_SET=baseline
```
- Prompts used: `tutor.srl_planning`, `tutor.explain_with_plan`, `tutor.self_critique` in `prompts/baseline.yaml`.

## Generating Datasets

Two ways to build datasets: mock (no DB) and agent-backed (DB needed).

### Option A: Mock Mode (fast, no DB)
- Input file: `sample/mock_observations.jsonl` (contains pre-populated observation skeletons)
- Command:
```
python scripts/tutor_rollout_bandit.py \
  --observations sample/mock_observations.jsonl \
  --out-dir /tmp/rollout_srl \
  --candidates 2 \
  --actions explain,ask \
  --mock
```
- Output:
  - `/tmp/rollout_srl/sft.jsonl`
  - `/tmp/rollout_srl/prefs.jsonl`
- Notes:
  - Mock mode emits deterministic responses; does not call the agent or DB.
  - SRL artifacts in observation (`observation.srl`) are only present in agent-backed runs; mock mode won’t include them.

### Option B: Agent-Backed (SRL artifacts included)
- Start services (at least Postgres):
```
docker-compose up -d postgres
```
- Ensure `.env` is configured (DB host/port, SRL flags as above). For local Postgres, use the compose defaults.
- Prepare observations JSONL where each entry has a `payload` with valid UUID `user_id` (session_id optional):
```
{"payload": {"message": "Explain conduction.", "user_id": "00000000-0000-0000-0000-000000000201", "target_concepts": ["Conduction"]}}
```
- Run rollout without `--mock` to use the agent flow (SRL planning, retrieval, execution, and optional critique):
```
python scripts/tutor_rollout_bandit.py \
  --observations path/to/your_observations.jsonl \
  --out-dir /tmp/rollout_srl_agent \
  --candidates 2 \
  --actions auto
```
- Output includes SRL artifacts under `observation.srl` when `TUTOR_RL_EXPORT_REASONING=1`.

### Option C: Observation Pipeline → Rollout (Heat Transfer Demo)

- Ensure you have ingested at least one heat transfer resource and configured
  its `resource_ids` in `scripts/observations/config/domain_heat_transfer.yaml`.
- Generate observations for the heat transfer domain (OBS-01–OBS-04 pipeline):
  - `make obs_heat_transfer`
  - This writes `datasets/heat_transfer/obs_latest/observations.jsonl`.
- Run a small mock rollout on these observations (no DB/agent required):
  - `make rollout_heat_transfer`
  - This writes SFT and preference datasets under `datasets/heat_transfer/rollout_latest/`.
- To run an agent-backed rollout instead, point `--observations` at the same
  `observations.jsonl` and omit `--mock`/`USE_LLM_MOCK` as described above.

### Validate Datasets
```
python scripts/validate_tutor_datasets.py --sft /tmp/rollout_srl/sft.jsonl --prefs /tmp/rollout_srl/prefs.jsonl
```

## Dataset Structure (key fields)

- **SFT record**
  - `observation`: Full observation including classifier, tutor state, retrieval, policy, session, action, and optionally `srl`.
  - `action`: Action metadata (type, params, source_chunk_ids, override fields).
  - `response`: Tutor response string.
  - `reward`: Component scores, total, weights, flags.
  - `critic`: Independent critic scores (clarity/accuracy/support/hallucination/confidence/notes).
  - `meta`: Candidate index, prompt_set tag, etc.
- **SRL artifacts**
  - `observation.srl.plan`: `thinking`, `intended_action`, `rationale`, `confidence`, `assumptions`, `risks`.
  - `observation.srl.critique`: `quality`, `issues`, `suggestions`, `should_revise` (if enabled).

## Training With SRL

You can train in two complementary ways:

### 1) Supervised Fine-Tuning (SFT)
- Goal: Learn to produce tutor responses conditioned on the observation and optionally the SRL plan.
- Input construction (recommended):
  - Concatenate: student message, focus concept, retrieved snippets, and (if present) plan rationale/thinking.
  - Target: `response`.
- Example prompt template for SFT (pseudo):
```
[Student] {{observation.user.message}}
[Focus] {{observation.tutor.focus_concept}} (level={{observation.tutor.concept_level}})
[Context]
{{#each observation.retrieval.chunks}}- {{this.snippet}}\n{{/each}}
{{#if observation.srl.plan}}
[Plan]
- Intended: {{observation.srl.plan.intended_action}}
- Rationale: {{observation.srl.plan.rationale}}
{{/if}}
[You] (respond concisely and ground in context):
```
- Tools: Hugging Face `transformers` + `trl` (SFTTrainer). Typical setup:
  - Convert JSONL to `datasets.Dataset`.
  - Map each row to `input_ids`/`labels` using the template.
  - Train with LoRA or full fine-tuning as desired.

### 2) Preference Optimization (DPO/IPO/ORPO)
- Use `prefs.jsonl` to optimize toward preferred responses (pairs with critic scores and reward).
- Construct pairs (preferred vs. rejected) from the candidates using `preference` decision.
- Train with TRL’s `DPOTrainer` or similar (IPO/ORPO) using your base SFT model.

## Practical Tips

- **PROMPT_SET**: Use `PROMPT_SET=baseline` (or your own set) to keep prompts consistent during rollout and training.
- **Model Overrides**: The agent accepts an optional `model_hint` per call; the rollout script can be extended to pass per-candidate models.
- **Logging**: With SRL flags, planning and critique summaries are logged for observability.
- **Fallbacks**: If planning fails, rule-based defaults keep the flow running (configurable).
- **Mock vs. Agent**: Use mock mode for quick smoke tests; use agent-backed rollout to capture SRL artifacts in `observation.srl`.

## Known Limitations

- **Revision Loop**: `TUTOR_SRL_REVISION_LOOP` is not implemented; critique does not automatically trigger revision.
- **External Deps**: Some tests require Postgres and/or real LLM access; mock mode skips these.

## Quick Checklist

- **Enable SRL**: Set flags in `.env`.
- **Run Rollout**: Use mock or agent-backed flow to produce `sft.jsonl` and `prefs.jsonl`.
- **Validate**: Run the dataset validator script.
- **Train**: SFT with plan-conditioned inputs + preference optimization.
- **Evaluate**: Use the same critic to measure gains and iterate prompts/weights.
