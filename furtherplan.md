I'll turn your bullet list into a concrete, ordered plan with phases, and map each point to specific tasks using the scripts and APIs you already have. I’ll also update the internal TODOs so we can track progress step‑by‑step.

Here’s a concrete, per‑point plan that stays aligned with your current codebase (ingestion → observations → rollouts → datasets → evals → training → deployment).

I’ll group your bullets into 3 phases and then break each bullet into actionable steps.

---

## Phase 1 – Ingestion, Observations, Rollouts, Datasets

### 1. Faster model for ingestion

**Goal:** Reduce cost/latency of chunk tagging, pedagogy roles, KG building, without killing quality.

**Plan:**

- **[1.1] Map ingestion LLM calls**
  - Where LLM is used now:
    - [backend/ingestion/hierarchical_tagger.py](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/ingestion/hierarchical_tagger.py:0:0-0:0) → `call_llm_json` for `ingest.chunk_tags_hierarchical`, `ingest.formula_extraction`.
    - `llm.extract_pedagogy_relations` via [backend/api/resources.py](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/api/resources.py:0:0-0:0) (pedagogy KG).
  - Identify which calls are *critical* vs “nice to have” (e.g., formulas can be lighter).

- **[1.2] Choose faster models**
  - For tagging/KG: use a mid‑size instruction model (e.g., GPT‑4o‑mini, Claude Haiku, Qwen2.5‑7B‑Instruct) via API, or a local 7B model on GPU.
  - Keep a higher‑end model (e.g., GPT‑4.1/Claude Opus) only for evals, not ingestion.

- **[1.3] Wire model selection**
  - Use your `llm.common` abstraction (and `model_override_context`) to:
    - Define envs like `INGEST_MODEL_HINT`, `PEDAGOGY_MODEL_HINT`.
    - For ingestion jobs, set those env vars or wrap ingestion in `model_override_context("fast-ingest-model")`.
  - Add a “fast mode” flag: e.g., `ENHANCED_CHUNKING_FAST=true` to:
    - Shorten prompts (truncate chunk text).
    - Disable expensive subroutines (e.g., formula metadata) when not needed.

- **[1.4] Benchmark quality vs speed**
  - Select ~50 representative chunks across domains.
  - Run tagging with current model vs faster candidate.
  - Compare:
    - Pedagogy_role distribution.
    - Difficulty estimates.
    - Concept coverage.
  - Decide acceptable trade‑offs and freeze ingest model choice.

---

### 2. Ingest resources with resource id

**Goal:** Simple, repeatable process to add new domain PDFs and ensure they are chunked, tagged, and in the KG.

**Plan:**

- **[2.1] Resource upload + parse job**
  - Use `/api/resources/upload` to store PDFs and enqueue parse jobs.
  - Save mapping `{domain, resource_title → resource_id}` in a small config file or DB table to reuse.

- **[2.2] Reindex on demand**
  - For each new resource_id:
    - `POST /api/resources/{resource_id}/reindex`
    - Ensure `ENHANCED_CHUNKING_ENABLED=true` so `enhanced_structural_chunk_resource` runs.
  - This writes chunks into `chunk` table, with tags partially filled.

- **[2.3] Enhanced KG backfill**
  - Run `scripts/kg_backfill_enhanced.py --resource-id <UUID> --use-sections` (default).
  - This aggregates by section and calls `build_enhanced_educational_kg`.
  - Make a simple driver script or Makefile target:
    - `make ingest RESOURCE_ID=<uuid> DOMAIN=<name>`

- **[2.4] Validation pass**
  - Add a quick script to:
    - Count chunks per resource, distribution of `tags.pedagogy_role` and difficulty.
    - Spot obviously broken parses (all `unknown`, empty text).

---

### 3. Generate observations

**Goal:** Build `observations.jsonl` per domain, optimized for pedagogy and verifiable rewards.

**Plan:**

- **[3.1] Implement “Observation Builder”**
  - A Python script (e.g., `scripts/build_observations.py`) that:
    - Reads concepts + chunk tags from DB for given `resource_id`/domain.
    - For each concept, chooses:
      - 2–4 scenario types (explain, worked_example, hint, concept_check, reflection, derivation).
      - Student message template with small randomness.
    - Samples 3–6 chunks matching:
      - `tags.pedagogy_role` (definition/explanation/example/summary, etc.)
      - `tags.difficulty` aligned to `concept_level`.
    - Fills out:
      - `payload` (message, user_id, target_concepts, resource_id).
      - `observation` overrides: `classifier`, `tutor`, `retrieval`, `policy`.

- **[3.2] Coverage and counts**
  - Target per domain:
    - 200–500 concepts × 2–4 scenarios → ~2k–5k observations.
  - Ensure mix of intents and levels:
    - `question` ~50%, `answer` ~20%, `reflection` ~15%, others ~15%.
    - `beginner` ~50%, `developing` ~30%, `proficient` ~20%.

- **[3.3] Quality checks**
  - Filter out any observation where:
    - <2 retrieved chunks.
    - Focus concept not found in any snippet.
  - Save as `datasets/{domain}/{date}/observations.jsonl`.

---

### 4. Run rollouts with SOTA

**Goal:** Use top models via API to generate candidate tutor responses for each observation.

**Plan:**

- **[4.1] Decide rollout config**
  - `candidates`: 3–4 per observation.
  - `actions`: e.g., `["auto", "explain", "ask", "hint"]`.
  - SRL flags:
    - `TUTOR_SRL_MODE=true`
    - `TUTOR_SRL_PLANNING_ENABLED=true`
    - `TUTOR_SRL_MULTI_STEP_EXECUTE=true`
    - `TUTOR_STEPWISE_RUBRIC_ENABLED=true`
    - `TUTOR_RL_EXPORT_STEP_SCORES=true`
    - `TUTOR_RL_EXPORT_REASONING=true`

- **[4.2] Use RL API wrapper**
  - Call `POST /api/rl/rollout` with:
    - `observations`: from `observations.jsonl`.
    - `model_per_candidate`: map action→SOTA model (GPT‑4.1, Claude 3.5, etc.).
    - `critic_model`: strong but maybe smaller than top SOTA.
    - `simplified`: false (for full data) or true for UI.
    - `detailed_steps`: true for SRL analysis.
  - Write output sft/prefs to `datasets/{domain}/{date}/sft.jsonl` and [prefs.jsonl](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/sample/mock_rollout/prefs.jsonl:0:0-0:0) (API uses same structure as CLI).

- **[4.3] Optional CLI rollouts**
  - For mock or offline runs:
    - `python scripts/tutor_rollout_bandit.py --observations datasets/.../observations.jsonl --out-dir datasets/.../rollout --candidates 3 --actions explain,ask,hint,reflect --prompt-set concise --mock/--no-mock`.

---

### 5. Generate dataset

**Goal:** Clean, validated, versioned SFT and preference datasets ready for training.

**Plan:**

- **[5.1] Validate schema**
  - `python scripts/validate_tutor_datasets.py --sft datasets/.../sft.jsonl --prefs datasets/.../prefs.jsonl --redact datasets/.../redacted/`
  - Fix any schema issues at the observation/rollout level.

- **[5.2] Filter and balance**
  - Create a small “dataset filter” script that:
    - Drops rows with:
      - `reward.total < 0.5`
      - `critic.hallucination_flag == true`
      - `advanced_concept_drift` or `prereq_gating_failed` flags.
    - Rebalances:
      - Actions (`explain`, `ask`, `hint`, `reflect`).
      - Concept levels.
      - Domains.

- **[5.3] Versioning**
  - Adopt a naming convention:
    - `datasets/tutor_rl/v1/{domain}/sft.jsonl`, [prefs.jsonl](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/sample/mock_rollout/prefs.jsonl:0:0-0:0).
    - Store metadata JSON with:
      - `prompt_set`, SRL flags.
      - ingest model, rollout models.
      - date, seed.

---

## Phase 2 – Evals, Open-Source Baselines, Infra

### 6. Make evals

**Goal:** Robust offline evaluation suite + templates for human eval.

**Plan:**

- **[6.1] Use existing offline eval**
  - `python scripts/eval_tutor_bandit.py --baseline-sft base_sft.jsonl --optimized-sft opt_sft.jsonl --out-dir evals/run1 --baseline-label base --optimized-label dpo_run1`
  - This recomputes validator + critic scores and gives per-metric deltas.

- **[6.2] Extend metrics**
  - Add:
    - Breakdown by:
      - action type.
      - concept_level.
      - domain.
    - Flag rates (hallucination, gating, advanced_term_drift) per bucket.

- **[6.3] Evaluation datasets**
  - Freeze a small set of held‑out observations (across domains and levels).
  - Use the same evaluation pipeline to compare:
    - Base vs DPO vs PPO vs RLVR models.
    - SOTA vs your trained open-source models.

- **[6.4] Human eval templates**
  - Design 1–2 small Google‑forms‑style rubrics:
    - Clarity (1–5), correctness (1–5), pedagogical helpfulness (1–5).
    - Short free‑text comments.
  - Use them on a subset (~100–200 examples) for sanity checks.

---

### 7. Test open source models on obs

**Goal:** Evaluate HF models (Llama/Qwen/Mistral) on the same observation set.

**Plan:**

- **[7.1] Inference harness**
  - Script: load a HF model + tokenizer.
  - For each `observation`:
    - Build a prompt that mirrors your tutor_agent prompt (or a simpler baseline prompt).
    - Generate response.
    - Wrap into a fake SFT record (`observation`, `action`, `response`).

- **[7.2] Score with validators + critic**
  - For each generated record:
    - Run `score_response` and [score_with_critic](cci:1://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/agents/tutor/critic.py:86:0-134:17).
    - Save SFT‑like JSONL so you can reuse `eval_tutor_bandit.py`.

- **[7.3] Compare to SOTA rollouts**
  - For the same observation subset, compare:
    - Open‑source model vs SOTA API vs trained PPO/DPO models.
  - Use both metrics and sample qualitative inspection.

---

### 8. Figure out GPU and deployment

**Goal:** Decide practical compute + deployment architecture for training and serving.

**Plan:**

- **[8.1] Training GPU plan**
  - For LoRA finetuning and PPO on 7B–8B models:
    - Single 24–48 GB GPU (e.g., 4090, A10, L4) is workable.
    - If targeting 13B/34B, aim for 2×L40s or a single A100 80GB.
  - Decide:
    - Local GPU vs cloud (Lambda, Paperspace, RunPod, GCP/AWS).
    - Budget per month.

- **[8.2] Environment**
  - Containerize:
    - Base image with CUDA + PyTorch + transformers + trl + peft + datasets.
    - Mount project repo inside container.
  - Ensure consistent `.env` for:
    - Database, Redis, MinIO, LLM API keys.

- **[8.3] Serving topology**
  - One inference service for:
    - Tutor model (serving your finetuned LoRA).
  - One “jobs” service for:
    - Rollouts with SOTA APIs.
    - PPO training (offline/cron, not exposed publicly).

- **[8.4] CI hooks**
  - Add smoke tests:
    - Tiny mock rollout.
    - Tiny training step (mock mode in `train_tutor_pref.py`).
    - Ensure changes don’t break these.

---

## Phase 3 – Training code (PPO, RLVR, SRL) + Tracking

### 9. Training code – PPO, RLVR, SRL

**Goal:** Implement training harnesses using TRL + your reward functions.

**Plan:**

- **[9.1] PPO RLHF**
  - New script `scripts/train_tutor_ppo.py`:
    - Load base model (HF).
    - Wrap with TRL `PPOTrainer`.
    - Prepare prompts from SFT records or directly from observations.
    - For each batch:
      - Generate responses.
      - Compute reward via:
        - `score_response` and [score_with_critic](cci:1://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/agents/tutor/critic.py:86:0-134:17).
        - Combine into scalar reward:
          - `R = α * reward.total + β * critic.confidence − penalties`.
      - Run PPO step.
    - Log:
      - Average reward.
      - Component scores.
      - KL divergence vs reference.

- **[9.2] RLVR integration**
  - Implement verifiers:
    - Grounding passes? Gating passes? Style within thresholds? No hallucination_flag?
  - Reward:
    - `R = 1` if all verifiers pass; `0` otherwise (or multi‑binary weighted).
  - Switch between “dense” reward (original) and “verifiable” reward (RLVR) via a flag:
    - `TUTOR_RLVR_MODE=true`.

- **[9.3] SRL self‑rewarding loop**
  - Exploit SRL fields exported by agent (`observation.srl.plan/critique`):
    - Use critic or a dedicated SRL prompt to judge:
      - Stepwise quality.
      - Plan vs execution consistency.
    - For iterative DPO:
      - Generate several candidate responses/actions.
      - Self‑label preferences via critic (like your [preference_with_critic](cci:1://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/backend/agents/tutor/critic.py:164:0-235:17)).
      - Aggregate into [prefs.jsonl](cci:7://file:///Users/shreyashkumar/coding/projects/StudyAgent/btp_studyAgent/sample/mock_rollout/prefs.jsonl:0:0-0:0) and train via `train_tutor_pref.py`.
    - For PPO:
      - Include SRL metrics (stepwise rubric, internal consistency flags) into reward.

- **[9.4] Safety hooks**
  - Add optional verifiers for:
    - Toxicity.
    - Refusal when necessary.
    - Escalation triggers.
  - Penalize these in PPO reward and record in evals.

---

### 10. Train and track !!

**Goal:** Reliable experiment management so you know what worked.

**Plan:**

- **[10.1] Choose tracker**
  - W&B or MLflow:
    - Log:
      - Config (model, LoRA params, reward weights, RLVR toggles).
      - Metrics (reward, component scores, eval metrics).
      - Artifacts (adapter weights, metrics.json, datasets).

- **[10.2] Config management**
  - Define simple YAML/JSON configs per experiment:
    - `config/rl_runs/ppo_rlvr_v1.yaml` with:
      - model, dataset paths, reward coefficients, batch size, steps, SRL flags.
  - Training scripts accept `--config-path` and log that into tracker.

- **[10.3] Run orchestration**
  - For each experiment:
    - `make train_dpo CONFIG=...`
    - `make train_ppo CONFIG=...`
  - After training:
    - Run eval command automatically.
    - Log eval summary to tracker.

- **[10.4] Promotion rules**
  - Define simple criteria to “promote” a model:
    - Offline eval: +Δtotal reward ≥ 0.05, no increase in hallucination/unsafe flags.
    - Human eval: +0.5 in helpfulness.
  - Only promote/promote those models to “candidate for deployment.”

---

### Where we are in the plan

From your original list:

- **faster model for ingestion** → Section 1.
- **ingest resources with resource id** → Section 2.
- **generate observations** → Section 3.
- **run rollouts with SOTA** → Section 4.
- **generate dataset** → Section 5.
- **make evals** → Section 6.
- **test open source models on obs** → Section 7.
- **figure out GPU and deployment** → Section 8.
- **training code- ppo, rlvr, srl** → Section 9.
- **train and track !!** → Section 10.  

**Summary:** I translated each of your bullets into a concrete plan with steps, flags, scripts, and outputs, aligned with your existing StudyAgent code and modern RLHF/RLVR/SRL practice.