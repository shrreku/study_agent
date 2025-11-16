# Observation Pipeline & RL Rollout Guide (OBS-01..OBS-05)

This guide explains how the observation pipeline (OBS-01..OBS-05) is wired in
StudyAgent, how to configure LLM usage (including model selection), and how to
run end-to-end tests from ingestion → observations → rollout.

For deeper background, see:

- `scripts/observations/README.md` — design and pipeline details.
- `SRL_GUIDE.md` — SRL planning, rollout usage, and dataset structure.

---

## 1. Components & Files

**Config & CLI**

- `scripts/observations/config/domain_heat_transfer.yaml`
- `scripts/observations/cli/build_observations.py`
- `Makefile` targets:
  - `make obs_heat_transfer`
  - `make rollout_heat_transfer`

**Builders**

- Concept inventory: `scripts/observations/builders/concept_inventory.py`
- Mastery sampler: `scripts/observations/builders/mastery_sampler.py`
- Retrieval sampler: `scripts/observations/builders/retrieval_sampler.py`
- Scenario templates (templates + LLM):
  - `scripts/observations/builders/scenario_templates.py`
- Observation builder:
  - `scripts/observations/builders/observation_builder.py`
- Validation:
  - `scripts/observations/validators/obs_validation.py`

**Rollout & API helpers**

- Rollout CLI: `scripts/tutor_rollout_bandit.py`
- RL simplifier (used by API tools): `backend/api/rl_simplifier.py`

---

## 2. Domain Config: Heat Transfer (OBS-01, OBS-02, OBS-03)

Config file: `scripts/observations/config/domain_heat_transfer.yaml`

Key fields:

- **Domain & resources**
  - `domain`: logical domain name (e.g., `"heat_transfer"`).
  - `resource_ids`: list of resource UUIDs (from `resource` table) used to
    build the concept inventory and retrieval.

- **Observation counts & scenario mix**
  - `observations_per_concept`: integer.
  - `scenario_distribution`: per-scenario probabilities for
    `explain`, `worked_example`, `concept_check`, `reflection`, `hint`.

- **LLM usage (OBS-02)**
  - `use_llm_messages`: when `true`, scenario templates will call an LLM to
    generate student messages instead of using only deterministic templates.
  - `use_llm_answers`: when `true`, concept-check scenarios may call an LLM to
    generate student answers (correct/near_miss/incorrect) plus
    `misconception_type`.
  - `llm_model`: optional model override for observation generation
    (student messages/answers). See **Section 3**.

- **Mastery (OBS-03)**
  - `mastery_distribution`: e.g., `{low: 0.5, medium: 0.3, high: 0.2}`.
  - `mastery_values`: nominal mastery value per bucket.
  - `concept_level_for_bucket`: maps bucket → `tutor.concept_level`.
  - `use_real_mastery`: when `true`, the builder uses
    `sample_mastery_for_user_concept` (hook for real `user_concept_mastery`).
    Currently this still delegates to synthetic sampling but exercises the hook.

- **Difficulty & concept filters (OBS-01)**
  - `difficulty_thresholds`: maps raw difficulty to `beginner`/`developing`/
    `proficient` bands.
  - `concept_include_prefixes` / `concept_exclude_prefixes`: optional filters
    over inferred concepts.

---

## 3. LLM Model Selection for Observation Generation (OBS-02)

LLM calls in the observation pipeline (student messages and concept-check
answers) are implemented in
`scripts/observations/builders/scenario_templates.py`.

### 3.1 Flags and model sources

- **Enable/disable LLM usage**
  - `use_llm_messages` and `use_llm_answers` (domain config) control *whether*
    the LLM is used at all.
  - If these are `false`, the pipeline uses deterministic templates only.

- **Model selection order** (for observation-generation LLM calls only):

  1. `llm_model` in domain config, if non-empty.
  2. `OBS_LLM_MODEL` environment variable, if set.
  3. Fallback to global defaults used by `backend.llm.call_llm_json`:
     - `LLM_MODEL_MINI` or `LLM_MODEL_NANO` env variables, or
     - hard-coded nano preview model.

### 3.2 Implementation details

- Helper `_resolve_obs_llm_model(domain_config)` picks the model based on the
  rules above.
- `_call_llm_json_for_observations(...)` imports `call_llm_json` from
  `backend.llm`/`llm` and wraps it with
  `model_override_context` (from `backend.llm.common`/`llm.common`) when a
  specific model is requested. This ensures the LLM calls for observation
  generation honor `llm_model` / `OBS_LLM_MODEL`.
- `_generate_student_message_llm(...)` and `generate_concept_check_answer(...)`
  both use `_call_llm_json_for_observations` and include:
  - strict JSON prompts,
  - length checks,
  - concept mention checks where applicable,
  - fallbacks to deterministic templates on any failure.

### 3.3 Mocking and environment

- `USE_LLM_MOCK=1` (env) short-circuits `call_llm_json` calls and returns the
  provided default payloads. This is ideal for CI and local deterministic
  testing; LLM model choice is ignored in mock mode.
- For real LLM calls, you must configure an OpenAI-compatible endpoint:
  - `OPENAI_API_BASE`
  - `OPENAI_API_KEY`
  - optional: `LLM_MODEL_MINI` / `LLM_MODEL_NANO`

---

## 4. What Each Builder Does (OBS-01, OBS-03, OBS-04)

**Concept inventory (`concept_inventory.py`)**

- Queries `chunk` for each `resource_id` in config.
- Uses `chunk.concepts` (TEXT[]) as the primary concept source.
- Aggregates per concept, infers an average difficulty, and assigns a
  `difficulty_band` using `difficulty_thresholds`.
- Applies include/exclude prefix filters and returns a sorted inventory:
  `{concept, learning_path, difficulty_band}`.

**Mastery sampler (`mastery_sampler.py`)**

- `sample_mastery(domain_config, rng)` samples a bucket from
  `mastery_distribution`, maps it to a `journey_stage`, and returns a
  `mastery_snapshot` (`mastery`, `attempts`, `correct`).
- `sample_mastery_for_user_concept(...)` is a hook for real mastery; currently it
  delegates to `sample_mastery` but is wired in via `use_real_mastery`.

**Retrieval sampler (`retrieval_sampler.py`)**

- Filters `chunk` rows by `resource_id` and concept.
- Scores rows by pedagogy role compatibility and difficulty-band alignment.
- Samples 2–6 chunks per observation, returning:
  - `chunk_ids`
  - `chunks` (with `id`, `pedagogy_role`, `snippet`, `page_number`)
  - `pedagogy_roles`

**Scenario templates (`scenario_templates.py`)**

- Chooses a scenario type using `scenario_distribution`.
- Maps to `intent` / `affect` and deterministic templates.
- When enabled, calls LLM for:
  - student messages (`use_llm_messages`),
  - concept-check answers (`use_llm_answers`).

**Observation builder (`observation_builder.py`)**

- For each concept, builds `observations_per_concept` entries:
  - samples mastery (synthetic or via `sample_mastery_for_user_concept`),
  - builds a scenario (deterministic or LLM-backed),
  - samples retrieval chunks,
  - constructs `{id, payload, observation, meta}`.
- Attaches mastery info:
  - `observation.tutor.mastery_snapshot`;
  - `meta.mastery_bucket`, `meta.journey_stage`, `meta.difficulty_band`.
- For concept-check scenarios, optionally attaches:
  - `meta.student_answer`, `meta.answer_correctness`, `meta.misconception_type`.
- Calls `validate_observation_entry` to enforce basic quality checks and skews
  stats to track skip/invalid reasons.

**Validation (`obs_validation.py`)**

- Ensures:
  - `payload.message` is non-empty,
  - at least 2 retrieval chunks,
  - focus concept appears in at least one snippet,
  - `intent` / `affect` values are valid.
- Returns `(is_valid, reason)` so the builder can drop invalid entries and track
  counts by reason.

---

## 5. CLI & Make Targets (OBS-01, OBS-04, OBS-05)

### 5.1 Build observations

**Using Make:**

```bash
make obs_heat_transfer
```

This runs:

```bash
python scripts/observations/cli/build_observations.py \
  --domain heat_transfer \
  --config scripts/observations/config/domain_heat_transfer.yaml \
  --output datasets/heat_transfer/obs_latest/observations.jsonl
```

Notes:

- Loads `.env` via `python-dotenv`.
- Connects to Postgres via `backend.core.db.get_db_conn`.
- Builds a concept inventory and then observations.
- Writes a single JSONL file with entries of the form:

```json
{"id": "obs-000001", "payload": {...}, "observation": {...}, "meta": {...}}
```

- Logs a summary including:
  - total observations,
  - per-scenario counts,
  - per-mastery bucket counts,
  - per-journey stage counts,
  - answer correctness counts.

You can also call the CLI directly and override counts:

```bash
python scripts/observations/cli/build_observations.py \
  --domain heat_transfer \
  --config scripts/observations/config/domain_heat_transfer.yaml \
  --observations-per-concept 3 \
  --max-observations 200 \
  --output /tmp/heat_transfer_obs.jsonl
```

### 5.2 Run rollout (mock)

From `SRL_GUIDE.md` and `Makefile`:

```bash
make rollout_heat_transfer
```

This runs a deterministic mock rollout:

```bash
USE_LLM_MOCK=1 python scripts/tutor_rollout_bandit.py \
  --observations datasets/heat_transfer/obs_latest/observations.jsonl \
  --out-dir datasets/heat_transfer/rollout_latest \
  --candidates 2 \
  --actions explain,ask \
  --mock \
  --seed 123
```

- `--mock` and `USE_LLM_MOCK=1` ensure the agent is not called and no external
  LLMs are used; responses are generated deterministically from the observation
  and retrieval context.
- Outputs:
  - `datasets/heat_transfer/rollout_latest/sft.jsonl`
  - `datasets/heat_transfer/rollout_latest/prefs.jsonl`

You can validate the resulting datasets with:

```bash
python scripts/validate_tutor_datasets.py \
  --sft datasets/heat_transfer/rollout_latest/sft.jsonl \
  --prefs datasets/heat_transfer/rollout_latest/prefs.jsonl
```

### 5.3 Run rollout (agent-backed)

To exercise the full agent (including SRL planning and critic), start the
backend + Postgres and run:

```bash
# Example env (adjust for your provider)
export USE_LLM_MOCK=0
export OPENAI_API_BASE="https://api.openai.com/v1"  # or AimlAPI base
export OPENAI_API_KEY="sk-..."
export LLM_MODEL_MINI="openai/gpt-4o-mini"
# Optional: override model just for observation-generation LLM calls
export OBS_LLM_MODEL="openai/gpt-5-nano-2025-08-07"

python scripts/tutor_rollout_bandit.py \
  --observations datasets/heat_transfer/obs_latest/observations.jsonl \
  --out-dir datasets/heat_transfer/rollout_llm \
  --candidates 2 \
  --actions auto \
  --seed 42
```

Tips:

- Set `TUTOR_SRL_MODE=1` and related flags per `SRL_GUIDE.md` to include SRL
  artifacts in `observation.srl`.
- Use `--prompt-set baseline` to align critic prompts with the chosen prompt
  set (also set via `PROMPT_SET`).

---

## 6. How to Test Each Ticket (OBS-01..OBS-05)

### OBS-01 — Observation Pipeline MVP

1. Ingest at least one heat-transfer resource and update
   `resource_ids` in `domain_heat_transfer.yaml`.
2. Run `make obs_heat_transfer`.
3. Confirm `datasets/heat_transfer/obs_latest/observations.jsonl` exists and is
   non-empty.
4. Spot-check a few lines:
   - `payload.message` is reasonable.
   - `observation.user`, `classifier`, `tutor`, `retrieval`, and `session`
     fields are present.
5. Optional: run a mock rollout (see OBS-05) to confirm schema compatibility.

### OBS-02 — LLM-backed Messages & Answers

1. Set `use_llm_messages: true` and/or `use_llm_answers: true` in
   `domain_heat_transfer.yaml`.
2. Configure LLM endpoint + key (`OPENAI_API_BASE`, `OPENAI_API_KEY`, etc.).
3. Optionally set `llm_model` or `OBS_LLM_MODEL` to test a specific model.
4. Run `make obs_heat_transfer`.
5. Inspect several observations:
   - For explain/reflect/hint scenarios, messages should be LLM-generated but
     still mention the concept.
   - For concept_check scenarios, `meta.answer_correctness` is one of
     `correct`, `near_miss`, `incorrect` when LLM answers are used.
6. Set `USE_LLM_MOCK=1` to verify fallback/deterministic behavior.

### OBS-03 — Mastery-aware Observations

1. Tune `mastery_distribution` and `mastery_values` in
   `domain_heat_transfer.yaml`.
2. Run `make obs_heat_transfer`.
3. Confirm each observation has:
   - `observation.tutor.mastery_snapshot` with `mastery`, `attempts`, `correct`.
   - `meta.mastery_bucket` and `meta.journey_stage`.
4. Check logs for per-bucket and per-stage counts to see the distribution.
5. Set `use_real_mastery: true` to exercise the
   `sample_mastery_for_user_concept` hook (still synthetic but wired).

### OBS-04 — Quality Checks & Metrics

1. Generate observations as above.
2. Verify logs include:
   - `total_observations`, `total_concepts`, `concepts_with_observations`.
   - Per-scenario and per-mastery counts.
   - Any `invalid::<reason>` counts (e.g., `too_few_chunks`,
     `concept_not_in_snippet`).
3. Manually inspect a sample of observations to confirm that invalid ones have
   been filtered and reasoning for skips looks sensible.

### OBS-05 — RL Rollout Integration

1. Run `make obs_heat_transfer` to produce observations.
2. Run `make rollout_heat_transfer` for a quick mock rollout:
   - Verify `sft.jsonl` and `prefs.jsonl` are created under
     `datasets/heat_transfer/rollout_latest/`.
3. Run the dataset validator as shown above.
4. For a real rollout, start the backend + Postgres and run
   `tutor_rollout_bandit.py` without `--mock` using the same observations.
5. Inspect a few SFT and preference records to confirm that:
   - `observation` includes user, classifier, tutor, retrieval, policy,
     session, action.
   - Rewards and critic scores look sane.

---

## 7. Quick End-to-End Checklist

1. **Ingest** heat-transfer resource(s); note their `resource_ids`.
2. **Configure** `domain_heat_transfer.yaml`:
   - `resource_ids`, `observations_per_concept`, `scenario_distribution`.
   - `use_llm_messages` / `use_llm_answers` and `llm_model` (or `OBS_LLM_MODEL`).
   - `mastery_distribution`, `mastery_values`, `use_real_mastery`.
3. **Generate observations**: `make obs_heat_transfer`.
4. **Run rollout (mock)**: `make rollout_heat_transfer`.
5. **Validate datasets** with `scripts/validate_tutor_datasets.py`.
6. **Run rollout (agent-backed)** with real LLM + SRL flags for deeper testing.
