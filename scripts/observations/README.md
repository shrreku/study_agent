# Observation Generation Pipeline

This folder will contain **code and configs** for generating high-quality, adjustable observations for RL training of the tutor agent.

The goal is to turn **ingested domain resources + concept graph + mastery data** into `observations.jsonl` batches that can be used by:

- `scripts/tutor_rollout_bandit.py`
- `/api/rl/rollout`

…to produce SFT / preference / RL datasets.

---

## 1. Design Goals

- **High-quality**: Observations include realistic student messages, concept context, pedagogy roles, and retrieval.
- **Adjustable**:
  - per-domain, per-concept, per-difficulty knobs
  - control counts (observations per concept / scenario)
  - control mastery buckets (low/med/high)
- **Grounded**: Always link to real chunks (with pedagogy tags & difficulty) from the DB.
- **Pedagogy-aware**: Include intent, affect, learning path, prerequisite status.
- **LLM-assisted but deterministic-first**: Use LLMs where they add value (student language, misconceptions), but keep the structure DB-driven with simple fallbacks so generation is robust and reproducible.
- **RL-ready**: Output observations in a format compatible with RL rollout scripts and schemas.

---

## 2. Folder Structure

Planned structure for this folder:

- `scripts/observations/`
  - `README.md` (this file)
  - `config/`
    - `domain_heat_transfer.yaml` (example domain config)
    - `defaults.yaml` (global knobs)
  - `builders/`
    - `concept_inventory.py` (build per-domain concept lists & learning paths)
    - `mastery_sampler.py` (sample realistic mastery snapshots from DB or heuristics)
    - `retrieval_sampler.py` (pick appropriate chunks given concept/pedagogy role/difficulty)
    - `scenario_templates.py` (define scenario types: explain, worked_example, hint, reflection, etc.)
    - `observation_builder.py` (core logic: produce observation dicts)
  - `cli/`
    - `build_observations.py` (CLI entrypoint; uses config + builders to emit jsonl)
  - `utils/`
    - `db.py` (thin wrappers around `get_db_conn` for read-only queries)
    - `logging_utils.py`

This plan does **not** require all files to exist immediately; they can be implemented incrementally.

---

## 3. Inputs and Outputs

### Inputs

1. **Domain config** (YAML)
   - Domain name, e.g. `"heat_transfer"`.
   - Resource IDs to use.
   - Concept filters (e.g. include/exclude prefixes).
   - Target counts:
     - `observations_per_concept`
     - `scenario_distribution` (percent for explain/ask/hint/etc.)
   - Mastery buckets + desired proportions.

2. **Database / KG**
   - `resource` and `chunk` tables (with `tags.pedagogy_role`, `tags.difficulty`, `key_concepts`, etc.).
   - Concept/prereq graph from Neo4j (optional; can be approximated via chunk tags).
   - `user_concept_mastery` table (for real mastery snapshots if available).

3. **Environment settings**
   - DB connection envs.
   - Feature flags (e.g. use real mastery vs synthetic, include SRL fields, etc.).

### Outputs

- `datasets/{domain}/{date}/observations.jsonl`
  - Each line: an **observation spec** compatible with rollout scripts, e.g.:
    - `{"payload": {...}, "observation": {...}, "meta": {...}}`
  - `payload` fields required by the tutor API (message, user_id, target_concepts, resource_id, etc.).
  - `observation` overrides: classifier/tutor/retrieval/policy blocks.
  - `meta` for debugging and analysis (scenario type, mastery bucket, domain, etc.).

---

## 4. Core Pipeline Stages

### 4.1 Concept Inventory

**Responsibility:** choose which concepts and learning paths to cover.

Implementation sketch (`builders/concept_inventory.py`):

- Query:
  - `chunk` table for `tags.key_concepts` and `tags.pedagogy_role`.
  - Optionally Neo4j for canonical concept/prerequisite structure.
- Build a list:
  - `[{"concept": "Fourier's law", "learning_path": [...], "difficulty_band": "introductory"}, ...]`
- Apply filters from domain config.

### 4.2 Mastery Sampler

**Responsibility:** produce a realistic `mastery_snapshot` and `journey_stage` for each (user, concept) sample.

Implementation sketch (`builders/mastery_sampler.py`):

- Two modes:
  1. **Real** mastery:
     - Query `user_concept_mastery` for sampled users per concept.
     - Bucket into low/medium/high.
  2. **Synthetic** mastery:
     - Use configurable priors (e.g. 50% low, 30% medium, 20% high) and sample from a Beta distribution or simple fixed values.
- Output for each concept:
  - `{"mastery": 0.2, "attempts": 3, "correct": 1, "journey_stage": "practice"}`.

### 4.3 Scenario Templates (with optional LLM support)

**Responsibility:** define the **student side** of the observation.

Implementation sketch (`builders/scenario_templates.py`):

- Define a small set of scenario types:
  - `explain_basic`, `worked_example`, `concept_check`, `hint_misconception`, `reflection`, `prereq_review`, `derivation`, `application`.
- For each scenario:
  - Map to classifier intent/affect (e.g., `question + confused`, `answer + unsure`).
  - **Deterministic template (always available):**
    - Simple string templates such as: "I don't understand ${concept}. Could you explain simply?".
  - **Optional LLM-backed variants:**
    - Call an LLM (via `llm.common`) with a small prompt that includes:
      - `concept`, scenario type, and optionally mastery bucket (low/medium/high).
      - Ask it to return JSON: `{ "message": "..." }` representing a realistic student utterance.
    - Apply cheap guardrails before accepting the LLM output:
      - non-empty string,
      - reasonable length,
      - mentions the focus concept when required.
    - If validation fails or the LLM call errors, fall back to the deterministic template.
  - Optionally align scenario choice with mastery stage (beginners get more explain/hint scenarios; proficient students get more reflection and application).

### 4.4 Retrieval Sampler

**Responsibility:** choose relevant chunks to feed as retrieval context.

Implementation sketch (`builders/retrieval_sampler.py`):

- Query `chunk` for a concept:
  - Filter by `tags.key_concepts` containing the concept.
  - Filter/sort by `tags.pedagogy_role` (definition/explanation/example/etc.).
  - Filter by `tags.difficulty` band roughly matching concept level (beginner/developing/proficient).
- Return a small set:
  - `chunk_ids`, `chunks` (with `snippet`, `pedagogy_role`, `tags`) for the observation.

### 4.5 Observation Builder

**Responsibility:** assemble everything into a single observation dict.

Implementation sketch (`builders/observation_builder.py`):

For each concept in a domain batch:

1. Sample a scenario type (according to config distribution).
2. Get a student message from `scenario_templates`:
   - Prefer the LLM-backed variant when enabled and valid.
   - Fall back to the deterministic template on any failure or validation issue.
3. For `concept_check` / answer scenarios, optionally use an LLM helper to generate:
   - a correct / near-miss / incorrect student answer,
   - an optional `misconception_type` label stored in `meta`.
4. Sample mastery snapshot from `mastery_sampler`.
5. Retrieve relevant chunks from `retrieval_sampler`.
6. Build:
   - `payload`:
     - `message`, `user_id` (synthetic or real), `target_concepts`, `resource_id`, `session_id`.
   - `observation` overrides:
     - `classifier`: `intent`, `affect` (based on scenario).
     - `tutor`: `focus_concept`, `concept_level`, `learning_path`, `mastery_snapshot`.
     - `retrieval`: `chunk_ids`, `chunks`, `pedagogy_roles`.
   - `meta`:
     - `scenario_type`, `domain`, `mastery_bucket`, `journey_stage`, `builder_version`,
     - optional `answer_correctness` / `misconception_type` tags when LLM-generated answers are used.

7. Write as one JSON line to the output file.

### 4.6 Quality Filters and Validation

Before finalizing `observations.jsonl`:

- Run simple validation rules:
  - At least 2 chunks present.
  - Focus concept appears in at least one snippet.
  - No obviously empty/degenerate messages.
- Optionally reuse existing dataset validators (or a slim version) to sanity-check observation structure.

---

## 5. CLI Entrypoint

`cli/build_observations.py` (to be implemented) will:

1. Parse arguments:
   - `--domain heat_transfer`
   - `--config scripts/observations/config/domain_heat_transfer.yaml`
   - `--output datasets/heat_transfer/2025-11-15/observations.jsonl`
2. Load config (domain, counts, distributions).
3. Initialize DB connection(s).
4. Instantiate builders (concept inventory, mastery sampler, retrieval sampler, observation builder).
5. Loop until target number of observations reached.
6. Write jsonl and print a short summary (concept coverage, mastery bucket coverage, scenario coverage).

---

## 6. Adjustability Knobs

Configs should allow you to tweak:

- **Counts**:
  - total observations
  - per-concept minimum/maximum
- **Scenario mix**:
  - e.g., 40% explain, 20% worked_example, 20% concept_check, 10% reflection, 10% hints
- **Mastery distribution**:
  - e.g., 60% low, 30% medium, 10% high
- **Difficulty bands**:
  - map concept_level ↔ difficulty tags
- **Domain and resources**:
  - which resource IDs to draw chunks from
- **Flags**:
  - `use_real_mastery` vs `use_synthetic_mastery`
  - `include_prereq_review_scenarios`
  - `include_srl_fields` (if you want to embed SRL structures as static hints).

---

## 7. LLM Usage Principles

To keep the pipeline **simple, robust, and efficient**, LLMs are used in a narrow, well-defined way:

- **Where LLMs are used:**
  - Generating realistic student messages for different scenarios and mastery buckets.
  - Optionally generating student answers (correct / near-miss / incorrect) and misconception labels for `concept_check` scenarios.
- **Where LLMs are *not* used:**
  - Concept selection, mastery buckets, and retrieval: these stay DB/KG-driven.
  - Core observation schema and structure.
- **Guardrails:**
  - Always validate LLM outputs (non-empty, length bounds, concept mention when needed).
  - Fall back to deterministic templates on any error or invalid output.
- **Efficiency:**
  - Keep prompts small and focused.
  - Allow running in "no-LLM" mode (templates only) via config flag, so generation can be cheap and fully deterministic when desired.

This keeps LLMs as an **optional quality booster** on top of a deterministic pipeline.

---

## 8. Integration Points

This pipeline is designed to feed directly into existing RL infrastructure:

- `scripts/tutor_rollout_bandit.py` and `/api/rl/rollout` already accept observations that include `payload` and optional observation overrides.
- You can generate separate observation batches per domain, per prompt_set, or per experiment and store them under `datasets/{domain}/{experiment}/`.

---

## 9. Implementation Plan

1. **Start with one domain (e.g., heat transfer)** and implement:
   - `domain_heat_transfer.yaml`
   - `concept_inventory.py` (basic version)
   - `retrieval_sampler.py` (simple tag-based retrieval)
   - `scenario_templates.py` (4–6 scenario types)
   - `observation_builder.py` (MVP)
   - `cli/build_observations.py` (MVP)
2. Generate a small batch (e.g., 1k observations) and inspect manually.
3. Iterate on:
   - mastery sampling strategy
   - scenario distribution
   - retrieval quality heuristics
   - LLM prompt design and validation rules for student messages/answers
4. Once stable, extend to more domains and more sophisticated mastery modelling.
