SHELL := /bin/bash

.PHONY: start stop logs restart test-backend lint-frontend smoke seed-demo obs_heat_transfer rollout_heat_transfer

start:
	docker-compose up -d --build

stop:
	docker-compose down

logs:
	docker-compose logs -f --tail=200

restart: stop start

e2e:
	bash scripts/smoke.sh

smoke:
	bash scripts/smoke.sh

test-backend:
	cd backend && pytest -q

lint-frontend:
	cd frontend && npm ci || true && npm run lint

seed-demo:
	bash scripts/seed_demo_data.sh

obs_heat_transfer:
	python scripts/observations/cli/build_observations.py \
	  --domain heat_transfer \
	  --config scripts/observations/config/domain_heat_transfer.yaml \
	  --output datasets/heat_transfer/obs_latest/observations.jsonl

rollout_heat_transfer:
	USE_LLM_MOCK=1 python scripts/tutor_rollout_bandit.py \
	  --observations datasets/heat_transfer/obs_latest/observations.jsonl \
	  --out-dir datasets/heat_transfer/rollout_latest \
	  --candidates 2 \
	  --actions explain,ask \
	  --mock \
	  --seed 123
