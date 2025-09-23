SHELL := /bin/bash

.PHONY: start stop logs restart test-backend lint-frontend smoke seed-demo

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
