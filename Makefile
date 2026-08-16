PYTHON ?= python3.11
UV ?= uv
API_PORT ?= 8102
UI_PORT ?= 3102
MLFLOW_PORT ?= 5102
export PYTHONPATH := src:$(PYTHONPATH)
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export KMP_DUPLICATE_LIB_OK := TRUE
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export KMP_DUPLICATE_LIB_OK := TRUE

GPU_CONFIG ?= configs/cuda/large.yaml
.PHONY: setup data smoke train evaluate test lint typecheck serve ui docker-up docker-down benchmark docs stream-demo drift-demo
.PHONY: gpu-setup gpu-check gpu-data gpu-train gpu-eval gpu-serve

setup:
	$(UV) python pin 3.11
	$(UV) sync --extra dev --no-cache
	@mkdir -p artifacts data/generated
	@touch artifacts/.gitkeep data/generated/.gitkeep

data:
	$(UV) run python scripts/generate_data.py --n-transactions 8000 --seed 42 --out data/generated/transactions.parquet

train:
	$(UV) run python scripts/train.py --config configs/model.yaml

evaluate:
	$(UV) run python scripts/evaluate.py --config configs/model.yaml

smoke:
	$(UV) run python scripts/generate_data.py --n-transactions 6000 --n-customers 400 --n-merchants 80 --seed 42 --out data/generated/transactions.parquet
	$(UV) run python scripts/train.py --config configs/model.yaml --n-optuna-trials 4
	$(UV) run python scripts/evaluate.py --config configs/model.yaml
	$(UV) run python scripts/collect_environment.py
	$(UV) run pytest tests/unit tests/data tests/model -q --tb=short

test:
	$(UV) run pytest tests -q --tb=short

lint:
	$(UV) run ruff check src scripts tests
	$(UV) run ruff format --check src scripts tests

typecheck:
	$(UV) run mypy src/sentinelml

serve:
	$(UV) run uvicorn sentinelml.api.app:app --host 0.0.0.0 --port $(API_PORT)

ui:
	cd dashboard && npm install && npm run dev -- --port $(UI_PORT)

docker-up:
	@echo "Docker daemon was DOWN at discovery. Attempting compose anyway."
	docker compose up -d --build || echo "BLOCKER: Docker daemon unavailable. Kafka was not started."

docker-down:
	docker compose down || true

benchmark:
	$(UV) run python scripts/benchmark.py --requests 200 --concurrency 8

docs:
	@echo "Docs live in README.md ARCHITECTURE.md docs/"

stream-demo:
	$(UV) run python scripts/stream_demo.py --rate 100 --n-events 500 --backend memory

drift-demo:
	$(UV) run python scripts/drift_demo.py

collect-env:
	$(UV) run python scripts/collect_environment.py

gpu-setup:
	$(UV) python pin 3.11
	$(UV) sync --python 3.11 --extra dev
	@echo "If make gpu-check fails, install a CUDA torch wheel. See GPU_SETUP.md."

gpu-check:
	$(UV) run python scripts/check_gpu.py

gpu-data: gpu-check
	$(UV) run python scripts/generate_data.py --n-transactions 200000 --n-customers 4000 --n-merchants 400 --n-days 180 --seed 42 --out data/generated/gpu_transactions.parquet

gpu-data-full: gpu-check
	$(UV) run python scripts/generate_data.py --n-transactions 1000000 --n-customers 12000 --n-merchants 800 --n-days 365 --seed 42 --out data/generated/gpu_transactions_full.parquet

gpu-train: gpu-check
	$(UV) run python scripts/train.py --config $(GPU_CONFIG)

gpu-eval: gpu-check
	$(UV) run python scripts/evaluate.py --config $(GPU_CONFIG)

gpu-serve: gpu-check
	MODEL_DIR=artifacts/gpu SENTINELML_DEVICE=cuda \
		$(UV) run uvicorn sentinelml.api.app:app --host 0.0.0.0 --port $(API_PORT)
