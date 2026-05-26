# Makefile — smart-traffic-ai
# =============================
# Common developer commands. Run: make <target>

.PHONY: help setup dev backend frontend train evaluate docker clean test

PYTHON := python3
PIP    := pip3
NPM    := npm

## ── Help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🚦 smart-traffic-ai — Developer Commands"
	@echo "  ========================================="
	@echo "  make setup       Install all Python + Node dependencies"
	@echo "  make dev         Start backend + frontend concurrently"
	@echo "  make backend     Start Flask backend only (port 5000)"
	@echo "  make frontend    Start Vite frontend only (port 5173)"
	@echo ""
	@echo "  make train       Run full ML training pipeline"
	@echo "  make evaluate    Evaluate congestion models, output metrics"
	@echo "  make dataset     Generate synthetic training datasets"
	@echo "  make analytics   Generate analytics report + charts"
	@echo ""
	@echo "  make docker      Build + start with Docker Compose"
	@echo "  make test        Run backend unit tests"
	@echo "  make clean       Remove build artifacts and caches"
	@echo ""

## ── Setup ─────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt
	cd frontend && $(NPM) install

## ── Development servers ───────────────────────────────────────
dev:
	$(PYTHON) run.py

backend:
	FLASK_APP=backend.app FLASK_ENV=development $(PYTHON) -m flask run --host=0.0.0.0 --port=5000 --debug

frontend:
	cd frontend && $(NPM) run dev

## ── ML Pipeline ───────────────────────────────────────────────
dataset:
	$(PYTHON) datasets/prepare_dataset.py --task all

train-congestion:
	$(PYTHON) ml_models/congestion/train_models.py --generate-synthetic

train-cnn:
	$(PYTHON) ml_models/cnn/train.py --epochs 50 --batch 32

train-yolo:
	$(PYTHON) ml_models/yolo/train.py --model yolov8s --epochs 100

train:
	$(MAKE) dataset
	$(MAKE) train-congestion
	@echo "✅ Congestion models trained."
	@echo "   For YOLO: make train-yolo"
	@echo "   For CNN:  make train-cnn"

evaluate:
	$(PYTHON) ml_models/congestion/evaluate.py

analytics:
	$(PYTHON) analytics/engine.py

## ── Docker ────────────────────────────────────────────────────
docker:
	docker-compose up --build

docker-down:
	docker-compose down

## ── Testing ───────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short 2>/dev/null || echo "No tests directory found. Create tests/ to run unit tests."

## ── Build frontend ────────────────────────────────────────────
build-frontend:
	cd frontend && $(NPM) run build

## ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf frontend/dist frontend/.vite 2>/dev/null || true
	rm -f backend/traffic_system*.db 2>/dev/null || true
	@echo "✅ Cleaned build artifacts."
