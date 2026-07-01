SHELL := /bin/bash

VENV := .venv_canar
PIP_CACHE_DIR := /mnt/backup/cereq/pip-cache
PIP_TMP_DIR := /mnt/backup/cereq/pip-tmp

.PHONY: help up down reset logs venv install install-torch run test lint format format-check ci

help:
	@echo "Targets:"
	@echo "  make up            - Start Qdrant + Postgres (docker compose)"
	@echo "  make down          - Stop containers"
	@echo "  make reset         - Stop + remove volumes"
	@echo "  make logs          - Follow docker logs"
	@echo "  make venv          - Create venv (.venv_canar)"
	@echo "  make install       - Install CanaR (editable) + dev tools"
	@echo "  make install-torch - Install CUDA 12.8 PyTorch wheel"
	@echo "  make run           - Run CanaR (entrypoint or Streamlit fallback)"
	@echo "  make test          - Run tests (pytest)"
	@echo "  make lint          - Run ruff lint (check)"
	@echo "  make format        - Auto-format with ruff"
	@echo "  make format-check  - Check formatting with ruff"
	@echo "  make ci            - Run lint + format-check + tests"

up:
	cd infra && docker compose up -d

down:
	cd infra && docker compose down

reset:
	cd infra && docker compose down -v

logs:
	cd infra && docker compose logs -f --tail=200

venv:
	python -m venv $(VENV)

install:
	mkdir -p $(PIP_CACHE_DIR) $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) \
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) \
	$(VENV)/bin/pip install -U pip
	TMPDIR=$(PIP_TMP_DIR) \
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) \
	$(VENV)/bin/pip install -e ".[dev]"

install-torch:
	mkdir -p $(PIP_CACHE_DIR) $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) \
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) \
	$(VENV)/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torch>=2.6,<3.0"

install-rerank:
	mkdir -p $(PIP_CACHE_DIR) $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) \
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) \
	$(VENV)/bin/pip install -e ".[rerank]"

run:
	. $(VENV)/bin/activate && (canar || streamlit run canar/app/main.py --server.headless true --server.port 8530)

test:
	. $(VENV)/bin/activate && pytest -q

lint:
	. $(VENV)/bin/activate && ruff check .

format:
	. $(VENV)/bin/activate && ruff format . && ruff check . --fix

format-check:
	. $(VENV)/bin/activate && ruff format --check .

ci: lint format-check test
