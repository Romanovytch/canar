SHELL := /bin/bash

.PHONY: help up down reset logs venv install run test lint format format-check ci

help:
	@echo "Targets:"
	@echo "  make up            - Start Qdrant + Postgres (docker compose)"
	@echo "  make down          - Stop containers"
	@echo "  make reset         - Stop + remove volumes"
	@echo "  make logs          - Follow docker logs"
	@echo "  make venv          - Create venv (.venv)"
	@echo "  make install       - Install CanaR (editable) + dev tools"
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
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

run:
	. .venv/bin/activate && (canar || streamlit run canar/app/main.py --server.headless true --server.port 8501)

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff format . && ruff check . --fix

format-check:
	. .venv/bin/activate && ruff format --check .

ci: lint format-check test