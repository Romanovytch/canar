SHELL := /bin/bash

.PHONY: help up down reset logs venv install run test

help:
	@echo "Targets:"
	@echo "  make up      - Start Qdrant + Postgres (docker compose)"
	@echo "  make down    - Stop containers"
	@echo "  make reset   - Stop + remove volumes"
	@echo "  make logs    - Follow logs"
	@echo "  make venv    - Create venv"
	@echo "  make install - Install CanaR (editable)"
	@echo "  make run     - Run CanaR"
	@echo "  make test    - Run tests"

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
	. .venv/bin/activate && pip install -U pip && pip install -e .

run:
	. .venv/bin/activate && canar || streamlit run canar/app/main.py

test:
	. .venv/bin/activate && pytest -q