"""Put the harness dir and the repo root on sys.path so tests import the same way
eval_e2e.py does (neither the harness nor canar is installed in this venv)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/harness/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the canar repo root
