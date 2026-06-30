"""Put the harness dir on sys.path so tests import its modules the same way
eval_e2e.py does (the harness is not an installed package)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/harness/
