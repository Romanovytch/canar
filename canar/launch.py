# canar/launch.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit.web.cli as stcli

from canar.app.utils.utils_logger import sys_logger

logger = logging.getLogger("CanarLauncher")


def main():
    sys_logger.info("Application démarrée")
    here = Path(__file__).resolve().parent
    app_path = here / "app" / "main.py"
    if not app_path.exists():
        logger.error(f"[canar] Missing {app_path}")
        sys.exit(1)

    argv = ["streamlit", "run", str(app_path)]
    if os.getenv("CANAR_HEADLESS", "true").lower() in {"1", "true", "yes"}:
        argv += ["--server.headless", "true"]
    if port := os.getenv("CANAR_PORT"):
        argv += ["--server.port", port]

    sys.argv = argv
    sys.exit(stcli.main())
