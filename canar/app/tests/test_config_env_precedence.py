import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_process_environment_overrides_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "QDRANT_COLLECTIONS=dotenv_collection\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "QDRANT_COLLECTIONS": "shell_collection",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from canar.app.config import AppConfig; "
                "print(','.join(AppConfig().qdrant_collections))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "shell_collection"
