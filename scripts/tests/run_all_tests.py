"""Run the test suite in a single command using pytest.

Usage:
    /home/smullercleve/.virtualenvs/pytorch/bin/python scripts/tests/run_all_tests.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "pytest", "scripts/tests"]
    proc = subprocess.run(cmd, cwd=repo_root)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
