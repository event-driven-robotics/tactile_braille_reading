"""Run the test suite in a single command using pytest.

Purpose
-------
Provides a stable, repository-local entry point for executing the full pytest
suite, independent of whether `pytest` is available on PATH.

How it works
------------
- Uses the current Python interpreter (`sys.executable`).
- Runs `python -m pytest scripts/tests` from repository root.
- Propagates pytest exit code for CI/local automation compatibility.

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
