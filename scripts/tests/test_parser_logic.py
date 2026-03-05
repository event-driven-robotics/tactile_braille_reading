"""Checks parser normalization and resume-path helper behavior.

What this file tests
--------------------
1) CLI alias normalization (`experimental`/`traditional` -> canonical names).
2) Boolean parsing paths for inference mode and artifact mode parsing.
3) Resume path resolution helper for selecting model and params files.

How it works
------------
- Imports the training script with isolated temporary output paths so tests do
    not write into real workspace result/model/log folders.
- Temporarily overrides `sys.argv` to exercise `parse_arguments()` exactly as a
    CLI invocation would.
- Uses temporary directory trees to validate resume-path helper behavior.
"""

from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_module_with_isolated_paths(temp_root: Path):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "braille_reading_rsnn.py",
            "--fig_path", str(temp_root / "figures"),
            "--model_path", str(temp_root / "model"),
            "--results_path", str(temp_root / "results"),
            "--log_path", str(temp_root / "logs"),
        ]
        br = importlib.import_module("scripts.braille_reading_rsnn")
        br = importlib.reload(br)
        return br
    finally:
        sys.argv = old_argv


@pytest.fixture
def br_module(tmp_path: Path):
    return _load_module_with_isolated_paths(tmp_path)


def _parse_with_argv(br, argv: list[str]) -> dict:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["braille_reading_rsnn.py", *argv]
        return br.parse_arguments()
    finally:
        sys.argv = old_argv


@pytest.mark.smoke
def test_mode_alias_normalization(br_module) -> None:
    """Ensure legacy e-prop mode aliases map to canonical internal values."""
    args_exp = _parse_with_argv(br_module, ["--eprop_mode", "experimental"])
    args_trad = _parse_with_argv(br_module, ["--eprop_mode", "traditional"])
    assert args_exp["eprop_mode"] == "frenkel"
    assert args_trad["eprop_mode"] == "bellec"


def test_inference_bool_and_record_mode_parse(br_module) -> None:
    """Verify inference flag booleans and `save_artifacts_for` parse as intended."""
    args_true = _parse_with_argv(br_module, ["--inference-only"])
    args_false = _parse_with_argv(br_module, ["--inference_only=false", "--save_artifacts_for", "none"])

    assert args_true["inference_only"] is True
    assert args_false["inference_only"] is False
    assert args_false["save_artifacts_for"] == "none"


def test_resolve_resume_paths(br_module, tmp_path: Path) -> None:
    """Validate resume helper finds best-model file and parameters JSON path."""
    run_id = "run_x"
    model_dir = tmp_path / "model" / run_id
    results_dir = tmp_path / "results" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "best_model_10_neurons_A_B_rep_1.pt"
    model_file.write_bytes(b"dummy")
    (results_dir / "experiment_parameters.json").write_text("{}")

    params = {
        "model_path": str(tmp_path / "model"),
        "results_path": str(tmp_path / "results"),
        "resume_model_name": "",
    }

    resume_from, resume_params = br_module._resolve_resume_paths(run_id, params)

    assert Path(resume_from).name == model_file.name
    assert Path(resume_params).name == "experiment_parameters.json"
