"""Checks that e-prop is only used with multiplicative exponential decay."""

from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.train_snn import _validate_eprop_decay_mode


@pytest.mark.parametrize("mode", ["frenkel", "bellec"])
def test_eprop_rejects_linear_decay_and_logs_error(mode: str, caplog) -> None:
    params = {"eprop": True, "eprop_mode": mode, "linear_decay": True}

    with caplog.at_level(logging.ERROR, logger="braille_training"):
        with pytest.raises(ValueError, match="requires exponential membrane decay"):
            _validate_eprop_decay_mode(params)

    assert "Linear decay is not supported" in caplog.text


@pytest.mark.parametrize("mode", ["frenkel", "bellec"])
def test_eprop_exponential_decay_is_logged(mode: str, caplog) -> None:
    params = {"eprop": True, "eprop_mode": mode, "linear_decay": False}

    with caplog.at_level(logging.INFO, logger="braille_training"):
        _validate_eprop_decay_mode(params, log_success=True)

    assert f"E-prop mode '{mode}' is using exponential membrane decay" in caplog.text


def test_bptt_still_allows_linear_decay(caplog) -> None:
    params = {"eprop": False, "linear_decay": True}

    with caplog.at_level(logging.INFO, logger="braille_training"):
        _validate_eprop_decay_mode(params)

    assert "E-prop mode" not in caplog.text


@pytest.mark.parametrize("mode", ["frenkel", "bellec"])
def test_eprop_rejects_unimplemented_synaptic_state(mode: str) -> None:
    params = {
        "eprop": True,
        "eprop_mode": mode,
        "linear_decay": False,
        "synapse": True,
    }
    with pytest.raises(ValueError, match="two-state synaptic-current"):
        _validate_eprop_decay_mode(params)
