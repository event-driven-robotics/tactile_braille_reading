"""Checks that neuron-model encoding couples simulation dt to upsampling."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts import event_transform


@pytest.mark.parametrize(
    ("model_name", "expected_internal_dt"),
    [
        ("AdExLIF_neuron", 0.002),
        ("CuBaLIF_neuron", 0.002),
        ("LIF_neuron", 0.002),
        ("MN_neuron", 0.002),
        ("IZ_neuron", 2.0),
    ],
)
def test_build_neuron_model_uses_upsample_dt(
    model_name: str,
    expected_internal_dt: float,
) -> None:
    """Use upsample dt as constructor dt; IZ stores it internally in ms."""
    model = event_transform.build_neuron_model(
        model_name,
        nb_inputs=2,
        neuron_params={},
        dt_s=0.002,
    )

    assert model.dt == pytest.approx(expected_internal_dt)


def test_build_neuron_model_upsample_dt_overrides_programmatic_dt() -> None:
    """The data sampling interval is the source of truth for neuron dt."""
    model = event_transform.build_neuron_model(
        "MN_neuron",
        nb_inputs=2,
        neuron_params={"dt": 0.01},
        dt_s=0.002,
    )

    assert model.dt == pytest.approx(0.002)


def test_dt_is_not_user_tunable_neuron_param() -> None:
    """Users should change --upsample-dt-s rather than --neuron-param dt=..."""
    defaults = event_transform.get_default_neuron_params("MN_neuron")

    assert "dt" not in defaults
    with pytest.raises(ValueError, match="--upsample-dt-s"):
        event_transform.parse_neuron_param_assignments(["dt=0.01"], "MN_neuron")
