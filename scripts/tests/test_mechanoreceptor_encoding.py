"""Regression tests for the event-based mechanoreceptor encoders."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.neuron_models import RA_I_mechanoreceptor


def test_ra_i_zero_refractory_period_emits_requested_events() -> None:
    """Zero disables rate limiting without causing a division by zero."""
    encoder = RA_I_mechanoreceptor(
        taxel_values=np.array([0.0]),
        fa_threshold=2.0,
        ref_period=0.0,
    )

    events = encoder.step(
        taxel_values=np.array([10.0]),
        current_time=0.01,
        last_time=0.0,
    )

    assert events.shape == (5, 2)
    assert np.all(events[:, 1] == 0)
    assert np.all(np.diff(events[:, 0]) >= 0)
    assert events[-1, 0] == 0.01


def test_ra_i_positive_refractory_period_still_caps_events() -> None:
    """A positive refractory period continues to limit the event count."""
    encoder = RA_I_mechanoreceptor(
        taxel_values=np.array([0.0]),
        fa_threshold=2.0,
        ref_period=0.003,
    )

    events = encoder.step(
        taxel_values=np.array([10.0]),
        current_time=0.01,
        last_time=0.0,
    )

    assert events.shape == (3, 2)
    assert np.all(np.diff(events[:, 0]) >= 0.003)
    assert events[-1, 0] == 0.01
