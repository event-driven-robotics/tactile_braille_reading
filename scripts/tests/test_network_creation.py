"""Network construction and forward-pass smoke checks."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.neuron_models import feedforward_layer, recurrent_layer
from utils.snn import run_snn


def _build_layers(batch_size: int, input_dim: int, hidden_dim: int, output_dim: int, eprop: bool) -> list:
    rec = recurrent_layer(
        nb_inputs=input_dim,
        nb_neurons=hidden_dim,
        batch_size=batch_size,
        fwd_weight_scale=0.5,
        rec_weight_scale=0.2,
        alpha=0.0,
        beta=0.9,
        eprop=eprop,
        linear_decay=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
        ref_per=None,
        gamma=15.0,
        spike_threshold=1.0,
        soft_reset=False,
    )
    ff = feedforward_layer(
        nb_inputs=hidden_dim,
        nb_neurons=output_dim,
        batch_size=batch_size,
        fwd_weight_scale=0.5,
        alpha=0.0,
        beta=0.9,
        eprop=eprop,
        linear_decay=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
        ref_per=None,
        gamma=15.0,
        spike_threshold=1.0,
        soft_reset=False,
    )
    return [rec, ff]


def _assert_finite(t: torch.Tensor, name: str) -> None:
    assert torch.isfinite(t).all(), f"Non-finite values detected in {name}"


@pytest.mark.smoke
def test_bptt_forward_shapes() -> None:
    batch_size, steps, input_dim, hidden_dim, output_dim = 3, 6, 4, 5, 2
    layers = _build_layers(batch_size, input_dim, hidden_dim, output_dim, eprop=False)

    inputs = torch.randn(batch_size, steps, input_dim, dtype=torch.float64)
    params = {
        "nb_input_copies": 1,
        "data_steps": steps,
        "lower_bound": None,
        "eprop": False,
        "quantize_weights": False,
    }

    readout, recs = run_snn(inputs=inputs, layers=layers, params=params)
    mem_hidden, spk_hidden, mem_readout = recs

    assert readout.shape == (batch_size, steps, output_dim)
    assert mem_hidden.shape == (batch_size, steps, hidden_dim)
    assert spk_hidden.shape == (batch_size, steps, hidden_dim)
    assert mem_readout.shape == (batch_size, steps, output_dim)

    _assert_finite(readout, "bptt readout")
    _assert_finite(mem_hidden, "bptt hidden membrane")


def test_eprop_quantized_forward_shapes() -> None:
    batch_size, steps, input_dim, hidden_dim, output_dim = 2, 5, 3, 4, 2
    layers = _build_layers(batch_size, input_dim, hidden_dim, output_dim, eprop=True)

    inputs = torch.randn(batch_size, steps, input_dim, dtype=torch.float64)
    possible_weights = torch.linspace(-1.0, 1.0, steps=9, dtype=torch.float64)
    params = {
        "nb_input_copies": 1,
        "data_steps": steps,
        "lower_bound": None,
        "eprop": True,
        "quantize_weights": True,
        "possible_weights": possible_weights,
        "beta_mem": 0.9,
        "readout_bias": 0.0,
    }

    readout, recs = run_snn(inputs=inputs, layers=layers, params=params)
    _, spk_hidden, mem_readout = recs

    assert readout.shape == (batch_size, steps, output_dim)
    assert spk_hidden.shape == (batch_size, steps, hidden_dim)
    assert mem_readout.shape == (batch_size, steps, output_dim)

    probs_sum = readout.sum(dim=2)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-8), (
        "E-prop readout should be softmax probabilities summing to 1"
    )

    _assert_finite(readout, "eprop readout")

