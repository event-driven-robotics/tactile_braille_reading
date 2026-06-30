"""Deterministic checks for low-level neuron dynamics.

What this file tests
--------------------
1) Feedforward single-neuron step dynamics with fixed input sequence.
2) Recurrent dynamics with explicit recurrent matrix, including verification
     that diagonal self-connections are masked during computation.

How it works
------------
- Builds layers directly from `utils.neuron_models` on CPU with deterministic
    parameters (`eprop=True`, fixed threshold, no synapse decay effects for tests).
- Uses hand-crafted input tensors and expected spike/membrane trajectories.
- Compares outputs with exact tensor equality to catch subtle regression in
    update ordering, thresholding, and recurrence handling.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.neuron_models import feedforward_layer, recurrent_layer


@pytest.mark.smoke
def test_feedforward_known_step_response() -> None:
    """Validate exact spike and membrane trajectory for a 1-neuron feedforward case.

    Expected behavior:
    - First timestep integrates input but does not spike.
    - Second timestep crosses threshold and spikes.
    - Subsequent timesteps remain silent with this setup.
    """
    layer = feedforward_layer(
        nb_inputs=1,
        nb_neurons=1,
        batch_size=1,
        fwd_weight_scale=1.0,
        alpha=0.0,
        beta=0.0,
        eprop=True,
        linear_decay=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
        ref_per=None,
        gamma=15.0,
        spike_threshold=1.0,
        soft_reset=False,
    )

    input_activity = torch.tensor([[[1.2], [0.2], [0.0], [0.0]]], dtype=torch.float64)
    spk_rec, mem_rec = layer.compute_activity(input_activity, nb_steps=4, lower_bound=None)

    expected_spikes = torch.tensor([[[0.0], [1.0], [0.0], [0.0]]], dtype=torch.float64)
    expected_mem = torch.tensor([[[0.0], [1.2], [0.0], [0.0]]], dtype=torch.float64)

    assert torch.equal(spk_rec, expected_spikes), (
        f"Feedforward spikes mismatch\nactual={spk_rec}\nexpected={expected_spikes}"
    )
    assert torch.equal(mem_rec, expected_mem), (
        f"Feedforward membrane mismatch\nactual={mem_rec}\nexpected={expected_mem}"
    )


@pytest.mark.smoke
def test_recurrent_diagonal_mask_and_offdiag_propagation() -> None:
    """Verify recurrent self-connections are masked and off-diagonal propagation works.

    The recurrent matrix intentionally contains strong diagonal values. The test
    ensures those do not induce self-spiking due to masking, while an
    off-diagonal connection still propagates activity to the second neuron.
    """
    layer = recurrent_layer(
        nb_inputs=2,
        nb_neurons=2,
        batch_size=1,
        fwd_weight_scale=1.0,
        rec_weight_scale=1.0,
        alpha=0.0,
        beta=0.0,
        eprop=True,
        linear_decay=False,
        device=torch.device("cpu"),
        dtype=torch.float64,
        ref_per=None,
        gamma=15.0,
        spike_threshold=1.0,
        soft_reset=False,
    )

    rec_weights = torch.tensor(
        [
            [5.0, 0.0],
            [2.0, 5.0],
        ],
        dtype=torch.float64,
    )

    input_activity = torch.tensor(
        [
            [
                [1.2, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        dtype=torch.float64,
    )

    spk_rec, _ = layer.compute_activity(
        input_activity=input_activity,
        nb_steps=5,
        lower_bound=None,
        rec_weights=rec_weights,
    )

    expected_neuron0 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    expected_neuron1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)

    assert torch.equal(spk_rec[0, :, 0], expected_neuron0), (
        f"Recurrent neuron0 spikes mismatch\nactual={spk_rec[0, :, 0]}\nexpected={expected_neuron0}"
    )
    assert torch.equal(spk_rec[0, :, 1], expected_neuron1), (
        f"Recurrent neuron1 spikes mismatch\nactual={spk_rec[0, :, 1]}\nexpected={expected_neuron1}"
    )
