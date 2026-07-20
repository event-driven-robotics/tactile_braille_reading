"""Equation-level numerical checks for Bellec and Frenkel e-prop.

The suite covers equivalence at zero output leak, the intended hidden-gradient
difference at nonzero output leak, common output gradients, an independent
autograd reference for Bellec, reset and refractory behavior, delayed losses,
recurrent diagonal masking, causal input timing, and replicated inputs.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.snn import expand_input_copies
from utils.train_snn import grads_batch


class _BellecTriangularSpike(torch.autograd.Function):
    """Hard spike with Bellec's triangular pseudo-derivative in backward."""

    @staticmethod
    def forward(ctx, voltage: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.threshold = float(threshold)
        ctx.save_for_backward(voltage)
        return (voltage > threshold).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        voltage, = ctx.saved_tensors
        threshold = ctx.threshold
        support = torch.clamp(
            1.0 - torch.abs((voltage - threshold) / threshold), min=0.0)
        return grad_output * (0.3 / threshold) * support, None


def _run_single_synapse_gradient(
    mode: str,
    *,
    soft_reset: bool,
    x: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    yo: torch.Tensor,
    refractory_mask: torch.Tensor | None = None,
    gamma: float = 2.0,
    beta_mem_rec: float = 0.5,
    beta_mem: float = 0.0,
    delayed_output: int | None = None,
    w_out_value: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Run a one-batch, one-neuron manual gradient calculation."""
    w_in = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    w_rec = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    w_out = torch.full(
        (1, 1), w_out_value, dtype=torch.float64, requires_grad=True)
    params = {
        "nb_hidden": 1,
        "dtype_torch": torch.float64,
        "device": torch.device("cpu"),
        "eprop": True,
        "eprop_mode": mode,
        "delayed_output": delayed_output,
        "gamma": gamma,
        "beta_mem_rec": beta_mem_rec,
        "beta_mem": beta_mem,
        "eprop_lr_layer": (1.0, 1.0, 1.0),
        "soft_reset": soft_reset,
    }

    grads_batch(
        x=x,
        yo=yo,
        yt=torch.zeros((1, 1), dtype=torch.float64),
        thr=1.0,
        v=v,
        z=z,
        w_in=w_in,
        w_rec=w_rec,
        w_out=w_out,
        params=params,
        refractory_mask=refractory_mask,
    )
    return {
        "in": w_in.grad.detach().clone(),
        "rec": w_rec.grad.detach().clone(),
        "out": w_out.grad.detach().clone(),
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("bellec", 0.3), ("frenkel", 2.0)],
)
def test_current_input_contributes_at_current_recorded_voltage(
    mode: str, expected: float
) -> None:
    """x[t] must contribute to eligibility paired with the recorded v[t]."""
    grad = _run_single_synapse_gradient(
        mode,
        soft_reset=True,
        x=torch.ones((1, 1, 1), dtype=torch.float64),
        v=torch.ones((1, 1, 1), dtype=torch.float64),
        z=torch.zeros((1, 1, 1), dtype=torch.float64),
        yo=torch.ones((1, 1, 1), dtype=torch.float64),
    )
    assert grad["in"].item() == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["bellec", "frenkel"])
def test_multiplicative_reset_clears_prior_eligibility(mode: str) -> None:
    """Hard reset contributes the exact stopped-spike (1-z) Jacobian gate."""
    inputs = torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64)
    voltages = torch.ones((2, 1, 1), dtype=torch.float64)
    spikes = torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64)
    errors = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64)

    hard_grad = _run_single_synapse_gradient(
        mode,
        soft_reset=False,
        x=inputs,
        v=voltages,
        z=spikes,
        yo=errors,
    )
    soft_grad = _run_single_synapse_gradient(
        mode,
        soft_reset=True,
        x=inputs,
        v=voltages,
        z=spikes,
        yo=errors,
    )

    assert hard_grad["in"].item() == pytest.approx(0.0)
    assert soft_grad["in"].item() > 0.0


@pytest.mark.parametrize("mode", ["bellec", "frenkel"])
def test_refractory_mask_zeros_spike_derivative(mode: str) -> None:
    """No hidden-weight update is emitted while the neuron is refractory."""
    grad = _run_single_synapse_gradient(
        mode,
        soft_reset=True,
        x=torch.ones((1, 1, 1), dtype=torch.float64),
        v=torch.ones((1, 1, 1), dtype=torch.float64),
        z=torch.zeros((1, 1, 1), dtype=torch.float64),
        yo=torch.ones((1, 1, 1), dtype=torch.float64),
        refractory_mask=torch.ones((1, 1, 1), dtype=torch.bool),
    )
    assert grad["in"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["bellec", "frenkel"])
def test_masked_recurrent_diagonal_gradient_is_zero(mode: str) -> None:
    """Manual gradients respect the forward pass's no-self-connection mask."""
    w_in = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    w_rec = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    w_out = torch.ones((1, 1), dtype=torch.float64, requires_grad=True)
    params = {
        "nb_hidden": 1,
        "dtype_torch": torch.float64,
        "device": torch.device("cpu"),
        "eprop": True,
        "eprop_mode": mode,
        "delayed_output": None,
        "gamma": 2.0,
        "beta_mem_rec": 0.5,
        "beta_mem": 0.0,
        "eprop_lr_layer": (1.0, 1.0, 1.0),
        "soft_reset": True,
    }

    grads_batch(
        x=torch.zeros((2, 1, 1), dtype=torch.float64),
        yo=torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64),
        yt=torch.zeros((1, 1), dtype=torch.float64),
        thr=1.0,
        v=torch.ones((2, 1, 1), dtype=torch.float64),
        z=torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64),
        w_in=w_in,
        w_rec=w_rec,
        w_out=w_out,
        params=params,
    )
    assert w_rec.grad.item() == pytest.approx(0.0)


def test_bellec_and_frenkel_hidden_gradients_match_when_kappa_is_zero() -> None:
    """Without output filtering, the two hidden-weight rules are identical."""
    common = {
        "soft_reset": True,
        "x": torch.tensor([[[1.0]], [[0.4]], [[0.2]]], dtype=torch.float64),
        "v": torch.tensor([[[1.0]], [[0.8]], [[1.2]]], dtype=torch.float64),
        "z": torch.tensor([[[0.0]], [[1.0]], [[0.0]]], dtype=torch.float64),
        "yo": torch.tensor([[[0.2]], [[0.7]], [[0.4]]], dtype=torch.float64),
        "gamma": 0.3,
        "beta_mem_rec": 0.6,
        "beta_mem": 0.0,
    }
    bellec = _run_single_synapse_gradient("bellec", **common)
    frenkel = _run_single_synapse_gradient("frenkel", **common)

    assert torch.allclose(bellec["in"], frenkel["in"], atol=1e-12)
    assert torch.allclose(bellec["out"], frenkel["out"], atol=1e-12)


def test_nonzero_kappa_preserves_history_only_in_bellec_hidden_gradient() -> None:
    """Bellec filters a past psi*p product that Frenkel intentionally omits."""
    common = {
        "soft_reset": True,
        "x": torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64),
        "v": torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float64),
        "z": torch.zeros((2, 1, 1), dtype=torch.float64),
        "yo": torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float64),
        "gamma": 0.3,
        "beta_mem_rec": 0.0,
        "beta_mem": 0.5,
    }
    bellec = _run_single_synapse_gradient("bellec", **common)
    frenkel = _run_single_synapse_gradient("frenkel", **common)

    assert bellec["in"].item() == pytest.approx(0.15)
    assert frenkel["in"].item() == pytest.approx(0.0)


def test_output_gradients_match_between_modes_for_nonzero_kappa() -> None:
    """Both variants retain the same filtered hidden-spike output trace."""
    common = {
        "soft_reset": True,
        "x": torch.zeros((3, 1, 1), dtype=torch.float64),
        "v": torch.ones((3, 1, 1), dtype=torch.float64),
        "z": torch.tensor([[[1.0]], [[0.0]], [[1.0]]], dtype=torch.float64),
        "yo": torch.tensor([[[0.2]], [[0.4]], [[0.7]]], dtype=torch.float64),
        "gamma": 0.3,
        "beta_mem_rec": 0.6,
        "beta_mem": 0.7,
    }
    bellec = _run_single_synapse_gradient("bellec", **common)
    frenkel = _run_single_synapse_gradient("frenkel", **common)

    assert torch.allclose(bellec["out"], frenkel["out"], atol=1e-12)


def test_bellec_matches_autograd_stopped_reset_reference() -> None:
    """Compare Bellec's recursion with a tiny surrogate-autograd graph."""
    threshold = 1.0
    beta_hidden = 0.6
    kappa = 0.7
    w_out_value = 0.8
    inputs = torch.tensor([1.0, 0.4, 0.2], dtype=torch.float64)
    w_in_reference = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)

    post_reset = torch.zeros((), dtype=torch.float64)
    readout = torch.zeros((), dtype=torch.float64)
    voltages = []
    spikes = []
    readouts = []
    for x_t in inputs:
        voltage = beta_hidden * post_reset + w_in_reference * x_t
        spike = _BellecTriangularSpike.apply(voltage, threshold)
        post_reset = voltage - spike.detach() * threshold
        readout = kappa * readout + w_out_value * spike
        voltages.append(voltage)
        spikes.append(spike)
        readouts.append(readout)

    loss = 0.5 * sum(value.square() for value in readouts)
    loss.backward()

    bellec = _run_single_synapse_gradient(
        "bellec",
        soft_reset=True,
        x=inputs.reshape(-1, 1, 1),
        v=torch.stack(voltages).detach().reshape(-1, 1, 1),
        z=torch.stack(spikes).detach().reshape(-1, 1, 1),
        yo=torch.stack(readouts).detach().reshape(-1, 1, 1),
        gamma=99.0,
        beta_mem_rec=beta_hidden,
        beta_mem=kappa,
        w_out_value=w_out_value,
    )

    assert bellec["in"].item() == pytest.approx(
        w_in_reference.grad.item(), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ["bellec", "frenkel"])
def test_delayed_output_limits_error_times_but_keeps_trace_history(mode: str) -> None:
    """A final-step loss still receives eligibility accumulated beforehand."""
    common = {
        "soft_reset": True,
        "x": torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64),
        "v": torch.ones((2, 1, 1), dtype=torch.float64),
        "z": torch.zeros((2, 1, 1), dtype=torch.float64),
        "yo": torch.ones((2, 1, 1), dtype=torch.float64),
        "gamma": 0.3,
        "beta_mem_rec": 0.5,
        "beta_mem": 0.0,
    }
    full = _run_single_synapse_gradient(mode, delayed_output=None, **common)
    delayed = _run_single_synapse_gradient(mode, delayed_output=1, **common)

    assert full["in"].item() == pytest.approx(0.45)
    assert delayed["in"].item() == pytest.approx(0.15)


@pytest.mark.parametrize("mode", ["bellec", "frenkel"])
def test_replicated_inputs_produce_matching_gradient_columns(mode: str) -> None:
    """Manual gradients use the exact feature expansion from the forward pass."""
    base_input = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64)
    expanded = expand_input_copies(base_input, 2)
    assert torch.equal(
        expanded, torch.tensor([[[2.0, 4.0, 2.0, 4.0]]], dtype=torch.float64))

    w_in = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    w_rec = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    w_out = torch.ones((1, 1), dtype=torch.float64, requires_grad=True)
    params = {
        "nb_hidden": 1,
        "dtype_torch": torch.float64,
        "device": torch.device("cpu"),
        "eprop": True,
        "eprop_mode": mode,
        "delayed_output": None,
        "gamma": 0.3,
        "beta_mem_rec": 0.5,
        "beta_mem": 0.0,
        "eprop_lr_layer": (1.0, 1.0, 1.0),
        "soft_reset": True,
    }
    grads_batch(
        x=expanded.permute(1, 0, 2),
        yo=torch.ones((1, 1, 1), dtype=torch.float64),
        yt=torch.zeros((1, 1), dtype=torch.float64),
        thr=1.0,
        v=torch.ones((1, 1, 1), dtype=torch.float64),
        z=torch.zeros((1, 1, 1), dtype=torch.float64),
        w_in=w_in,
        w_rec=w_rec,
        w_out=w_out,
        params=params,
    )

    expected = torch.tensor([[0.6, 1.2, 0.6, 1.2]], dtype=torch.float64)
    assert torch.allclose(w_in.grad, expected, atol=1e-12)
