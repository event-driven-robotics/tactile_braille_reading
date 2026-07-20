"""snn.py

Core spiking neural network layer implementations and neuronal dynamics.

Implements feedforward and recurrent spiking neural network layers with Leaky 
Integrate-and-Fire (LIF) neuron models. Includes surrogate gradient functions 
for differentiable spike generation, straight-through estimator for weight 
quantization, synaptic dynamics, refractory periods, and support for both 
exponential and linear membrane potential decay. Compatible with e-prop and 
BPTT learning algorithms for tactile braille reading applications.

Key Components:
- STEFunction: Straight-through estimator for weight discretization
- SurrGradSpike: Surrogate gradient function for differentiable spike generation
- feedforward_layer: Fully-connected feedforward spiking layer
- recurrent_layer: Fully-connected recurrent spiking layer with temporal dynamics
- compute_winning_neuron: Winner-take-all prediction with random tie-breaking
- run_snn: Forward pass through two-layer spiking neural network

Author: Simon F. Muller-Cleve
Date: January 13, 2026
"""

import torch

from .neuron_models import ste_fn


def expand_input_copies(inputs: torch.Tensor, nb_input_copies: int) -> torch.Tensor:
    """Return the exact presynaptic feature tensor used by the network."""
    copies = int(nb_input_copies)
    if copies < 1:
        raise ValueError(f"nb_input_copies must be >= 1, got {copies}")
    if copies == 1:
        return inputs
    return inputs.tile((1, 1, copies))


def compute_winning_neuron(spk_rec_readout: torch.Tensor, params: dict) -> tuple:
    """
    Compute predictions from output spike counts with random tie-breaking.

    Sums spikes over time, selects the output neuron with the highest spike count
    as the prediction, and randomly breaks ties when multiple neurons have the 
    same maximum spike count. This ensures consistent prediction logic across all
    evaluation contexts (training, testing, confusion matrix, etc.).

    Parameters
    ----------
    spk_rec_readout : torch.Tensor
        Output layer spike recordings with shape [batch, timesteps, output_neurons]
    params : dict
        Dictionary containing experimental parameters:
        - 'random_tie_breaking' : bool
            If True, randomly breaks ties when multiple neurons have equal max spike counts

    Returns
    -------
    tuple
        (spike_counts, predictions) where:
        - spike_counts : torch.Tensor
            Total spike counts per output neuron with shape [batch, output_neurons]
        - predictions : torch.Tensor
            Predicted class indices with shape [batch]

    Notes
    -----
    - Spike counts are summed over the time dimension (dim=1)
    - Winner-take-all: neuron with most spikes wins
    - Ties are broken randomly to avoid systematic bias toward lower-indexed neurons
    - This function is the single source of truth for prediction logic
    - Used consistently in: training, testing, validation, confusion matrix, network activity

    Examples
    --------
    >>> spikes = torch.tensor([[[0, 1], [1, 0], [0, 1]]])  # shape: [1, 3, 2]
    >>> params = {'random_tie_breaking': True}
    >>> summed_spikes, neuron_idc = compute_winning_neuron(spikes, params)
    >>> print(summed_spikes)  # spike counts: tensor([[1, 2]])
    >>> print(neuron_idc)  # prediction: tensor([1])  # neuron 1 has more spikes
    """

    summed_spikes = torch.sum(spk_rec_readout,
                              dim=1)  # sum over time: [batch, output_neurons]

    # Select winner based on spike counts
    max_nb_spikes, neuron_idc = torch.max(
        summed_spikes, dim=1)  # argmax over output units
    if params['random_tie_breaking']:
        # Handle ties: if multiple neurons have the same max spike count, select randomly
        mask = torch.sum(
            summed_spikes == max_nb_spikes.unsqueeze(-1), dim=-1) > 1
        if mask.any():
            true_indices = torch.nonzero(mask, as_tuple=True)[0]
            for i in true_indices:
                candidates = torch.nonzero(
                    summed_spikes[i] == max_nb_spikes[i], as_tuple=True)[0]
                neuron_idc[i] = candidates[torch.randint(
                    0, len(candidates), (1,))]

    return summed_spikes, neuron_idc


def run_snn(inputs: torch.Tensor, layers: list, params: dict) -> tuple:
    """
    Execute forward pass through a spiking neural network.

    This function runs input data through a two-layer spiking neural network (recurrent hidden layer
    and feedforward readout layer) and computes network activity. Gradient computation is handled
    in the training loop based on params["eprop"] setting.

    Parameters
    ----------
    inputs : torch.Tensor
        Input spike trains with shape [batch, time, input_features]
    layers : list
        List containing [recurrent_layer, feedforward_layer] layer objects
    params : dict
        Dictionary containing experimental parameters:
        - 'nb_input_copies' : int
            Number of times to replicate input channels (for increased representation)
        - 'data_steps' : int
            Number of simulation timesteps
        - 'lower_bound' : float or None
            Minimum membrane potential (clamping threshold)

    Returns
    -------
    tuple
        (spk_rec_readout, other_recs) where:
        - spk_rec_readout : torch.Tensor
            Output layer spike recordings [batch, time, output_neurons]
        - other_recs : list
            [mem_rec_hidden, spk_rec_hidden, mem_rec_readout] recordings from layers
            - mem_rec_hidden : torch.Tensor [batch, time, hidden_neurons]
            - spk_rec_hidden : torch.Tensor [batch, time, hidden_neurons]
            - mem_rec_readout : torch.Tensor [batch, time, output_neurons]

    Notes
    -----
    - This function only performs the forward pass
    - Gradient computation (e-prop or BPTT) is handled in the train() function
    - For e-prop mode: uses hard threshold spike generation (no gradient through spikes)
    - For BPTT mode: uses surrogate gradient function for differentiable spike generation
    - Supports input replication via params["nb_input_copies"] for increased input representation
    - Hidden layer uses both feedforward and recurrent connections
    - Readout layer is feedforward-only (no recurrence)
    - n_spike can be computed externally via torch.cumsum(spk_rec_readout, dim=1) when needed
    """
    return_syn = params.get("return_extended_recs", False)
    rec_layer, ff_layer = layers

    # Standard STE quantization path:
    # - keep floating-point master parameters for optimizer updates
    # - use quantized proxy tensors only in the forward pass
    if params.get("quantize_weights", False):
        possible_weights = params.get("possible_weights")
        if possible_weights is None:
            raise ValueError("possible_weights must be set when quantize_weights=True")
        rec_ff_weights = ste_fn(rec_layer.ff_weights, possible_weights)
        rec_rec_weights = ste_fn(rec_layer.rec_weights, possible_weights)
        out_ff_weights = ste_fn(ff_layer.ff_weights, possible_weights)
    else:
        rec_ff_weights = rec_layer.ff_weights
        rec_rec_weights = rec_layer.rec_weights
        out_ff_weights = ff_layer.ff_weights

    expanded_inputs = expand_input_copies(
        inputs, params["nb_input_copies"])
    h1 = torch.einsum(
        "abc,cd->abd", expanded_inputs, rec_ff_weights.t())

    if return_syn:
        spk_rec_hidden, mem_rec_hidden, syn_rec_hidden = rec_layer.compute_activity(
            h1, params['data_steps'], params["lower_bound"], rec_weights=rec_rec_weights, return_syn=return_syn)
    else:
        spk_rec_hidden, mem_rec_hidden = rec_layer.compute_activity(
            h1, params['data_steps'], params["lower_bound"], rec_weights=rec_rec_weights, return_syn=return_syn)
    
    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec_hidden, out_ff_weights.t()))

    if params.get("eprop", False):
        beta_out = float(params.get("beta_mem", ff_layer.beta))
        readout_bias = float(params.get("readout_bias", 0.0))

        mem = torch.zeros(
            (h2.shape[0], h2.shape[2]),
            device=h2.device,
            dtype=h2.dtype,
        )
        mem_rec = []
        for t in range(params['data_steps']):
            mem = beta_out * mem + h2[:, t, :] + readout_bias
            mem_rec.append(mem)

        mem_rec_readout = torch.stack(mem_rec, dim=1)
        # Keep first return value compatible with downstream code paths.
        spk_rec_readout = torch.softmax(mem_rec_readout, dim=2)
        syn_rec_readout = h2
    else:
        if return_syn:
            spk_rec_readout, mem_rec_readout, syn_rec_readout = ff_layer.compute_activity(
                h2, params['data_steps'], params["lower_bound"], return_syn=return_syn)
        else:
            spk_rec_readout, mem_rec_readout = ff_layer.compute_activity(
                h2, params['data_steps'], params["lower_bound"], return_syn=return_syn)

    if return_syn:
        other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout,
                      syn_rec_hidden, syn_rec_readout]
    else:
        other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout]

    return spk_rec_readout, other_recs
