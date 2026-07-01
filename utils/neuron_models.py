"""neuron_models.py

Spiking neuron models and mechanoreceptor implementations for event-based processing.

This module provides classes and functions for simulating various neuron and mechanoreceptor models,
including Leaky Integrator (LI), Current-Based Leaky Integrator (CuBaLI), Leaky Integrate-and-Fire (LIF),
Current-Based Leaky Integrate-and-Fire (CuBaLIF), and their recurrent variants. It also includes event-based
mechanoreceptor models (RA-I and SA-II) and a fascicle response model for aggregating and processing
spike events through configurable neuron populations.

The models are implemented using PyTorch for efficient computation and support both feedforward and recurrent
architectures, surrogate gradient learning, and event-based input processing. Utility functions for visualizing
network connectivity are also provided.

Classes:
    - SurrGradSpike: Surrogate gradient spiking nonlinearity for PyTorch autograd.
    - LI, CuBaLI, LIF, CuBaLIF, RLIF, CuBaRLIF: Neuron layer models.
    - RA_I_mechanoreceptor, SA_II_mechanoreceptor: Event-based mechanoreceptor models.
    - fascicle_response: Aggregates and processes events through neuron populations.
    - Utility functions for network visualization.

Author: Simon F. Muller-Cleve
Date: January 12, 2026
"""

from collections import namedtuple
import logging
from typing import cast

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# NOTE: one can use device=self.device, dtype=input.dtype to infer the device and dtype from the input tensor


class STEFunction(torch.autograd.Function):
    """
    Here we define the Straight-Through Estimator (STE) function.
    This function allows us to ignore the non-differentiable part
    in our network, i.e. the discretization of the weights.
    The function applies the discretization and the clamping.
    """
    @staticmethod
    def forward(ctx, input, possible_weight_values):
        diffs = torch.abs(input.unsqueeze(-1) - possible_weight_values)
        min_indices = torch.argmin(diffs, dim=-1)
        return possible_weight_values[min_indices]

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Straight-through estimator: gradient passes through unchanged
        grad_output = grad_outputs[0]
        return grad_output.clone(), None


ste_fn = STEFunction.apply


class SurrGradSpike(torch.autograd.Function):
    """
    Spiking nonlinearity with surrogate gradient for differentiable spike generation.

    Implements a step function in the forward pass (binary spike output) with a
    surrogate gradient in the backward pass for gradient-based learning. Uses the
    normalized negative part of a fast sigmoid as surrogate gradient, following
    Zenke & Ganguli (2018).

    The scale parameter controls the steepness of the surrogate gradient:
    - Higher scale: steeper gradient (sharper spike threshold)
    - Lower scale: smoother gradient (more gradual changes)

    Class Attributes
    ----------------
    scale : float
        Surrogate gradient scale factor (default: 15.0). Controls the steepness
        of the surrogate gradient in the backward pass.
    threshold : float
        Spike threshold (default: 0). Membrane potential above this value triggers
        a spike (output = 1.0).
    """

    @staticmethod
    def forward(ctx, input, scale=None, threshold=None):
        """
        Forward pass: compute step function of input.

        In the forward pass, computes a step function where spikes occur when the
        input exceeds the threshold. The context saves the input and scale for
        use in the backward pass.

        Parameters
        ----------
        input : torch.Tensor
            Membrane potential tensor (any shape).
        scale : float, optional
            Surrogate gradient scale factor. If provided, overrides the class
            attribute for this forward-backward pair. If None, uses the class
            attribute value.
        threshold : float, optional
            Spike threshold. If provided, overrides the class attribute.
            If None, uses the class attribute value.

        Returns
        -------
        torch.Tensor
            Binary spike output (1.0 where input > threshold, 0.0 elsewhere),
            same shape as input.
        """
        # Store scale for backward pass if provided
        ctx.scale = scale

        # Use provided threshold or class default
        thr = threshold

        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass: compute surrogate gradient.

        Uses the normalized negative part of a fast sigmoid as surrogate gradient:
            grad = grad_output / (scale * |input| + 1)^2

        This allows gradient flow through the non-differentiable spike function,
        enabling backpropagation-based learning.

        Parameters
        ----------
        grad_outputs : tuple[torch.Tensor]
            Gradients from the next layer (for this function, one tensor matching
            the forward output shape).

        Returns
        -------
        tuple
            (grad_input, grad_scale) where:
            - grad_input : gradient w.r.t. input (same shape as input)
            - grad_scale : None (scale parameter is not differentiable)
        """
        grad_output = grad_outputs[0]
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None, None  # Return None for scale and threshold gradients


def spike_fn(input, scale=15.0, threshold=1.0):
    """
    Apply SurrGradSpike function with optional scale and threshold parameters.

    Parameters
    ----------
    input : torch.Tensor
        Membrane potential tensor.
    scale : float, optional
        Surrogate gradient scale factor. If None, uses default class attribute.
    threshold : float, optional
        Spike threshold. If None, uses default class attribute.

    Returns
    -------
    torch.Tensor
        Binary spike output.

    Examples
    --------
    >>> mem = torch.randn(32, 100)
    >>> spikes = spike_fn(mem)  # Use default scale (15.0) and threshold (1.0)
    >>> spikes = spike_fn(mem, scale=20.0, threshold=0.5)  # Use custom parameters
    """
    return SurrGradSpike.apply(input, scale, threshold)


class feedforward_layer:
    """
    Spiking feedforward layer with Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a fully-connected feedforward layer of spiking neurons with
    optional synaptic dynamics, exponential or linear membrane potential decay, and
    optional refractory period. Supports both e-prop and BPTT learning via surrogate gradients.

    Attributes
    ----------
    nb_inputs : int
        Number of input features/channels
    nb_neurons : int
        Number of neurons in this layer
    batch_size : int
        Maximum batch size for pre-allocation
    fwd_weight_scale : float
        Weight initialization scale factor
    alpha : float
        Synaptic current decay factor (0 for no synapse, or exp(-dt/tau_syn))
    beta : float
        Membrane potential decay factor (exp(-dt/tau_mem) or linear decay rate)
    eprop : bool
        Whether to use e-prop (True) or BPTT (False)
    linear_decay : bool
        Use linear decay instead of exponential decay
    device : str
        Device for tensor allocation (e.g., "cuda:0", "cpu")
    dtype : torch.dtype
        Data type for tensors
    ref_per : int or None
        Refractory period duration in timesteps (None to disable)
    ff_weights : torch.Tensor
        Feedforward weight matrix [nb_neurons, nb_inputs]
    ref_per_tensor : torch.Tensor
        Tracks remaining refractory period timesteps per neuron [batch_size, nb_neurons]

    Notes
    -----
    - Weights initialized with Gaussian distribution N(0, fwd_weight_scale/sqrt(nb_inputs))
    - Spike generation uses hard threshold for e-prop, surrogate gradient for BPTT
    - Refractory period prevents neurons from spiking during cooldown
    """

    def __init__(self, nb_inputs, nb_neurons, batch_size, fwd_weight_scale, alpha, beta, weight_variance=None, eprop=False, linear_decay=False, device=torch.device("cuda:0"), dtype=torch.float64, ref_per=None, gamma=None, spike_threshold=1.0, soft_reset=False):
        """
        Initialize feedforward spiking layer.
        # DEBUG: Log device and dtype
        logger.debug(f"Initializing feedforward_layer: device={device}, dtype={dtype}")
        # Warn for unusual parameters
        if nb_neurons <= 0:
            logger.warning(f"Feedforward layer initialized with non-positive number of neurons: {nb_neurons}")
        if nb_inputs <= 0:
            logger.warning(f"Feedforward layer initialized with non-positive number of inputs: {nb_inputs}")
        if fwd_weight_scale <= 0:
            logger.warning(f"Feedforward layer initialized with non-positive weight scale: {fwd_weight_scale}")
        if batch_size <= 0:
            logger.warning(f"Feedforward layer initialized with non-positive batch size: {batch_size}")

        Parameters
        ----------
        nb_inputs : int
            Number of input features/channels
        nb_neurons : int
            Number of neurons in this layer
        batch_size : int
            Maximum batch size for memory pre-allocation
        fwd_weight_scale : float
            Weight initialization scale (weights drawn from N(0, scale/sqrt(nb_inputs)))
        alpha : float
            Synaptic current decay factor (0.0 to disable synaptic dynamics)
        beta : float
            Membrane potential decay factor (exp(-dt/tau_mem) for exponential decay)
        eprop : bool, optional
            Whether to use e-prop (True) or BPTT (False) (default: False)
        linear_decay : bool, optional
            Use linear decay instead of exponential (default: False)
        device : str, optional
            Device for tensor allocation (default: "cuda:0")
        dtype : torch.dtype, optional
            Data type for tensors (default: torch.float64)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
        gamma : float, optional
            Surrogate gradient scale factor for spike function (default: None, uses class default)
        """
        # Network dimensions
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.batch_size = batch_size

        # Weight initialization
        self.fwd_weight_scale = fwd_weight_scale

        # Neuron dynamics parameters
        self.alpha = alpha  # Synaptic current decay
        self.beta = beta    # Membrane potential decay
        self.use_synapse = self.alpha > 0.0

        # Learning and simulation flags
        self.eprop = eprop
        self.linear_decay = linear_decay

        # Surrogate gradient parameter
        self.gamma = gamma  # Scale factor for surrogate gradient
        self.spike_threshold = spike_threshold  # Spike threshold
        # If True: subtract threshold on spike; else hard reset to 0
        self.soft_reset = soft_reset

        self.weight_variance = self.spike_threshold*weight_variance / \
            100 if weight_variance is not None else None

        # Device and dtype
        self.device = device
        self.dtype = dtype

        # Optional features
        self.ref_per = ref_per
        if ref_per is not None and ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=self.device, dtype=torch.int)
        self.create_layer()
        # DEBUG: Check for NaN/Inf in weights
        if torch.isnan(self.ff_weights).any() or torch.isinf(self.ff_weights).any():
            logger.debug(
                f"Feedforward weights contain NaN or Inf after initialization!")
        else:
            logger.debug(
                f"Feedforward weights initialized: mean={self.ff_weights.mean().item():.6f}, std={self.ff_weights.std().item():.6f}")

    def reset_refractory_perdiod_counter(self):
        """
        Reset all refractory period counters to zero.

        Called at the start of each forward pass to clear refractory states
        from previous batches.
        """
        self.ref_per_tensor = torch.zeros_like(
            self.ref_per_tensor, dtype=torch.int)

    def update_refractory_perdiod_counter(self, spk):
        """
        Update refractory period counters based on spike activity.

        Decrements active counters by 1 and sets counters to ref_per for
        neurons that just spiked.

        Parameters
        ----------
        spk : torch.Tensor
            Binary spike tensor [batch, neurons] where 1 indicates a spike

        Notes
        -----
        Only operates on the current batch slice to handle variable batch sizes.
        """
        current_batch_size = spk.shape[0]
        current_neurons = spk.shape[1]
        if self.ref_per is None:
            raise ValueError(
                "ref_per must be set to update refractory period counter.")
        ref_per_value = int(self.ref_per)
        # Only operate on the current batch slice
        batch_slice = self.ref_per_tensor[:current_batch_size,
                                          :current_neurons]
        batch_slice[batch_slice > 0.0] -= 1
        batch_slice[spk > 0.0] = ref_per_value
        self.ref_per_tensor[:current_batch_size,
                            :current_neurons] = batch_slice

    def create_layer(self):
        """
        Initialize feedforward weight matrix.

        Creates and initializes the weight matrix with Gaussian distribution 
        following the Xavier/Glorot Initialization.

        Notes
        -----
        - Feedforward weights: [nb_neurons, nb_inputs]
        - Initialization: N(0, fwd_weight_scale/sqrt(nb_inputs))
        - Requires gradient for learning
        """
        self.ff_weights = torch.empty(
            (self.nb_neurons, self.nb_inputs),
            device=self.device,
            dtype=self.dtype,
        )
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.fwd_weight_scale / (self.nb_inputs ** 0.5))

    def compute_activity(self, input_activity, nb_steps, lower_bound=None, return_syn=False):
        """
        Compute spiking activity of feedforward layer over time.

        Simulates LIF neuron dynamics with optional synaptic filtering,
        refractory period, and membrane potential clamping. Supports both e-prop
        (hard threshold) and BPTT (surrogate gradient) spike generation.

        Parameters
        ----------
        input_activity : torch.Tensor
            Input currents with shape [batch, timesteps, nb_inputs]
        nb_steps : int
            Number of simulation timesteps
        lower_bound : float or None, optional
            Minimum membrane potential (clamping threshold, default: None)

        Returns
        -------
        tuple
            (spk_rec, mem_rec) where:
            - spk_rec : torch.Tensor
                Spike recordings [batch, timesteps, nb_neurons]
            - mem_rec : torch.Tensor
                Membrane potential recordings [batch, timesteps, nb_neurons]

        Notes
        -----
        - Spike threshold: 1.0
                - Reset mechanism:
                    - Hard reset (default): multiplicative (voltage * (1 - spike))
                    - Soft reset (optional): subtract threshold on spike (voltage - spike * threshold)
        - For e-prop: uses hard threshold (no gradient through spikes)
        - For BPTT: uses surrogate gradient function for differentiability
        - Synaptic dynamics: syn = alpha * syn + input (if alpha > 0)
        - Membrane dynamics: mem = beta * mem + syn (exponential decay)
          or mem = mem - sign(mem)*beta + syn (linear decay)
        - Refractory period blocks synaptic input when active
        """
        syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)
        new_syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                              device=self.device, dtype=self.dtype)
        mem = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)
        out = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)

        # always reset the refractory period counter at the beginning of a new forward pass
        if self.ref_per is not None and self.ref_per > 0:
            self.reset_refractory_perdiod_counter()

        mem_rec = []
        spk_rec = []
        syn_rec = []

        # Compute feedforward layer activity
        for t in range(nb_steps):
            if self.weight_variance is not None:
                noisy_spike_threshold = torch.normal(mean=torch.tensor(
                    self.spike_threshold), std=torch.tensor(self.weight_variance))
                mthr = mem - noisy_spike_threshold
            else:
                mthr = mem - self.spike_threshold
            # Use surrogate gradient for BPTT compatibility
            if self.eprop:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient with gamma parameter
                out = cast(torch.Tensor, spike_fn(
                    mthr, scale=self.gamma, threshold=0.0))
            if self.ref_per is not None and self.ref_per > 0:
                refractory_mask = self.ref_per_tensor[:out.shape[0],
                                                      :out.shape[1]] > 0
                out = out.masked_fill(refractory_mask, 0.0)

            rst = out.detach()

            # update the correct counter
            if self.ref_per is not None and self.ref_per > 0:
                self.update_refractory_perdiod_counter(rst)
                # take care of last batch
                mask = self.ref_per_tensor[:syn.shape[0], :syn.shape[1]] == 0.0
                if self.use_synapse:
                    new_syn = self.alpha * syn
                    new_syn[mask] = (self.alpha*syn[mask] +
                                     input_activity[:, t][mask])
                    syn_drive = syn
                else:
                    syn_drive = torch.zeros_like(syn)
                    syn_drive[mask] = input_activity[:, t][mask]
            else:
                if self.use_synapse:
                    new_syn = self.alpha*syn + input_activity[:, t]
                    syn_drive = syn
                else:
                    syn_drive = input_activity[:, t]

            if self.linear_decay:
                # torch.sign returns: 1 if x > 0, -1 if x < 0, and 0 if x == 0
                membrane_drive = (mem-torch.sign(mem)*self.beta) + syn_drive
            else:
                membrane_drive = self.beta*mem + syn_drive

            if self.soft_reset:
                new_mem = membrane_drive - rst * self.spike_threshold
            else:
                new_mem = membrane_drive * (1.0-rst)
            if lower_bound:
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)
            syn_rec.append(syn_drive)

            mem = new_mem
            if self.use_synapse:
                syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        syn_rec = torch.stack(syn_rec, dim=1)
        if return_syn:
            return spk_rec, mem_rec, syn_rec
        return spk_rec, mem_rec


class recurrent_layer:
    """
    Spiking recurrent layer with Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a fully-connected recurrent layer of spiking neurons with
    recurrent connections, optional synaptic dynamics, exponential or linear membrane
    potential decay, and optional refractory period. Supports both e-prop and BPTT
    learning via surrogate gradients.

    Attributes
    ----------
    nb_inputs : int
        Number of input features/channels
    nb_neurons : int
        Number of recurrent neurons in this layer
    batch_size : int
        Maximum batch size for pre-allocated tensors
    fwd_weight_scale : float
        Feedforward weight initialization scale factor
    rec_weight_scale : float
        Recurrent weight initialization scale factor
    alpha : float
        Synaptic current decay factor (0 for no synapse, or exp(-dt/tau_syn))
    beta : float
        Membrane potential decay factor (exp(-dt/tau_mem) or linear decay rate)
    eprop : bool
        Whether to use e-prop (True) or BPTT (False)
    linear_decay : bool
        Use linear decay instead of exponential decay
    device : str
        Device for tensor allocation (e.g., "cuda:0", "cpu")
    dtype : torch.dtype
        Data type for tensors
    ref_per : int or None
        Refractory period duration in timesteps (None to disable)
    ff_weights : torch.Tensor
        Feedforward weight matrix [nb_neurons, nb_inputs]
    rec_weights : torch.Tensor
        Recurrent weight matrix [nb_neurons, nb_neurons]
    ref_per_tensor : torch.Tensor
        Tracks remaining refractory period timesteps per neuron [batch_size, nb_neurons]

    Notes
    -----
    - Feedforward and recurrent weights initialized with Gaussian distribution
    - Spike generation uses hard threshold for e-prop, surrogate gradient for BPTT
    - Recurrent connections provide temporal memory and dynamics
    - Refractory period prevents neurons from spiking during cooldown
    """

    def __init__(self, nb_inputs, nb_neurons, batch_size, fwd_weight_scale, rec_weight_scale, alpha, beta, weight_variance=None, eprop=False, linear_decay=False, device=torch.device("cuda:0"), dtype=torch.float64, ref_per=None, gamma=None, spike_threshold=1.0, soft_reset=False):
        """
        Initialize recurrent spiking layer.
        # DEBUG: Log device and dtype
        logger.debug(f"Initializing recurrent_layer: device={device}, dtype={dtype}")
        # Warn for unusual parameters
        if nb_neurons <= 0:
            logger.warning(f"Recurrent layer initialized with non-positive number of neurons: {nb_neurons}")
        if nb_inputs <= 0:
            logger.warning(f"Recurrent layer initialized with non-positive number of inputs: {nb_inputs}")
        if fwd_weight_scale <= 0:
            logger.warning(f"Recurrent layer initialized with non-positive feedforward weight scale: {fwd_weight_scale}")
        if rec_weight_scale <= 0:
            logger.warning(f"Recurrent layer initialized with non-positive recurrent weight scale: {rec_weight_scale}")
        if batch_size <= 0:
            logger.warning(f"Recurrent layer initialized with non-positive batch size: {batch_size}")

        Parameters
        ----------
        nb_inputs : int
            Number of input features/channels
        nb_neurons : int
            Number of recurrent neurons in this layer
        batch_size : int
            Maximum batch size for memory pre-allocation
        fwd_weight_scale : float
            Feedforward weight initialization scale (N(0, fwd_weight_scale/sqrt(nb_inputs)))
        rec_weight_scale : float
            Recurrent weight initialization scale (N(0, rec_weight_scale/sqrt(nb_neurons)))
        alpha : float
            Synaptic current decay factor (0.0 to disable synaptic dynamics)
        beta : float
            Membrane potential decay factor (exp(-dt/tau_mem) for exponential decay)
        eprop : bool, optional
            Whether to use e-prop (True) or BPTT (False) (default: False)
        linear_decay : bool, optional
            Use linear decay instead of exponential (default: False)
        device : str, optional
            Device for tensor allocation (default: "cuda:0")
        dtype : torch.dtype, optional
            Data type for tensors (default: torch.float64)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
        gamma : float, optional
            Surrogate gradient scale factor for spike function (default: None, uses class default)
        """
        # Network dimensions
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.batch_size = batch_size

        # Weight initialization
        self.fwd_weight_scale = fwd_weight_scale
        self.rec_weight_scale = rec_weight_scale

        # Neuron dynamics parameters
        self.alpha = alpha  # Synaptic current decay
        self.beta = beta    # Membrane potential decay
        self.use_synapse = self.alpha > 0.0

        # Learning and simulation flags
        self.eprop = eprop
        self.linear_decay = linear_decay

        # Surrogate gradient parameter
        self.gamma = gamma  # Scale factor for surrogate gradient
        self.spike_threshold = spike_threshold  # Spike threshold
        # If True: subtract threshold on spike; else hard reset to 0
        self.soft_reset = soft_reset

        self.weight_variance = self.spike_threshold*weight_variance / \
            100 if weight_variance is not None else None

        # Device and dtype
        self.device = device
        self.dtype = dtype

        # Optional features
        self.ref_per = ref_per
        if ref_per is not None and ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=self.device, dtype=torch.int)
        self.create_layer()
        # DEBUG: Check for NaN/Inf in weights
        if torch.isnan(self.ff_weights).any() or torch.isinf(self.ff_weights).any():
            logger.debug(
                f"Recurrent layer feedforward weights contain NaN or Inf after initialization!")
        else:
            logger.debug(
                f"Recurrent layer feedforward weights initialized: mean={self.ff_weights.mean().item():.6f}, std={self.ff_weights.std().item():.6f}")
        if torch.isnan(self.rec_weights).any() or torch.isinf(self.rec_weights).any():
            logger.debug(
                f"Recurrent layer recurrent weights contain NaN or Inf after initialization!")
        else:
            logger.debug(
                f"Recurrent layer recurrent weights initialized: mean={self.rec_weights.mean().item():.6f}, std={self.rec_weights.std().item():.6f}")

    def reset_refractory_perdiod_counter(self):
        """
        Reset all refractory period counters to zero.

        Called at the start of each forward pass to clear refractory states
        from previous batches.
        """
        self.ref_per_tensor = torch.zeros_like(
            self.ref_per_tensor, dtype=torch.int)

    def update_refractory_perdiod_counter(self, spk):
        """
        Update refractory period counters based on spike activity.

        Decrements active counters by 1 and sets counters to ref_per for
        neurons that just spiked.

        Parameters
        ----------
        spk : torch.Tensor
            Binary spike tensor [batch, neurons] where 1 indicates a spike

        Notes
        -----
        Only operates on the current batch slice to handle variable batch sizes.
        """
        current_batch_size = spk.shape[0]
        current_neurons = spk.shape[1]
        if self.ref_per is None:
            raise ValueError(
                "ref_per must be set to update refractory period counter.")
        ref_per_value = int(self.ref_per)
        # Only operate on the current batch slice
        batch_slice = self.ref_per_tensor[:current_batch_size,
                                          :current_neurons]
        batch_slice[batch_slice > 0.0] -= 1
        batch_slice[spk > 0.0] = ref_per_value
        self.ref_per_tensor[:current_batch_size,
                            :current_neurons] = batch_slice

    def create_layer(self):
        """
        Initialize feedforward and recurrent weight matrices.

        Creates and initializes both weight matrices with Gaussian distributions 
        following the Xavier/Glorot Initialization.

        Notes
        -----
        - Feedforward weights: [nb_neurons, nb_inputs]
          Initialization: N(0, fwd_weight_scale/sqrt(nb_inputs))
        - Recurrent weights: [nb_neurons, nb_neurons]
          Initialization: N(0, rec_weight_scale/sqrt(nb_neurons))
        - Both require gradients for learning
        """
        self.ff_weights = torch.empty(
            (self.nb_neurons, self.nb_inputs),
            device=self.device,
            dtype=self.dtype,
        )
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.fwd_weight_scale / (self.nb_inputs ** 0.5))
        self.rec_weights = torch.empty(
            (self.nb_neurons, self.nb_neurons),
            device=self.device,
            dtype=self.dtype,
        )
        torch.nn.init.normal_(self.rec_weights, mean=0.0,
                              std=self.rec_weight_scale / (self.nb_neurons ** 0.5))

    def compute_activity(self, input_activity, nb_steps, lower_bound=None, rec_weights=None, return_syn=False):
        """
        Compute spiking activity of recurrent layer over time.

        Simulates recurrent LIF neuron dynamics with optional synaptic filtering,
        refractory period, and membrane potential clamping. Includes recurrent
        connections for temporal processing. Supports both e-prop (hard threshold)
        and BPTT (surrogate gradient) spike generation.

        Parameters
        ----------
        input_activity : torch.Tensor
            Input currents with shape [batch, timesteps, nb_inputs]
        nb_steps : int
            Number of simulation timesteps
        lower_bound : float or None, optional
            Minimum membrane potential (clamping threshold, default: None)
        rec_weights : torch.Tensor or None, optional
            Optional recurrent weight matrix override used only for this
            forward pass. If None, uses self.rec_weights.

        Returns
        -------
        tuple
            (spk_rec, mem_rec) where:
            - spk_rec : torch.Tensor
                Spike recordings [batch, timesteps, nb_neurons]
            - mem_rec : torch.Tensor
                Membrane potential recordings [batch, timesteps, nb_neurons]

        Notes
        -----
        - Spike threshold: 1.0
                - Reset mechanism:
                    - Hard reset (default): multiplicative (voltage * (1 - spike))
                    - Soft reset (optional): subtract threshold on spike (voltage - spike * threshold)
        - For e-prop: uses hard threshold (no gradient through spikes)
        - For BPTT: uses surrogate gradient function for differentiability
        - Total input: feedforward input + recurrent input (previous spikes)
        - Synaptic dynamics: syn = alpha * syn + total_input (if alpha > 0)
        - Membrane dynamics: mem = beta * mem + syn (exponential decay)
          or mem = mem - sign(mem)*beta + syn (linear decay)
        - Refractory period blocks synaptic input when active
        """
        syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)
        new_syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                              device=self.device, dtype=self.dtype)
        mem = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)
        out = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=self.device, dtype=self.dtype)

        # always reset the refractory period counter at the beginning of a new forward pass
        if self.ref_per is not None and self.ref_per > 0:
            self.reset_refractory_perdiod_counter()

        mem_rec = []
        spk_rec = []
        syn_rec = []

        recurrent_weights = self.rec_weights if rec_weights is None else rec_weights
        rec_mask = 1.0 - torch.eye(
            self.nb_neurons,
            device=recurrent_weights.device,
            dtype=recurrent_weights.dtype,
        )
        recurrent_weights = recurrent_weights * rec_mask

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + \
                torch.einsum("ab,bc->ac", (out, recurrent_weights.t()))
            if self.weight_variance is not None:
                noisy_spike_threshold = torch.normal(mean=torch.tensor(
                    self.spike_threshold), std=torch.tensor(self.weight_variance))
                mthr = mem - noisy_spike_threshold
            else:
                mthr = mem - self.spike_threshold
            # Use surrogate gradient for BPTT compatibility
            if self.eprop:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient with gamma parameter
                out = cast(torch.Tensor, spike_fn(
                    mthr, scale=self.gamma, threshold=0.0))
            if self.ref_per is not None and self.ref_per > 0:
                refractory_mask = self.ref_per_tensor[:out.shape[0],
                                                      :out.shape[1]] > 0
                out = out.masked_fill(refractory_mask, 0.0)

            rst = out.detach()  # We do not want to backprop through the reset

            if self.ref_per is not None and self.ref_per > 0:
                self.update_refractory_perdiod_counter(rst)
                # only update the membrane potential if not in refractory period
                # take care of last batch
                mask = self.ref_per_tensor[:syn.shape[0], :syn.shape[1]] == 0.0
                if self.use_synapse:
                    new_syn = self.alpha * syn
                    new_syn[mask] = (self.alpha*syn[mask] + h1[mask])
                    syn_drive = syn
                else:
                    syn_drive = torch.zeros_like(syn)
                    syn_drive[mask] = h1[mask]
            else:
                if self.use_synapse:
                    new_syn = self.alpha*syn + h1
                    syn_drive = syn
                else:
                    syn_drive = h1

            if self.linear_decay:
                membrane_drive = (mem-torch.sign(mem)*self.beta) + syn_drive
            else:
                membrane_drive = self.beta*mem + syn_drive

            if self.soft_reset:
                new_mem = membrane_drive - rst * self.spike_threshold
            else:
                new_mem = membrane_drive * (1.0-rst)

            if lower_bound:
                # clamp membrane potential
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)
            syn_rec.append(syn_drive)

            mem = new_mem
            if self.use_synapse:
                syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        syn_rec = torch.stack(syn_rec, dim=1)
        if return_syn:
            return spk_rec, mem_rec, syn_rec
        return spk_rec, mem_rec


class RA_I_mechanoreceptor():
    """
    Models the fast-adapting (RA-I) mechanoreceptor response in tactile sensors.

    This class monitors rapid changes in taxel sensor values and converts these changes into discrete events 
    when a specified threshold is exceeded. The step() method computes the number and timing of events based 
    on the difference between current taxel readings and the last stored values, taking into account a 
    refractory period. The remove_old_events() method clears events older than a given time, and the reset() 
    method reinitializes the internal state. Events are stored as a numpy array with each row containing the event 
    time and the corresponding channel index.
    """

    def __init__(self, taxel_values, fa_threshold, ref_period=0.003):
        """
        Initialize the RA_I_mechanoreceptor.

        Args:
            channels (list of int): List of channel indices to monitor.
            taxel_values (array-like): Initial taxel values for each channel.
            split_fa_channels (bool): If True, events are split by polarity (increase/decrease).
            fa_threshold (float): Minimum change in taxel value to trigger an event.
            ref_period (float): Refractory period for event generation.
        """

        # self.channels = channels
        self.fa_threshold = fa_threshold
        self.last_taxel_value = np.array(taxel_values, copy=True)
        self.events = np.empty((0, 2), dtype=float)  # time, channel
        self.ref_period = ref_period
        self.last_events = np.zeros_like(self.last_taxel_value, dtype=float)

    def reset(self):
        """
        Reset the last taxel values and clear all stored events.

        This is useful for reinitializing the RA-I response, for example, at the start of a new sequence or epoch.
        """

        self.last_taxel_value.fill(0.0)
        self.last_events.fill(0.0)
        self.events = np.empty((0, 2), dtype=float)

    def remove_old_events(self, t):
        """
        Remove all events that occurred before time t.

        Args:
            t (float): The cutoff time. Events with time < t are removed.
        """

        self.events = self.events[self.events[:, 0] >= t]

    def step(self, taxel_values, current_time, last_time):
        """
        Update the RA-I response based on new taxel values and generate events if the threshold is exceeded.

        For each channel, if the change in taxel value since the last update exceeds the threshold,
        one or more events are generated. If split_fa_channels is True, the polarity of the change
        (increase or decrease) is recorded.

        Args:
            taxel_values (array-like): New taxel values for each channel.
            current_time (float): Current time.
            last_time (float): Last update time.
        """

        # Strict threshold-crossing count per channel.
        nb_events_req = np.floor(
            np.abs(taxel_values - self.last_taxel_value) / self.fa_threshold).astype(int)

        active_channels = np.where(nb_events_req > 0)[0]
        if active_channels.size == 0:
            return np.empty((0, 2), dtype=float)

        eps = 1e-12
        new_events = np.empty((0, 2), dtype=float)
        for channel in active_channels:
            requested = int(nb_events_req[channel])

            # Keep one-sided window policy: exclude last_time, allow current_time.
            first_allowed = max(
                last_time + eps, self.last_events[channel] + self.ref_period)
            if first_allowed > current_time:
                self.last_taxel_value[channel] = taxel_values[channel]
                continue

            max_events = int(
                np.floor((current_time - first_allowed) / self.ref_period)) + 1
            n_events = min(requested, max_events)
            if n_events <= 0:
                self.last_taxel_value[channel] = taxel_values[channel]
                continue

            if n_events == 1:
                times = np.array([current_time], dtype=float)
            else:
                dt = (current_time - first_allowed) / (n_events - 1)
                dt = max(dt, self.ref_period)
                times = current_time - \
                    np.arange(n_events - 1, -1, -1, dtype=float) * dt

            y_arr = np.full(times.shape, channel)
            new_events = np.vstack(
                (new_events, np.column_stack((times, y_arr))))

            self.last_events[channel] = times[-1]
            self.last_taxel_value[channel] = taxel_values[channel]

        if new_events.size > 0:
            self.events = np.append(self.events, new_events, axis=0)
            return new_events[np.argsort(new_events[:, 0])]
        return np.empty((0, 2), dtype=float)


class SA_II_mechanoreceptor():
    """
    Models the event-based response of Slowly-Adapting type II (SA-II) mechanoreceptors.

    This class generates events for each channel based on the absolute value of the taxel (tactile sensor) values.
    The event rate is determined by the distance from the maximum taxel value (255), scaled by a slope parameter.
    Events are generated at regular intervals according to this rate and stored as a numpy array with columns for event time and channel index.

    Attributes:
        channels (list of int): List of channel indices to monitor.
        slope (float): Scaling factor for converting taxel value to event rate.
        events (np.ndarray): Array of shape (N, 2) where each row is [time, channel].
    """

    def __init__(self, channels, max_frequ, ref_period=0.003):
        """
        Initialize the SA_II_mechanoreceptor object.

        Args:
            channels (list of int): List of channel indices to monitor.
            slope (float): Scaling factor for converting taxel value to event rate.
        """
        slope = max_frequ / 255.0  # Hz per taxel value
        self.slope = slope
        self.events = np.empty((0, 2), dtype=float)  # time, channel
        self.last_events = np.zeros(channels, dtype=float)
        self.ref_period = ref_period

    def reset(self):
        """
        Reset the event storage for all channels.

        This method clears all stored events, reinitializing the SA-II response.
        """

        self.events = np.empty((0, 2), dtype=float)
        self.last_events.fill(0.0)

    def remove_old_events(self, t):
        """
        Remove events that occurred before time t for all channels.

        Args:
            t (float): The cutoff time. Events with time < t are removed.
        """

        self.events = self.events[self.events[:, 0] >= t]

    def step(self, taxel_values, current_time, last_time):
        """
        Update the SA-II response based on new taxel values and generate events at the appropriate rate.

        For each channel, computes the event rate as a function of the taxel value and slope, and generates events at regular intervals between the last event time and the current time. Adds new events to the event storage for each channel.

        Args:
            taxel_values (array-like): New taxel values for each channel.
            current_time (float): Current time.
            last_time (float): Last update time.
        """

        # Only channels with value > 0 can generate events
        active_mask = taxel_values > 0
        active_channels = np.where(active_mask)[0]
        if active_channels.size == 0:
            # logger.debug("No active channels for SA-II mechanoreceptor.")
            return np.empty((0, 2), dtype=float)

        local_frequ = taxel_values[active_channels] * self.slope  # Hz
        dt_sa_events = np.maximum(1 / local_frequ, self.ref_period)  # sec

        # For each active channel, generate event times
        new_events = np.empty((0, 2), dtype=float)
        for idx, channel in enumerate(active_channels):
            dt = dt_sa_events[idx]
            # Find last event time for this channel
            last_event_time = self.last_events[channel]
            # Generate event times within the window current_time and last_time taking into last_event_time

            eps = 1e-12
            first_allowed = max(last_time + eps, last_event_time + dt)
            if first_allowed > current_time:
                continue

            n_events = int(np.floor((current_time - first_allowed) / dt)) + 1
            event_times = current_time - \
                np.arange(n_events - 1, -1, -1, dtype=float) * dt

            if event_times.size > 0:
                # Channel indices start from 0
                y_arr = np.full(event_times.shape, channel)
                # Stack as (N, 2) array
                new_events = np.vstack(
                    (new_events, np.column_stack((event_times, y_arr))))

        if new_events.size > 0:
            # Append all new events at once
            self.events = np.append(self.events, new_events, axis=0)

            # Update per-channel last event time with the last event emitted in this step.
            for channel in active_channels:
                channel_events = new_events[new_events[:, 1] == channel]
                if channel_events.size > 0:
                    self.last_events[channel] = channel_events[-1, 0]

            return new_events[np.argsort(new_events[:, 0])]
        return np.empty((0, 2), dtype=float)


class SA_II_alt_mechanoreceptor():
    """
    Models the event-based response of Slowly-Adapting type II (SA-II) mechanoreceptors with alternative encoding.

    This class generates events for each channel based on the taxel (tactile sensor) values with an encoding
    opposite to the standard SA-II. The event rate is proportional to the taxel value itself, meaning:
    - High taxel values (near 255) generate high frequency events
    - Low taxel values (near 0) generate low frequency events

    This represents the alternative SA-II type found in research where frequency increases with pressure.

    Attributes:
        channels (list of int): List of channel indices to monitor.
        slope (float): Scaling factor for converting taxel value to event rate.
        events (np.ndarray): Array of shape (N, 2) where each row is [time, channel].
    """

    def __init__(self, channels, slope):
        """
        Initialize the SA_II_mechanoreceptor object.

        Args:
            channels (list of int): List of channel indices to monitor.
            slope (float): Scaling factor for converting taxel value to event rate.
        """

        self.slope = slope
        self.events = np.empty((0, 2), dtype=float)  # time, channel
        self.last_events = np.zeros(channels, dtype=float)

    def reset(self):
        """
        Reset the event storage for all channels.

        This method clears all stored events, reinitializing the SA-II response.
        """

        self.events = np.empty((0, 2), dtype=float)

    def remove_old_events(self, t):
        """
        Remove events that occurred before time t for all channels.

        Args:
            t (float): The cutoff time. Events with time < t are removed.
        """

        self.events = self.events[self.events[:, 0] >= t]

    def step(self, taxel_values, current_time, last_time):
        """
        Update the SA-II alt response based on new taxel values and generate events at the appropriate rate.

        For each channel, computes the event rate as a function of the inverse taxel value and slope, and generates events at regular intervals between the last event time and the current time. Adds new events to the event storage for each channel.

        Args:
            taxel_values (array-like): New taxel values for each channel.
            current_time (float): Current time.
            last_time (float): Last update time.
        """

        # Only channels with value > 0 can generate events
        active_mask = taxel_values < 255
        active_channels = np.where(active_mask)[0]
        if active_channels.size == 0:
            # logger.debug("No active channels for SA-II mechanoreceptor.")
            return np.empty((0, 2), dtype=float)

        local_frequ = (255 - taxel_values[active_channels]) * self.slope  # Hz
        dt_sa_events = 1 / local_frequ  # sec

        # For each active channel, generate event times
        new_events = np.empty((0, 2), dtype=float)
        for idx, channel in enumerate(active_channels):
            dt = dt_sa_events[idx]
            # Find last event time for this channel
            last_event_time = self.last_events[channel]
            # Generate event times within the window current_time and last_time taking into last_event_time

            if last_event_time + dt > current_time:
                # No new events if first event is after current time (no projection into the future)
                continue
            elif last_event_time + dt < last_time:
                # set first event to last_time if last event + dt is before last_time (no projection into the past)
                event_times = np.arange(
                    last_time, current_time, dt)
            else:
                # set first event to last_event_time + dt (falls within the window and respects the correct frequency)
                event_times = np.arange(
                    last_event_time + dt, current_time, dt)

            if event_times.size > 0:
                # Channel indices start from 1
                y_arr = np.full(event_times.shape, channel + 1)
                # Stack as (N, 2) array
                new_events = np.vstack(
                    (new_events, np.column_stack((event_times, y_arr))))

        if new_events.size > 0:
            # Append all new events at once
            self.events = np.append(self.events, new_events, axis=0)

            # Get the index of the last occurrence for each unique channel id
            last_indices = np.r_[np.nonzero(np.diff(new_events[:, 1]))[
                0], len(new_events[:, 1])-1]
            unique_channels = new_events[last_indices, 1].astype(int) - 1
            unique_times = new_events[last_indices, 0]

            # Update last_events for each channel found in new_events
            self.last_events[unique_channels] = unique_times

            return new_events[np.argsort(new_events[:, 0])]
        return np.empty((0, 2), dtype=float)


class AdExLIF_neuron(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'W', 'spk'])

    def __init__(
            self,
            nb_inputs,
            dt=1.,
            Vr=0.0,
            Vth=1.0,
            Vrh=0.0,
            Vreset=0.0,
            a=0.0,
            b=0.0,
            R=1.0,
            taum=1.0,
            tauw=1.0,
            device='cpu'
    ):
        super(AdExLIF_neuron, self).__init__()

        self.linear = torch.ones(1, nb_inputs).to(device)
        self.dt = dt
        self.N = nb_inputs

        self.Vr = Vr
        self.Vth = Vth
        self.Vrh = Vrh
        self.Vreset = Vreset
        self.a = a
        self.b = b
        self.R = R
        self.taum = taum
        self.tauw = tauw

        self.device = device

        self.state = self.NeuronState(
            V=torch.zeros(1, self.N, device=self.device) + self.Vr,
            W=torch.zeros(1, self.N, device=self.device),
            spk=torch.zeros(1, self.N, device=self.device)
        )

    def forward(self, input):
        # print(1)
        V = self.state.V
        W = self.state.W
        I = (self.linear * input)
        dV = (-(V - self.Vr) + self.dt * torch.exp((V - self.Vrh) /
              self.dt) + self.R * (I - W)) / (self.taum)
        dW = (self.a * (V - self.Vr) - W) / self.tauw

        V = V + self.dt * dV
        W = W + self.dt * dW

        spk = spike_fn(V - self.Vth)

        W = (1 - spk) * W + (spk) * (W + self.b)
        V = (1 - spk) * V + (spk) * self.Vreset

        self.state = self.NeuronState(V=V, W=W, spk=spk)
        return spk

    def reset(self):
        self.state = self.NeuronState(
            V=torch.zeros(1, self.N, device=self.device) + self.Vr,
            W=torch.zeros(1, self.N, device=self.device),
            spk=torch.zeros(1, self.N, device=self.device)
        )


class CuBaLIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "syn", "spk"])

    def __init__(
            self,
            nb_inputs,
            dt=1/1000,
            alpha=1.0,
            beta=1.0,
            thr=1.0,
            R=1.0,
            device='cpu'
    ):
        super(CuBaLIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.alpha = alpha
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        self.device = device

        self.state = self.NeuronState(
            V=torch.zeros(1, self.nb_inputs,
                          device=self.device),
            syn=torch.zeros(1, self.nb_inputs,
                            device=self.device),
            spk=torch.zeros(1, self.nb_inputs, device=self.device),
        )

    def forward(self, input):
        V = self.state.V
        spk = self.state.spk
        syn = self.state.syn

        # syn = self.alpha*syn + spk
        # V = (self.beta * V + (1.0-self.beta) * input *
        #      self.R + (1.0-self.beta)*syn) * (1.0 - spk)
        syn = self.alpha*syn + input*self.R
        V = (self.beta * V + syn) * (1.0 - spk)  # reset mechanism: zero
        spk = spike_fn(V-self.threshold)

        self.state = self.NeuronState(V=V, syn=syn, spk=spk)

        return spk

    def reset(self):
        self.state = self.NeuronState(
            V=torch.zeros(1, self.nb_inputs, device=self.device),
            syn=torch.zeros(1, self.nb_inputs, device=self.device),
            spk=torch.zeros(1, self.nb_inputs, device=self.device),
        )


class IZ_neuron(nn.Module):
    # u = membrane recovery variable
    NeuronState = namedtuple('NeuronState', ['V', 'u', 'spk'])

    def __init__(
        self,
        nb_inputs,
        dt=1/1000,
        a=0.02,
        b=0.2,
        c=-65,
        d=8,
        device='cpu'
    ):
        super(IZ_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs))
        self.N = nb_inputs
        # define some constants
        self.spike_value = 35  # spike threshold

        # define parameters
        self.a = a
        self.b = b
        self.c = c  # reset potential
        self.d = d
        self.dt = dt*1E3  # convert from sec to ms

        self.device = device

        self.state = self.NeuronState(
            V=torch.ones(1, self.N, device=self.device) * self.c,
            u=torch.zeros(
                1, self.N, device=self.device) * self.b*self.c,
            spk=torch.zeros(1, self.N, device=self.device)
        )

    def forward(self, input):
        V = self.state.V
        u = self.state.u

        numerical_res = round(self.dt)
        if self.dt > 1:
            output_spike = torch.zeros_like(self.state.spk)
            for i in range(numerical_res):
                V = V + (((0.04 * V + 5) * V) + 140 - u + input)
                u = u + self.a * (self.b * V - u)

                # create spike when threshold reached
                spk = spike_fn(V - self.spike_value)
                output_spike = output_spike + spk

                # (reset membrane voltage) or (only update)
                V = (spk * self.c) + ((1 - spk) * V)
                # (reset recovery) or (update currents)
                u = (spk * (u + self.d)) + ((1 - spk) * u)
        else:
            V = V + self.dt*(((0.04 * V + 5) * V) + 140 - u + input)
            u = u + self.dt*self.a * (self.b * V - u)

            # create spike when threshold reached
            spk = spike_fn(V - self.spike_value)
            output_spike = spk

            # (reset membrane voltage) or (only update)
            V = (spk * self.c) + ((1 - spk) * V)
            # (reset recovery) or (update currents)
            u = (spk * (u + self.d)) + ((1 - spk) * u)

        self.state = self.NeuronState(V=V, u=u, spk=spk)

        return spk

    def reset(self):
        self.state = self.NeuronState(
            V=torch.ones(1, self.N, device=self.device) * self.c,
            u=torch.zeros(
                1, self.N, device=self.device) * self.b*self.c,
            spk=torch.zeros(1, self.N, device=self.device)
        )


class LIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "spk"])

    def __init__(
            self,
            nb_inputs,
            dt=1/1000,
            beta=1.0,
            thr=1.0,
            R=1.0,
            device='cpu'
    ):
        super(LIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        self.device = device

        self.state = self.NeuronState(
            V=torch.zeros(1, self.nb_inputs, device=self.device),
            spk=torch.zeros(1, self.nb_inputs, device=self.device),
        )

    def forward(self, input):

        V = self.state.V
        spk = self.state.spk

        # V = (self.beta * V + (1.0-self.beta) * input * self.R) * (1.0 - spk)
        V = (self.beta * V + input * self.R) * \
            (1.0 - spk)  # reset mechanism: zero
        spk = spike_fn(V-self.threshold)

        self.state = self.NeuronState(V=V, spk=spk)

        return spk

    def reset(self):
        self.state = self.NeuronState(
            V=torch.zeros(1, self.nb_inputs, device=self.device),
            spk=torch.zeros(1, self.nb_inputs, device=self.device),
        )


class MN_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "i1", "i2", "Thr", "spk"])

    def __init__(
        self,
        nb_inputs,
        dt=1 / 1000,
        a=5,
        A1=10,
        A2=-0.6,
        b=10,
        G=50,
        k1=200,
        k2=20,
        R1=0,
        R2=1,
        device='cpu'
    ):  # default combination: M2O of the original paper
        super(MN_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs))
        self.N = nb_inputs
        one2N_matrix = torch.ones(1, nb_inputs)
        # define some constants
        self.C = 1
        self.EL = -0.07  # V
        self.Vr = -0.07  # V
        self.Tr = -0.06  # V
        self.Tinf = -0.05  # V

        # define parameters
        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.b = b  # 1/s
        self.G = G * self.C  # 1/s
        self.k1 = k1  # 1/s
        self.k2 = k2  # 1/s
        self.R1 = R1  # not Ohm?
        self.R2 = R2  # not Ohm?
        self.dt = dt  # get dt from sample rate!

        # set up missing parameters
        self.a = nn.Parameter(one2N_matrix * self.a)
        self.A1 = nn.Parameter(one2N_matrix * self.A1 *
                               self.C)
        self.A2 = nn.Parameter(one2N_matrix * self.A2 *
                               self.C)

        self.device = device
        self.state = self.NeuronState(
            V=torch.ones(1, self.N, device=self.device) * self.EL,
            i1=torch.zeros(1, self.N, device=self.device),
            i2=torch.zeros(1, self.N, device=self.device),
            Thr=torch.ones(1, self.N, device=self.device) * self.Tinf,
            spk=torch.zeros(1, self.N, device=self.device),
        )

    def forward(self, input):
        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt
        V += self.dt * (self.linear * input + i1 + i2 -
                        self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = spike_fn(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
        Thr = ((1 - spk) * Thr) + \
            ((spk) * torch.max(Thr, torch.tensor(self.Tr, device=self.device)))
        V = ((1 - spk) * V) + ((spk) * self.Vr)

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = self.NeuronState(
            V=torch.ones(1, self.N, device=self.device) * self.EL,
            i1=torch.zeros(1, self.N, device=self.device),
            i2=torch.zeros(1, self.N, device=self.device),
            Thr=torch.ones(1, self.N, device=self.device) * self.Tinf,
            spk=torch.zeros(1, self.N, device=self.device),
        )
