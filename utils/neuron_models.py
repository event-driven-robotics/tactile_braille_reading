"""neuron_models.py

Spiking neuron models and mechanoreceptor implementations for event-based processing.

This module provides classes and functions for simulating various neuron and mechanoreceptor models,
including Leaky Integrator (LI), Current-Based Leaky Integrator (CuBaLI), Leaky Integrate-and-Fire (LIF),
Current-Based Leaky Integrate-and-Fire (CuBaLIF), and their recurrent variants. It also includes event-based
mechanoreceptor models (FA-I and SA-II) and a fascicle response model for aggregating and processing
spike events through configurable neuron populations.

The models are implemented using PyTorch for efficient computation and support both feedforward and recurrent
architectures, surrogate gradient learning, and event-based input processing. Utility functions for visualizing
network connectivity are also provided.

Classes:
    - SurrGradSpike: Surrogate gradient spiking nonlinearity for PyTorch autograd.
    - LI, CuBaLI, LIF, CuBaLIF, RLIF, CuBaRLIF: Neuron layer models.
    - FA_I_mechanoreceptor, SA_II_mechanoreceptor: Event-based mechanoreceptor models.
    - fascicle_response: Aggregates and processes events through neuron populations.
    - Utility functions for network visualization.

Author: Simon F. Muller-Cleve
Date: January 12, 2026
"""

import logging
from typing import cast

import numpy as np
import torch

logger = logging.getLogger(__name__)

# NOTE: one can use device=input.device, dtype=input.dtype to infer the device and dtype from the input tensor


class STEFunction(torch.autograd.Function):
    """
    Here we define the Straight-Through Estimator (STE) function.
    This function allows us to ignore the non-differentiable part
    in our network, i.e. the discretization of the weights.
    The function applys the discretization and the clamping.
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

    scale = 15.0
    threshold = 1.0

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
        if scale is not None:
            ctx.scale = scale
        else:
            ctx.scale = SurrGradSpike.scale

        # Use provided threshold or class default
        thr = threshold if threshold is not None else SurrGradSpike.threshold

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


def spike_fn(input, scale=None, threshold=None):
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
        self.soft_reset = soft_reset  # If True: subtract threshold on spike; else hard reset to 0

        self.weight_variance = self.spike_threshold*weight_variance/100 if weight_variance is not None else None

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
            logger.debug(f"Feedforward weights contain NaN or Inf after initialization!")
        else:
            logger.debug(f"Feedforward weights initialized: mean={self.ff_weights.mean().item():.6f}, std={self.ff_weights.std().item():.6f}")

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
            raise ValueError("ref_per must be set to update refractory period counter.")
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
                noisy_spike_threshold = torch.normal(mean=torch.tensor(self.spike_threshold), std=torch.tensor(self.weight_variance))
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
                refractory_mask = self.ref_per_tensor[:out.shape[0], :out.shape[1]] > 0
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
        self.soft_reset = soft_reset  # If True: subtract threshold on spike; else hard reset to 0

        self.weight_variance = self.spike_threshold*weight_variance/100 if weight_variance is not None else None

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
            logger.debug(f"Recurrent layer feedforward weights contain NaN or Inf after initialization!")
        else:
            logger.debug(f"Recurrent layer feedforward weights initialized: mean={self.ff_weights.mean().item():.6f}, std={self.ff_weights.std().item():.6f}")
        if torch.isnan(self.rec_weights).any() or torch.isinf(self.rec_weights).any():
            logger.debug(f"Recurrent layer recurrent weights contain NaN or Inf after initialization!")
        else:
            logger.debug(f"Recurrent layer recurrent weights initialized: mean={self.rec_weights.mean().item():.6f}, std={self.rec_weights.std().item():.6f}")

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
            raise ValueError("ref_per must be set to update refractory period counter.")
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
                noisy_spike_threshold = torch.normal(mean=torch.tensor(self.spike_threshold), std=torch.tensor(self.weight_variance))
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
                refractory_mask = self.ref_per_tensor[:out.shape[0], :out.shape[1]] > 0
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


class FA_I_mechanoreceptor():
    """
    Models the fast-adapting (FA-I) mechanoreceptor response in tactile sensors.

    This class monitors rapid changes in taxel sensor values and converts these changes into discrete events 
    when a specified threshold is exceeded. The step() method computes the number and timing of events based 
    on the difference between current taxel readings and the last stored values, taking into account a 
    refractory period. The remove_old_events() method clears events older than a given time, and the reset() 
    method reinitializes the internal state. Events are stored as a numpy array with each row containing the event 
    time and the corresponding channel index.
    """

    def __init__(self, taxel_values, fa_threshold, ref_period=0.003):
        """
        Initialize the FA_I_mechanoreceptor.

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

    def reset(self):
        """
        Reset the last taxel values and clear all stored events.

        This is useful for reinitializing the FA-I response, for example, at the start of a new sequence or epoch.
        """

        self.last_taxel_value.fill(0.0)
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
        Update the FA-I response based on new taxel values and generate events if the threshold is exceeded.

        For each channel, if the change in taxel value since the last update exceeds the threshold,
        one or more events are generated. If split_fa_channels is True, the polarity of the change
        (increase or decrease) is recorded.

        Args:
            taxel_values (array-like): New taxel values for each channel.
            current_time (float): Current time.
            last_time (float): Last update time.
        """

        nb_events = np.floor(
            np.abs(taxel_values - self.last_taxel_value) / self.fa_threshold).astype(int)
        # logger.debug(f"FA-I events detected: {nb_events}")
        dt_fa_events = (current_time - last_time) / (nb_events + 1)
        # TODO check if dt_fa_events at any point smaller ref_per and if so update that to ref_per and recalculate number of events
        if np.min(dt_fa_events) < self.ref_period:
            dt_smaller_ref_mask = dt_fa_events < self.ref_period
            dt_fa_events[dt_smaller_ref_mask] = self.ref_period
            nb_events[dt_smaller_ref_mask] = (
                (current_time - last_time)/self.ref_period).astype(int)
        # direction = np.sign(taxel_values - self.last_taxel_value)

        # Find channels with events
        active_channels = np.where(nb_events > 0)[0]
        total_new_events = np.sum(nb_events)

        if total_new_events > 0:
            # Preallocate arrays for all new events (maximum possible)
            event_times = np.empty(total_new_events)
            y_arr = np.empty(total_new_events)

            idx = 0
            for channel in active_channels:
                dt = dt_fa_events[channel]
                # ensure we do not create more spikes then possible with refractory period
                n_events = nb_events[channel]
                # Calculate times
                times = last_time + np.arange(1, n_events + 1) * dt
                event_times[idx:idx+len(times)] = times
                # Channel indices start from 0
                y_arr[idx:idx+len(times)] = channel
                # Update last_taxel_value for this channel
                # NOTE: now we have 'trailing' of event because value at last time is saved and not the possibly lower sample value
                # self.last_taxel_value[channel] += direction[channel] * \
                #     len(times) * self.fa_threshold
                # hard reset, no trailing
                # introduces a slight inprecision, because we should actually compare changes to the value present after the ref period
                self.last_taxel_value[channel] = taxel_values[channel]
                idx += len(times)

            # Stack and append all new events at once
            new_events = np.column_stack((event_times[:idx], y_arr[:idx]))
            self.events = np.append(self.events, new_events, axis=0)
            return new_events[np.argsort(new_events[:, 0])]
        else:
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

    def __init__(self, channels, max_frequ):
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
                # Channel indices start from 0
                y_arr = np.full(event_times.shape, channel)
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
