"""
neuron_models.py

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
License: [License to be filled in, e.g., MIT, GPL, etc.]
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# NOTE: one can use device=input.device, dtype=input.dtype to infer the device and dtype from the input tensor


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 10
    threshold = 0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > SurrGradSpike.threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad


spike_fn = SurrGradSpike.apply


class LI:
    """
    Class to initialize and compute a feedforward layer of Leaky Integrator (LI) neurons.

    This class implements a feedforward layer of Leaky Integrator (LI) neurons, which accumulate
    input over time with a leaky (decaying) membrane potential. The layer supports computation
    of membrane potentials for each neuron at each time step, using a simple leaky integration
    model without spiking or synaptic currents.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LI neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the LI neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LI neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the membrane potential tensor to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.mem.zero_()

    def step(self, input_activity_t):
        """
        Compute the membrane potential of the LI neuron layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs)
                                             for a single time step.

        Returns:
            torch.Tensor: Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.mem = (self.beta * self.mem +
                    torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights))

        return self.mem


class CuBaLI:
    """
    Class to initialize and compute a feedforward layer of Current-Based Leaky Integrator (CuBaLI) neurons.

    This class implements a feedforward layer of Current-Based Leaky Integrator (CuBaLI) neurons, where each neuron's membrane potential is influenced by a leaky integration of synaptic currents, and the synaptic currents themselves are leaky integrators of the weighted input. The model supports efficient computation of synaptic currents and membrane potentials for each neuron at each time step, and is suitable for use in deep learning frameworks with GPU acceleration and automatic differentiation.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of neurons in the layer.
        alpha (float): Synaptic current decay constant (leak rate for synaptic current).
        beta (float): Membrane potential decay constant (leak rate for membrane potential).
        device (str or torch.device): Device for tensor storage and computation (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, alpha, beta, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the CuBaLI neuron layer with the specified parameters and optionally provided weights.

        Sets up the feedforward weight matrix, synaptic current, and membrane potential tensors. The weights are initialized with a normal distribution if not provided. All tensors are allocated on the specified device and with the specified data type.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            alpha (float): Synaptic current decay constant.
            beta (float): Membrane potential decay constant.
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device for tensor storage (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the synaptic current and membrane potential tensors to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.syn.zero_()
        self.mem.zero_()

    def step(self, input_activity_t):
        """
        Compute the synaptic current and membrane potential for the CuBaLI neuron layer for a single time step.

        The synaptic current is updated using a leaky integration of the weighted input, and the membrane potential is updated using a leaky integration of the synaptic current. Returns the updated synaptic current and membrane potential tensors.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.syn = self.alpha * self.syn + \
            torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights)
        self.mem = (self.beta * self.mem + self.syn)

        return self.syn, self.mem


class LIF:
    """
    Class to initialize and compute a feedforward layer of Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a feedforward layer of LIF neurons, which accumulate input over time with a leaky (decaying) membrane potential and emit spikes when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LIF neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the LIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LIF neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the membrane potential and reset state tensors to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.mem.zero_()
        self.rst.zero_()

    def step(self, input_activity_t):
        """
        Compute the membrane potential and spike output of the LIF neuron layer for a single time step.

        The membrane potential is updated using a leaky integration of the input, and a spike is emitted if the membrane potential crosses the threshold. The reset state is updated to reflect which neurons have spiked.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.mem = (self.beta * self.mem + torch.einsum("ab,bc->ac",
                    input_activity_t, self.ff_weights)) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        return self.rst, self.mem


class CuBaLIF:
    """
    Class to initialize and compute a feedforward layer of Current-Based Leaky Integrate-and-Fire (CuBaLIF) neurons.

    This class implements a feedforward layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons. Each neuron's membrane potential is influenced by a leaky integration of synaptic currents, and the synaptic currents themselves are leaky integrators of the weighted input. The model supports computation of synaptic currents, membrane potentials, and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of feedforward neurons.
        alpha (float): Synaptic decay constant.
        beta (float): Membrane decay constant.
        device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, alpha, beta, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the CuBaLIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of feedforward neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            alpha (float): Synaptic decay constant.
            beta (float): Membrane decay constant.
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights
            logger.info("Using provided weights for CuBaLIF layer.")
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))
            logger.info("Initialized CuBaLIF layer with random weights.")

        # Initialize the synaptic current and membrane potential
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the synaptic current, membrane potential, and reset state tensors to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.syn.zero_()
        self.mem.zero_()
        self.rst.zero_()

    def step(self, input_activity_t):
        """
        Compute the activity of the feedforward CuBaLIF layer for a single time step.

        The synaptic current is updated using a leaky integration of the weighted input, and the membrane potential is updated using a leaky integration of the synaptic current. A spike is emitted if the membrane potential crosses a threshold, and the reset state is updated accordingly.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.syn = self.alpha * self.syn + \
            torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights)
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        return self.rst, self.syn, self.mem


class RLIF:
    """
    Class to initialize and compute a recurrent layer of Leaky Integrate-and-Fire (RLIF) neurons.

    This class implements a recurrent layer of LIF neurons with both feedforward and recurrent connections. Each neuron's membrane potential is updated by a leaky integration of the weighted input and recurrent activity, and a spike is emitted when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of recurrent neurons.
        beta (float): Membrane decay constant.
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        rec_weights (torch.Tensor): Recurrent weight matrix of shape (nb_neurons, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, fwd_scale=0.1, rec_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the RLIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of recurrent neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            beta (float): Membrane decay constant.
            weights (tuple of torch.Tensor, optional): Tuple containing predefined feedforward and recurrent weight matrices.
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights[0]
            self.rec_weights = weights[1]
        else:
            # Initialize feedforward and recurrent weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

            self.rec_weights = torch.empty(
                (nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.rec_weights, mean=0.0,
                                  std=rec_scale / np.sqrt(nb_neurons))

        # # ensure, that recurrent connections to a neuron itself are zero (no self connections)
        # self.rec_weights[torch.arange(nb_neurons),
        #                torch.arange(nb_neurons)] = 0.0

        # Initialize synaptic current, membrane potential, and spike output
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the membrane potential and reset state tensors to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.mem.zero_()
        self.rst.zero_()

    def step(self, input_activity_t):
        """
        Compute the activity of the recurrent LIF layer for a single time step.

        The membrane potential is updated using a leaky integration of the feedforward input and recurrent activity. A spike is emitted if the membrane potential crosses the threshold, and the reset state is updated accordingly.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        # Compute input and recurrent contributions
        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights) + \
            torch.einsum("ab,bc->ac", self.rst, self.rec_weights)

        # Update synaptic current and membrane potential
        self.mem = (self.beta * self.mem + h1) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()  # Reset spikes

        return self.rst, self.mem


class CuBaRLIF:
    """
    Class to initialize and compute a recurrent layer of Current-Based Leaky Integrate-and-Fire (CuBaRLIF) neurons.

    This class implements a recurrent layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons, where each neuron's membrane potential is influenced by a leaky integration of synaptic currents, and the synaptic currents themselves are leaky integrators of both the weighted input and recurrent activity. The model supports computation of synaptic currents, membrane potentials, and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of recurrent neurons.
        alpha (float): Synaptic current decay constant.
        beta (float): Membrane potential decay constant.
        device (str or torch.device): Device for tensor storage and computation (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        rec_weights (torch.Tensor): Recurrent weight matrix of shape (nb_neurons, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, alpha, beta, fwd_scale=0.1, rec_scale=0.1, weights=None, device="cuda", dtype=torch.float, requires_grad=True):
        """
        Initialize the CuBaRLIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of recurrent neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            alpha (float): Synaptic current decay constant.
            beta (float): Membrane potential decay constant.
            weights (tuple of torch.Tensor, optional): Tuple containing predefined feedforward and recurrent weight matrices.
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            self.ff_weights = weights[0]
            self.rec_weights = weights[1]
        else:
            # Initialize feedforward and recurrent weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

            self.rec_weights = torch.empty(
                (nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.rec_weights, mean=0.0,
                                  std=rec_scale / np.sqrt(nb_neurons))

        # # ensure, that recurrent connections to a neuron itself are zero (no self connections)
        # self.rec_weights[torch.arange(nb_neurons),
        #                torch.arange(nb_neurons)] = 0.0

        # Initialize synaptic current, membrane potential, and spike output
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def reset(self):
        """
        Reset the synaptic current, membrane potential, and reset state tensors to zero.

        This method is useful for reinitializing the layer state, for example, at the start of a new sequence or epoch.
        """

        self.syn.zero_()
        self.mem.zero_()
        self.rst.zero_()

    def step(self, input_activity_t):
        """
        Compute the activity of the recurrent CuBaLIF layer for a single time step.

        The synaptic current is updated using a leaky integration of the weighted input, and the membrane potential is updated using a leaky integration of the synaptic current. A spike is emitted if the membrane potential crosses a threshold, and the reset state is updated accordingly.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.syn = self.alpha * self.syn + \
            torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights)
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        return self.rst, self.syn, self.mem


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
                y_arr[idx:idx+len(times)] = channel  # Channel indices start from 0
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
                continue  # No new events if first event is after current time (no projection into the future)
            elif last_event_time + dt < last_time:
                # set first event to last_time if last event + dt is before last_time (no projection into the past)
                event_times = np.arange(
                    last_time, current_time, dt)
            else:
                # set first event to last_event_time + dt (falls within the window and respects the correct frequency)
                event_times = np.arange(
                    last_event_time + dt, current_time, dt)

            if event_times.size > 0:
                y_arr = np.full(event_times.shape, channel)  # Channel indices start from 0
                # Stack as (N, 2) array
                new_events = np.vstack((new_events, np.column_stack((event_times, y_arr))))

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
                continue  # No new events if first event is after current time (no projection into the future)
            elif last_event_time + dt < last_time:
                # set first event to last_time if last event + dt is before last_time (no projection into the past)
                event_times = np.arange(
                    last_time, current_time, dt)
            else:
                # set first event to last_event_time + dt (falls within the window and respects the correct frequency)
                event_times = np.arange(
                    last_event_time + dt, current_time, dt)

            if event_times.size > 0:
                y_arr = np.full(event_times.shape, channel + 1)  # Channel indices start from 1
                # Stack as (N, 2) array
                new_events = np.vstack((new_events, np.column_stack((event_times, y_arr))))

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


class fascicle_response():
    """
    Models the response of neural fascicles using a population of LIF (Leaky Integrate-and-Fire) neurons.

    This class aggregates sparse event data from multiple input channels, converts them into a dense time-binned representation,
    processes them through a configurable neuron model (default: CuBaLIF), and accumulates the resulting spike events for each fascicle.
    The class supports both single-shot and sequential (binned) processing, allowing for continuous accumulation of spikes over time.

    Attributes:
        channels (list of int): List of input channel indices to monitor.
        nb_neurons_fascilce (int): Number of fascicle (output) neurons.
        batch_size (int): Batch size for processing (default: 1).
        time_bins (int): Number of time bins for event aggregation.
        dtype (torch.dtype): Data type for tensors.
        device (str or torch.device): Device for tensor storage and computation.
        spk_tensor (torch.Tensor): Tensor storing spike outputs for each fascicle and time bin.
        dense_events (torch.Tensor): Dense representation of input events (batch, channels, time_bins).
        events (np.ndarray): Array of shape (N, 2) where each row is [time, fascicle_index].
        neuron (object): Instantiated neuron model (e.g., CuBaLIF, LIF) used for fascicle computation.
        neuron_model (str): Name of the neuron model used.
    """

    def __init__(self, channels, nb_neurons_fascilce, batch_size=1, time_bins=100, dtype=torch.float32, device="cuda", neuron_model_config=None):
        """
        Initialize the fascicle_response object.

        Args:
            channels (list of int): List of input channel indices to monitor.
            nb_neurons_fascilce (int): Number of fascicle (output) neurons.
            batch_size (int, optional): Batch size for processing (default: 1).
            time_bins (int, optional): Number of time bins for event aggregation (default: 100).
                                     This value is fixed to preserve temporal dynamics.
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float32).
            device (str or torch.device, optional): Device for tensor storage (default: "cuda").
            neuron_model_config (dict, optional): Dictionary with keys "model" (class) and "params" (dict of model parameters).
        """

        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.nb_channels = len(self.channels)
        self.batch_size = batch_size
        self.time_bins = time_bins
        self.nb_neurons_fascilce = nb_neurons_fascilce

        self.spk_tensor = torch.zeros(
            (self.batch_size, nb_neurons_fascilce, self.time_bins), dtype=self.dtype, device=self.device)
        self.dense_events = torch.zeros(
            (self.batch_size, self.nb_channels, self.time_bins), dtype=self.dtype, device=self.device)
        self.events = np.empty((0, 2), dtype=float)  # time, channel

        # Use neuron_model_config to instantiate the neuron model
        if neuron_model_config is not None:
            model_cls = neuron_model_config["model"]
            self.neuron_model = model_cls.__name__
            model_params = neuron_model_config["params"]
            self.neuron = model_cls(
                batch_size=self.batch_size,
                nb_inputs=self.nb_channels,
                nb_neurons=self.nb_neurons_fascilce,
                device=self.device,
                dtype=self.dtype,
                **model_params
            )
        else:
            # fallback to default with optimized parameters for real-time
            self.neuron_model = "CuBaLIF"
            self.neuron = CuBaLIF(
                batch_size=self.batch_size,
                nb_inputs=self.nb_channels,
                nb_neurons=self.nb_neurons_fascilce,
                alpha=0.9,
                beta=0.8,
                device=self.device,
                dtype=self.dtype,
                requires_grad=False
            )

    def reset(self):
        """
        Reset the fascicle response by clearing all accumulated events and resetting the neuron model state.

        This method clears the event storage for all fascicles and resets the neuron model and internal tensors,
        preparing the object for a new sequence or epoch.
        """

        self.events = np.empty((0, 2), dtype=float)  # time, channel

        self.neuron.reset()
        self.spk_tensor.fill_(0.0)
        self.dense_events.fill_(0.0)

    def remove_old_events(self, t):
        """
        Remove all events that occurred before time t for all fascicles.

        Args:
            t (float): The cutoff time. Events with time < t are removed.
        """

        self.events = self.events[self.events[:, 0] >= t]

    def compute_fascicle_response(self, event_ticks, time):
        """
        Compute the fascicle response for a given time window (optimized version).

        Converts sparse input events to a dense time-binned representation, processes them through the neuron model,
        and updates the sparse event representation for each fascicle. This function can be called sequentially
        for binned processing, and will accumulate spikes over time unless reset() is called.

        Args:
            event_ticks: Object containing event data for each channel (must have an 'events' attribute).
            time (array-like): Array of time points [t_start, t_end] defining the time window for processing.
        """

        self.events_to_dense_representation_fast(
            event_ticks=event_ticks, time=time)
        self.spk_tensor.fill_(0.0)  # Reset spike tensor for new computation

        # Vectorized processing: process all time steps at once
        if "CuBa" in self.neuron_model:
            # For CuBa models, we need to step through time sequentially due to state dependencies
            for t in range(self.time_bins):
                input_activity_t = self.dense_events[:, :, t]
                spk, _, _ = self.neuron.step(input_activity_t)
                self.spk_tensor[:, :, t] = spk
        else:
            # For simpler models, we can potentially vectorize more
            for t in range(self.time_bins):
                input_activity_t = self.dense_events[:, :, t]
                spk, _ = self.neuron.step(input_activity_t)
                self.spk_tensor[:, :, t] = spk

        self.events_to_sparse_representation_fast(time=time)

    def events_to_dense_representation_fast(self, event_ticks, time):
        """
        Optimized version: Convert sparse event data for each channel into a dense tensor representation over discrete time bins.

        Args:
            event_ticks: Object containing event data for each channel (must have an 'events' attribute).
            time (array-like): Array of time points [t_start, t_end] defining the time window for processing.

        Notes:
            - Uses vectorized operations for better performance
            - Avoids loops where possible
            - Minimizes memory allocations
        """

        self.dense_events.fill_(0.0)  # Reset dense events tensor

        # Early return if no events
        if event_ticks.events.shape[0] == 0:
            return

        t_start, t_end = time[-2], time[-1]
        dt = (t_end - t_start) / self.time_bins

        # Filter events within time window
        event_times = event_ticks.events[:, 0]
        event_channels = event_ticks.events[:, 1]

        # Vectorized time window filtering
        time_mask = (event_times >= t_start) & (event_times < t_end)
        if not np.any(time_mask):
            return

        valid_times = event_times[time_mask]
        valid_channels = event_channels[time_mask]

        # Vectorized bin assignment
        bin_indices = ((valid_times - t_start) / dt).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.time_bins - 1)

        # Convert channel indices to 0-based and filter for our channels
        channel_indices = (valid_channels - 1).astype(np.int32)

        # Create mask for valid channels
        valid_channel_mask = np.isin(channel_indices, self.channels)
        if not np.any(valid_channel_mask):
            return

        final_bin_indices = bin_indices[valid_channel_mask]
        final_channel_indices = channel_indices[valid_channel_mask]

        # Map global channel indices to local indices
        channel_mapping = {ch: i for i, ch in enumerate(self.channels)}
        local_channel_indices = np.array(
            [channel_mapping.get(ch, -1) for ch in final_channel_indices])
        valid_local_mask = local_channel_indices >= 0

        if np.any(valid_local_mask):
            final_bin_indices = final_bin_indices[valid_local_mask]
            local_channel_indices = local_channel_indices[valid_local_mask]

            # Vectorized assignment using advanced indexing
            self.dense_events[0, local_channel_indices,
                              final_bin_indices] = 1.0

    def events_to_dense_representation(self, event_ticks, time):
        """
        Convert sparse event data for each channel into a dense tensor representation over discrete time bins.

        Args:
            event_ticks: Object containing event data for each channel (must have an 'events' attribute).
            time (array-like): Array of time points [t_start, t_end] defining the time window for processing.

        Notes:
            - The dense tensor is filled with 1.0 for each event in the corresponding time bin and channel.
            - Events outside the specified time window are ignored.
        """

        self.dense_events.fill_(0.0)  # Reset dense events tensor
        # Check if there are any events in the event_ticks
        if event_ticks.events.shape[0] == 0:
            # If no events, fill dense_events with zeros
            self.dense_events.fill_(0.0)
        else:
            t_start = time[-2]
            t_end = time[-1]
            bin_edges = np.linspace(t_start, t_end, self.time_bins + 1)
            for channel in self.channels:
                channel_mask = np.where(
                    event_ticks.events[:, 1] == channel + 1)[0]
                times = event_ticks.events[channel_mask, 0]
                # Select events in the current window
                mask = (times >= t_start) & (times < t_end)
                times_in_window = times[mask]
                # Assign each event to a bin
                # -1 to convert to 0-based index
                bin_indices = np.digitize(times_in_window, bin_edges) - 1
                # Clamp indices to valid range
                bin_indices = bin_indices[(bin_indices >= 0)
                                          & (bin_indices < self.time_bins)]
                for b in bin_indices:
                    # Set to 1 for each event
                    self.dense_events[0, channel, b] = 1.0
        pass

    def events_to_sparse_representation_fast(self, time):
        """
        Optimized version: Convert the dense spike tensor output from the neuron layer into a sparse event
        representation for each fascicle.

        Args:
            time (array-like): Array of time points [t_start, t_end] defining the time window for processing.

        Notes:
            - Uses vectorized operations for better performance
            - Avoids loops where possible
            - Minimizes CPU-GPU transfers
        """

        t_min, t_max = time[-2], time[-1]
        dt = (t_max - t_min) / self.time_bins

        # Find all spikes at once using vectorized operations
        spike_indices = torch.nonzero(self.spk_tensor[0, :, :], as_tuple=False)

        if spike_indices.numel() == 0:
            return

        # Extract fascicle and time indices
        fascicle_indices = spike_indices[:, 0]  # Shape: (num_spikes,)
        time_indices = spike_indices[:, 1]      # Shape: (num_spikes,)

        # Convert to spike times (vectorized)
        spike_times = time_indices.float() * dt + t_min

        # Convert to CPU only once
        spike_times_cpu = spike_times.cpu().numpy()
        fascicle_indices_cpu = (
            fascicle_indices + 1).cpu().numpy()  # Convert to 1-based

        # Create events array in one go
        if len(spike_times_cpu) > 0:
            new_events = np.column_stack(
                (spike_times_cpu, fascicle_indices_cpu))

            # Use more efficient concatenation
            if self.events.shape[0] == 0:
                self.events = new_events
            else:
                self.events = np.vstack((self.events, new_events))

    def events_to_sparse_representation(self, time):
        """
        Convert the dense spike tensor output from the neuron layer into a sparse event
        representation for each fascicle.

        Args:
            time (array-like): Array of time points [t_start, t_end] defining the time window for processing.

        Notes:
            - The output events are appended to self.events and are not cleared unless reset() is called.
            - Each event is represented as [event_time, fascicle_index].
        """

        t_min = time[-2]
        t_max = time[-1]
        dt = (t_max - t_min) / self.time_bins

        # Create a sparse representation of the events
        for fascicle in range(self.nb_neurons_fascilce):
            if torch.sum(self.spk_tensor[:, fascicle, :]) > 0:
                # , out=self.spk_tensor[:, fascicle, :]*dt
                spike_times = torch.where(
                    self.spk_tensor[:, fascicle, :] > 0)[-1]
                spike_times = spike_times * dt + t_min
                spike_times = spike_times.cpu().numpy()
                # Create a sparse event representation for this fascicle
                fascicle_events = np.column_stack(
                    (spike_times, np.full(spike_times.shape, fascicle + 1)))
                # Append to the global events array
                self.events = np.append(self.events, fascicle_events, axis=0)
        pass
        # Create a sparse representation of the events
        for fascicle in range(self.nb_neurons_fascilce):
            if torch.sum(self.spk_tensor[:, fascicle, :]) > 0:
                # , out=self.spk_tensor[:, fascicle, :]*dt
                spike_times = torch.where(
                    self.spk_tensor[:, fascicle, :] > 0)[-1]
                spike_times = spike_times * dt + t_min
                spike_times = spike_times.cpu().numpy()
                # Create a sparse event representation for this fascicle
                fascicle_events = np.column_stack(
                    (spike_times, np.full(spike_times.shape, fascicle + 1)))
                # Append to the global events array
                self.events = np.append(self.events, fascicle_events, axis=0)
                pass
                fascicle_events = np.column_stack(
                    (spike_times, np.full(spike_times.shape, fascicle + 1)))
                # Append to the global events array
                self.events = np.append(self.events, fascicle_events, axis=0)
        pass
