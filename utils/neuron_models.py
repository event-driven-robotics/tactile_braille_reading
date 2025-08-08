import numpy as np
import torch


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 15.0
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
        ctx.save_for_backward(input, possible_weight_values, min_indices)
        return possible_weight_values[min_indices]

    @staticmethod
    def backward(ctx, grad_output):
        input, possible_weight_values, min_indices = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None


ste_fn = STEFunction.apply


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
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def step(self, input_activity_t):
        """
        Compute the activity of the feedforward CuBaLIF layer for a single time step.

        The synaptic current is updated using a leaky integration of the weighted input, and the membrane potential is updated using a leaky integration of the synaptic current. A spike is emitted if the membrane potential crosses the threshold, and the reset state is updated accordingly.

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

    def step(self, input_activity_t):
        """
        Compute the activity of the recurrent CuBaRLIF layer for a single time step.

        The synaptic current is updated using a leaky integration of the weighted input and recurrent activity, and the membrane potential is updated using a leaky integration of the synaptic current. A spike is emitted if the membrane potential crosses the threshold, and the reset state is updated accordingly.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        # Compute input and recurrent contributions
        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights) + \
            torch.einsum("ab,bc->ac", self.rst, self.rec_weights)

        # Update synaptic current and membrane potential
        self.syn = self.alpha * self.syn + h1
        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()  # Reset spikes
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        # Record values
        # self.syn_rec.append(self.syn.detach().cpu().numpy())
        # self.mem_rec.append(self.mem.detach().cpu().numpy())
        # self.out_rec.append(self.rst.cpu().numpy())

        return self.rst, self.syn, self.mem


class CuBaLIF_HW_Aware:
    """
    Class to initialize and compute spiking feedforward layer of CUBA LIF neurons.

    This class implements a feedforward layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons.
    It supports the computation of synaptic currents, membrane potentials, and spike outputs over time.
    The layer uses surrogate gradients for backpropagation through spikes.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of feedforward neurons.
        alpha (float): Synaptic decay constant.
        beta (float): Membrane decay constant.
        device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_layer (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons).
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, alpha, firing_threshold, beta, device, dtype, lower_bound=None, ref_per_timesteps=None, weights=None, requires_grad=True):
        """
        Initialize the feedforward layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of feedforward neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            alpha (float): Synaptic decay constant.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.device = device
        self.dtype = dtype

        self.firing_threshold = firing_threshold * \
            torch.ones((batch_size, self.nb_neurons),
                       device=device, dtype=dtype)

        if self.ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, nb_neurons), device=device, dtype=dtype)

        if weights is not None:
            self.ff_weights = weights[0]
        else:
            # Initialize feedforward and recurrent weights
            self.ff_weights = torch.empty(
                (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)
            torch.nn.init.normal_(self.ff_weights, mean=0.0,
                                  std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def update_refractory_period_counter(self):
        """
        Fully vectorized refractory-period decrement + reset.
        """
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(self.rst > 0,
                                           self.ref_per_timesteps,
                                           self.ref_per_counter)
        return self.ref_per_counter

    def reset(self):
        """
        Reset the synaptic current, membrane potential, and reset state tensors.
        This is useful for reinitializing the layer at the start of a new sequence or epoch.
        """
        self.syn.zero_()
        self.mem.zero_()
        self.rst.zero_()

        if self.ref_per_timesteps is not None:
            self.ref_per_counter.zero_()

    def step(self, input_activity_t):
        """
        Compute the activity of the feedforward layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output of shape (batch_size, nb_neurons).
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights)
        self.syn = self.syn[:h1.shape[0], :]
        self.mem = self.mem[:h1.shape[0], :]
        self.rst = self.rst[:h1.shape[0], :]

        mthr = self.mem - self.firing_threshold
        out = spike_fn(mthr)
        self.rst = out.detach()

        if self.ref_per_timesteps is not None:
            # 1) decrement and reset the refractory counter in one kernel
            self.update_refractory_period_counter()
            # 2) update syn in two in‐place ops (fast on GPU)
            mask_ready = (self.ref_per_counter == 0).float()
            # syn = alpha * syn
            self.syn.mul_(self.alpha)
            # syn += h1 * mask_ready
            self.syn.addcmul_(mask_ready, h1, value=1.0)

        else:
            self.syn = self.alpha*self.syn + h1

        # 3) membrane update as usual
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        if self.lower_bound:
            # clamp membrane potential
            self.mem = torch.clamp(self.mem, min=self.lower_bound)

        return out, self.syn, self.mem


class CuBaRLIF_HW_Aware:
    """
    Class to initialize and compute spiking recurrent layer of CUBA LIF neurons.

    This class implements a recurrent layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons.
    It supports the computation of synaptic currents, membrane potentials, and spike outputs over time,
    with both feedforward and recurrent connections. The layer uses surrogate gradients for backpropagation
    through spikes.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of recurrent neurons.
        alpha (float): Synaptic decay constant.
        firing_threshold (torch.Tensor): Firing threshold tensor of shape (batch_size, nb_neurons).
        beta_thr (float): Threshold decay constant for ALIF neuron.
        dump_thr (float): Dumping threshold for ALIF neuron.
        beta (float): Membrane decay constant.
        device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_layer (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        rec_layer (torch.Tensor): Recurrent weight matrix of shape (nb_neurons, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons).
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, rec_scale, alpha, firing_threshold, beta_thr, dump_thr, beta, device, dtype, n_alif=0, lower_bound=None, ref_per_timesteps=None, weights=None, requires_grad=True):
        """
        Initialize the recurrent layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of recurrent neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            alpha (float): Synaptic decay constant.
            firing_threshold (float): Firing threshold for neurons.
            beta_thr (float): Threshold decay constant for ALIF neuron.
            dump_thr (float): Dumping threshold for ALIF neuron.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
            n_alif (int): Number of ALIF neurons in the layer.
            lower_bound (float): Lower bound for membrane potential.
            ref_per_timesteps (int): Refractory period in time steps.
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.n_alif = n_alif
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.alpha = alpha
        self.beta = beta
        self.beta_thr = beta_thr
        self.dump_thr = dump_thr
        self.device = device
        self.dtype = dtype

        self.firing_threshold = firing_threshold * \
            torch.ones((batch_size, self.nb_neurons),
                       device=device, dtype=dtype)

        if ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, nb_neurons), device=device, dtype=dtype)

        # Initialize feedforward and recurrent weights
        if weights is not None:
            # TODO ensure, that weights can be loaded/set seperatly. Now both must be set or not.
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
        # self.rec_layer[torch.arange(nb_neurons),
        #                torch.arange(nb_neurons)] = 0.0
        self.a_thr = torch.zeros((batch_size, n_alif),
                                 device=device, dtype=dtype)
        # Initialize synaptic current, membrane potential, and spike output
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def update_refractory_period_counter(self):
        """
        Fully vectorized refractory-period decrement + reset.
        """
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(self.rst > 0,
                                           self.ref_per_timesteps,
                                           self.ref_per_counter)
        return self.ref_per_counter

    def reset(self):
        """
        Reset the synaptic current, membrane potential, and reset state tensors.
        This is useful for reinitializing the layer at the start of a new sequence or epoch.
        """
        self.syn.zero_()
        self.mem.zero_()
        self.rst.zero_()
        self.a_thr.zero_()

        if self.ref_per_timesteps is not None:
            self.ref_per_counter.zero_()

    def step(self, input_activity_t):
        """
        Compute the activity of the recurrent layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output of shape (batch_size, nb_neurons).
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        # Compute input and recurrent contributions
        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights) + \
            torch.einsum("ab,bc->ac", self.rst, self.rec_weights)

        self.a_thr = self.a_thr[:h1.shape[0], :]
        self.syn = self.syn[:h1.shape[0], :]
        self.mem = self.mem[:h1.shape[0], :]
        self.rst = self.rst[:h1.shape[0], :]
        self.firing_threshold = self.firing_threshold[:h1.shape[0], :]

        if (self.n_alif > 0):
            # Update synaptic current and membrane potential
            # a[t+1] = rho*a[t] + s[t], a[t] = rho*a[t-1] + s[t-1]
            self.a_thr = (self.beta_thr*self.a_thr) + self.rst[:, :self.n_alif]
            self.firing_threshold[:, :self.n_alif] = self.firing_threshold + \
                self.dump_thr * self.a_thr  # A[t] = v_th + beta*a[t]

        mthr = self.mem - self.firing_threshold
        out = spike_fn(mthr)
        self.rst = out.detach()  # Reset spikes

        if self.ref_per_timesteps is not None:
            # 1) decrement and reset the refractory counter in one kernel
            self.update_refractory_period_counter()
            # 2) update syn in two in‐place ops (fast on GPU)
            mask_ready = (self.ref_per_counter == 0).float()
            # syn = alpha * syn
            self.syn.mul_(self.alpha)
            # syn += h1 * mask_ready
            self.syn.addcmul_(mask_ready, h1, value=1.0)
        else:
            self.syn = self.alpha*self.syn + h1

        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        if self.lower_bound:
            # clamp membrane potential
            self.mem = torch.clamp(self.mem, min=self.lower_bound)

        return out, self.syn, self.mem


class CuBaLIF_HW_Aware_OG:
    """
    Class to initialize and compute spiking feedforward layer of CUBA LIF neurons.

    This class implements a feedforward layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons.
    It supports the computation of synaptic currents, membrane potentials, and spike outputs over time.
    The layer uses surrogate gradients for backpropagation through spikes.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of feedforward neurons.
        alpha (float): Synaptic decay constant.
        beta (float): Membrane decay constant.
        device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_layer (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons).
        syn_rec (list): List to record synaptic currents over time.
        mem_rec (list): List to record membrane potentials over time.
        out_rec (list): List to record spike outputs over time.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, alpha, firing_threshold, beta, device, dtype, lower_bound=None, ref_per_timesteps=None, weights=None, requires_grad=True):
        """
        Initialize the feedforward layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of feedforward neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            alpha (float): Synaptic decay constant.
            firing_threshold (float): Firing threshold for neurons.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
            lower_bound (float): Lower bound for membrane potential.
            ref_per_timesteps (int): Refractory period in time steps.
            weights (torch.Tensor, optional): Predefined weights.
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.device = device
        self.dtype = dtype
        self.theta = firing_threshold
        self.firing_threshold = firing_threshold * torch.ones((batch_size, self.nb_neurons), device = device, dtype=dtype)

        if self.ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, nb_neurons), device=device, dtype=dtype)


        if weights is not None:
            self.ff_weights = torch.nn.Parameter(weights.to(device=device, dtype=dtype))
        else:
            # Initialize feedforward
            self.ff_weights = torch.empty((nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)

            torch.nn.init.normal_(self.ff_weights, mean=0.0, std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.syn     = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem     = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst     = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.new_mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.n_spike = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

    def step(self, input_activity_t):
        """
        Compute the activity of the feedforward layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output of shape (batch_size, nb_neurons).
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """
        mthr = self.mem - self.firing_threshold
        out = spike_fn(mthr)

        self.n_spike[out == 1.0] = self.n_spike[out == 1.0] + 1

        self.rst = out.detach()

        if self.ref_per_timesteps is not None:
            self.update_refractory_period_counter()
            # only update the membrane potential if not in refractory period
            # take care of last batch
            mask = self.ref_per_counter[:self.syn.shape[0], :self.syn.shape[1]] == 0.0
            new_syn = self.alpha * self.syn
            new_syn[mask] = (self.alpha*self.syn[mask] + input_activity_t[mask])
        else:
            new_syn = self.alpha*self.syn + input_activity_t

        self.mem = (self.beta*self.mem + self.syn)*(1.0-self.rst)

        if self.lower_bound:
            # clamp membrane potential
            self.mem[self.mem < self.lower_bound] = self.lower_bound

        self.syn = new_syn
        return out.clone(), self.syn, self.mem, self.n_spike


    def update_refractory_period_counter(self):
        """
        Fully vectorized refractory‐period decrement + reset.
        """
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(self.rst > 0,
                                           self.ref_per_timesteps,
                                           self.ref_per_counter)
        return self.ref_per_counter

class CuBaRLIF_HW_Aware_OG:
    """
    Class to initialize and compute spiking recurrent layer of CUBA LIF neurons.

    This class implements a recurrent layer of Current-Based Leaky Integrate-and-Fire (CUBA LIF) neurons.
    It supports the computation of synaptic currents, membrane potentials, and spike outputs over time,
    with both feedforward and recurrent connections. The layer uses surrogate gradients for backpropagation
    through spikes.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of recurrent neurons.
        alpha (float): Synaptic decay constant.
        firing_threshold (torch.Tensor): Firing threshold tensor of shape (batch_size, nb_neurons).
        beta_thr (float): Threshold decay constant for ALIF neuron.
        dump_thr (float): Dumping threshold for ALIF neuron.
        beta (float): Membrane decay constant.
        device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_layer (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        rec_layer (torch.Tensor): Recurrent weight matrix of shape (nb_neurons, nb_neurons).
        syn (torch.Tensor): Synaptic current tensor of shape (batch_size, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons).
        syn_rec (list): List to record synaptic currents over time.
        mem_rec (list): List to record membrane potentials over time.
        out_rec (list): List to record spike outputs over time.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, rec_scale, alpha, firing_threshold, beta, device, dtype, lower_bound=None, ref_per_timesteps=None, weights=None, requires_grad=True):
        """
        Initialize the recurrent layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of recurrent neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            alpha (float): Synaptic decay constant.
            firing_threshold (float): Firing threshold for neurons.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
            lower_bound (float): Lower bound for membrane potential.
            ref_per_timesteps (int): Refractory period in time steps.
        """

        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype
        self.theta = firing_threshold
        self.firing_threshold = self.theta * torch.ones((batch_size, self.nb_neurons), device = device, dtype=dtype)

        if self.ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, nb_neurons), device=device, dtype=dtype)


        if weights is not None:
            self.ff_weights = torch.nn.Parameter(weights[0].to(device=device, dtype=dtype))
            self.rec_weights = torch.nn.Parameter(weights[1].to(device=device, dtype=dtype))
        else:
            # Initialize feedforward and recurrent weights
            self.ff_weights = torch.empty((nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)

            torch.nn.init.normal_(self.ff_weights, mean=0.0, std=fwd_scale / np.sqrt(nb_inputs))

            self.rec_weights = torch.empty((nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=requires_grad)

            torch.nn.init.normal_(self.rec_weights, mean=0.0, std=fwd_scale*rec_scale / np.sqrt(nb_inputs))

        # # ensure, that recurrent connections to a neuron itself are zero (no self connections)
        # self.rec_layer[torch.arange(nb_neurons),
        #                torch.arange(nb_neurons)] = 0.0
        # Initialize synaptic current, membrane potential, and spike output
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.new_mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

        self.out = torch.zeros((batch_size, nb_neurons),
                                 device=device, dtype=dtype)

    def step(self, input_activity_t, rec_weights):
        """
        Compute the activity of the recurrent layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output of shape (batch_size, nb_neurons).
                - syn (torch.Tensor): Updated synaptic current tensor of shape (batch_size, nb_neurons).
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """
        # Compute input and recurrent contributions
        h1 = input_activity_t + \
            torch.einsum("ab,bc->ac", self.out[:input_activity_t.shape[0],:], rec_weights)

        mthr = self.mem - self.firing_threshold
        self.out = spike_fn(mthr)
        self.rst = self.out.detach()  # Reset spikes

        if self.ref_per_timesteps is not None:
            self.update_refractory_period_counter()
            # only update the membrane potential if not in refractory period
            # take care of last batch
            mask = self.ref_per_counter[:self.syn.shape[0], :self.syn.shape[1]] == 0.0
            new_syn = self.alpha * self.syn
            new_syn[mask] = (self.alpha*self.syn[mask] + h1[mask])
        else:
            new_syn = self.alpha*self.syn + h1

        self.mem = (self.beta*self.mem + self.syn)*(1.0-self.rst)

        if self.lower_bound:
            # clamp membrane potential
            self.mem[self.mem < self.lower_bound] = self.lower_bound
        # Record values
        # self.syn_rec.append(self.syn.detach().cpu().numpy())
        # self.mem_rec.append(self.mem.detach().cpu().numpy())
        # self.out_rec.append(self.rst.cpu().numpy())

        self.syn = new_syn

        return self.out.clone(), self.syn, self.mem


    def update_refractory_period_counter(self):
        """
        Fully vectorized refractory‐period decrement + reset.
        """
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(self.rst > 0,
                                           self.ref_per_timesteps,
                                           self.ref_per_counter)
        return self.ref_per_counter
