import torch
import numpy as np


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


class FeedforwardLayer:
    """
    Class to initialize and compute spiking feedforward layer of CUBA LIF neurons.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, alpha, beta, device, dtype):
        """
        Initialize the feedforward layer with weights and parameters.

        Args:
            nb_inputs (int): Number of input channels.
            nb_neurons (int): Number of neurons.
            fwd_scale (float): Scaling factor for weight initialization.
            alpha (float): Synaptic decay constant.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        """
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype

        # Initialize the feedforward layer weights
        self.ff_layer = torch.empty(
            (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_layer, mean=0.0,
                              std=fwd_scale / np.sqrt(nb_inputs))

        # Initialize the synaptic current and membrane potential
        self.syn = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)
        self.rst = torch.zeros((batch_size, nb_neurons),
                               device=device, dtype=dtype)

        self.syn_rec = []
        self.mem_rec = []
        self.out_rec = []

    def update(self, input_activity_t):
        """
        Compute the activity of the feedforward layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            out (torch.Tensor): Spike output of shape (batch_size, nb_outputs).
            syn (torch.Tensor): Updated synaptic current tensor.
            mem (torch.Tensor): Updated membrane potential tensor.
        """

        self.syn = self.alpha * self.syn + \
            torch.einsum("ab,bc->ac", input_activity_t, self.ff_layer)
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        self.syn_rec.append(self.syn.detach().cpu().numpy())
        self.mem_rec.append(self.mem.detach().cpu().numpy())
        self.out_rec.append(self.rst.cpu().numpy())

        return self.rst, self.syn, self.mem

    def results(self):
        """
        Return the recorded spikes, membrane potentials, and synaptic currents.

        This method retrieves the recorded spike outputs, membrane potentials, 
        and synaptic currents from the feedforward layer. The returned values 
        are converted to numpy arrays for further analysis or visualization.

        Returns:
            tuple:
                - spikes (numpy.ndarray): Recorded spike outputs of shape (timesteps, batch_size, nb_neurons).
                - membrane_potentials (numpy.ndarray): Recorded membrane potentials of shape (timesteps, batch_size, nb_neurons).
                - synaptic_currents (numpy.ndarray): Recorded synaptic currents of shape (timesteps, batch_size, nb_neurons).
        """
        return np.array(self.out_rec), np.array(self.mem_rec), np.array(self.syn_rec)


class RecurrentLayer:
    """
    Class to initialize and compute spiking recurrent layer of CUBA LIF neurons.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, rec_scale, alpha, beta, device, dtype):
        """
        Initialize the recurrent layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of recurrent neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            alpha (float): Synaptic decay constant.
            beta (float): Membrane decay constant.
            device (torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        """
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.dtype = dtype

        # Initialize feedforward and recurrent weights
        self.ff_layer = torch.empty(
            (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_layer, mean=0.0,
                              std=fwd_scale / np.sqrt(nb_inputs))

        self.rec_layer = torch.empty(
            (nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.rec_layer, mean=0.0,
                              std=rec_scale / np.sqrt(nb_neurons))

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

        # Recordings
        self.syn_rec = []
        self.mem_rec = []
        self.out_rec = []

    def update(self, input_activity_t):
        """
        Compute the activity of the recurrent layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            out (torch.Tensor): Spike output of shape (batch_size, nb_neurons).
        """
        # Compute input and recurrent contributions
        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_layer) + \
            torch.einsum("ab,bc->ac", self.rst, self.rec_layer)

        # Update synaptic current and membrane potential
        self.syn = self.alpha * self.syn + h1
        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()  # Reset spikes
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        # Record values
        self.syn_rec.append(self.syn.detach().cpu().numpy())
        self.mem_rec.append(self.mem.detach().cpu().numpy())
        self.out_rec.append(self.rst.cpu().numpy())

        return self.rst, self.syn, self.mem

    def results(self):
        """
        Return the recorded spikes, membrane potentials, and synaptic currents.

        Returns:
            tuple:
                - spikes (numpy.ndarray): Recorded spike outputs of shape (timesteps, batch_size, nb_neurons).
                - membrane_potentials (numpy.ndarray): Recorded membrane potentials of shape (timesteps, batch_size, nb_neurons).
                - synaptic_currents (numpy.ndarray): Recorded synaptic currents of shape (timesteps, batch_size, nb_neurons).
        """
        return np.array(self.out_rec), np.array(self.mem_rec), np.array(self.syn_rec)


# Define parameters
nb_inputs = 100  # Number of input neurons
nb_ff_neurons = 50  # Number of output neurons
nb_rec_neurons = 100  # Number of recurrent neurons
fwd_scale = 0.1  # Scaling factor for weight initialization
rec_scale = 0.2  # Scaling factor for weight initialization
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available
dtype = torch.float  # Data type for tensors
nb_steps = 1000  # Number of time steps
alpha = 0.9  # Synaptic decay constant
beta = 0.8  # Membrane decay constant

# Create input activity tensor (batch_size, nb_steps, nb_inputs)
batch_size = 32
input_activity = torch.rand(
    (batch_size, nb_steps, nb_inputs), device=device, dtype=dtype)

# Instantiate the feedforward layer
feedforward_layer = FeedforwardLayer(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_ff_neurons,
                                     fwd_scale=fwd_scale, alpha=alpha, beta=beta, device=device, dtype=dtype)
recurrent_layer = RecurrentLayer(batch_size=batch_size, nb_inputs=nb_ff_neurons, nb_neurons=nb_rec_neurons,
                                 fwd_scale=fwd_scale, rec_scale=rec_scale, alpha=alpha, beta=beta, device=device, dtype=dtype)

# Loop over time steps
for t in range(nb_steps):
    # Extract input activity for the current time step
    input_activity_t = input_activity[:, t]
    ff_rst, _, _ = feedforward_layer.update(input_activity_t)
    rec_rst, _, _ = recurrent_layer.update(ff_rst)


# Stack the recorded spikes and membrane potentials
ff_out = feedforward_layer.results()
ff_spk_rec = ff_out[0]  # Spike recordings
ff_mem_rec = ff_out[1]  # Membrane potential recordings
ff_syn_rec = ff_out[2]  # Synaptic current recordings

rec_out = recurrent_layer.results()
rec_spk_rec = rec_out[0]  # Spike recordings
rec_mem_rec = rec_out[1]  # Membrane potential recordings
rec_syn_rec = rec_out[2]  # Synaptic current recordings
