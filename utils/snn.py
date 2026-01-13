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
    def backward(ctx, grad_output):
        # Straight-through estimator: gradient passes through unchanged
        return grad_output.clone(), None


ste_fn = STEFunction.apply


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
    use_eprop : bool
        Whether to use e-prop (True) or BPTT (False)
    use_linear_decay : bool
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

    def __init__(self, nb_inputs, nb_neurons, batch_size, fwd_weight_scale, alpha, beta, use_eprop=False, use_linear_decay=False, device=torch.device("cuda:0"), dtype=torch.float64, ref_per=None):
        """
        Initialize feedforward spiking layer.

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
        use_eprop : bool, optional
            Whether to use e-prop (True) or BPTT (False) (default: False)
        use_linear_decay : bool, optional
            Use linear decay instead of exponential (default: False)
        device : str, optional
            Device for tensor allocation (default: "cuda:0")
        dtype : torch.dtype, optional
            Data type for tensors (default: torch.float64)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
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

        # Learning and simulation flags
        self.use_eprop = use_eprop
        self.use_linear_decay = use_linear_decay

        # Device and dtype
        self.device = device
        self.dtype = dtype

        # Optional features
        self.ref_per = ref_per
        if ref_per is not None or ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=self.device, dtype=torch.int)
        self.create_layer()

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
        # Only operate on the current batch slice
        batch_slice = self.ref_per_tensor[:current_batch_size,
                                          :current_neurons]
        batch_slice[batch_slice > 0.0] -= 1
        batch_slice[spk > 0.0] = self.ref_per
        self.ref_per_tensor[:current_batch_size,
                            :current_neurons] = batch_slice

    def create_layer(self):
        """
        Initialize feedforward weight matrix.

        Creates and initializes the weight matrix with Gaussian distribution.

        Notes
        -----
        - Feedforward weights: [nb_neurons, nb_inputs]
        - Initialization: N(0, fwd_weight_scale/sqrt(nb_inputs))
        - Requires gradient for learning
        """
        self.ff_weights = torch.empty((self.nb_neurons, self.nb_inputs),
                                      device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.fwd_weight_scale / (self.nb_inputs ** 0.5))

    def compute_activity(self, input_activity, nb_steps, lower_bound=None):
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
        - Reset mechanism: multiplicative (voltage * (1 - spike))
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
        if self.ref_per is not None:
            self.reset_refractory_perdiod_counter()

        mem_rec = []
        spk_rec = []
        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            # Use surrogate gradient for BPTT compatibility
            if self.use_eprop:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient
                out = spike_fn(mthr)
            rst = out.detach()

            # update the correct counter
            if self.ref_per is not None:
                self.update_refractory_perdiod_counter(rst)
                # take care of last batch
                mask = self.ref_per_tensor[:syn.shape[0], :syn.shape[1]] == 0.0
                new_syn = self.alpha * syn
                new_syn[mask] = (self.alpha*syn[mask] +
                                 input_activity[:, t][mask])
            else:
                new_syn = self.alpha*syn + input_activity[:, t]

            if self.use_linear_decay:
                # torch.sign returns: 1 if x > 0, -1 if x < 0, and 0 if x == 0
                new_mem = ((mem-torch.sign(mem)*self.beta) + syn)*(1.0-rst)
            else:
                new_mem = (self.beta*mem + syn)*(1.0-rst)
            if lower_bound:
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
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
    use_eprop : bool
        Whether to use e-prop (True) or BPTT (False)
    use_linear_decay : bool
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

    def __init__(self, nb_inputs, nb_neurons, batch_size, fwd_weight_scale, rec_weight_scale, alpha, beta, use_eprop=False, use_linear_decay=False, device=torch.device("cuda:0"), dtype=torch.float64, ref_per=None):
        """
        Initialize recurrent spiking layer.

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
        use_eprop : bool, optional
            Whether to use e-prop (True) or BPTT (False) (default: False)
        use_linear_decay : bool, optional
            Use linear decay instead of exponential (default: False)
        device : str, optional
            Device for tensor allocation (default: "cuda:0")
        dtype : torch.dtype, optional
            Data type for tensors (default: torch.float64)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
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

        # Learning and simulation flags
        self.use_eprop = use_eprop
        self.use_linear_decay = use_linear_decay

        # Device and dtype
        self.device = device
        self.dtype = dtype

        # Optional features
        self.ref_per = ref_per
        if ref_per is not None or ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=self.device, dtype=torch.int)
        self.create_layer()

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
        # Only operate on the current batch slice
        batch_slice = self.ref_per_tensor[:current_batch_size,
                                          :current_neurons]
        batch_slice[batch_slice > 0.0] -= 1
        batch_slice[spk > 0.0] = self.ref_per
        self.ref_per_tensor[:current_batch_size,
                            :current_neurons] = batch_slice

    def create_layer(self):
        """
        Initialize feedforward and recurrent weight matrices.

        Creates and initializes both weight matrices with Gaussian distributions.

        Notes
        -----
        - Feedforward weights: [nb_neurons, nb_inputs]
          Initialization: N(0, fwd_weight_scale/sqrt(nb_inputs))
        - Recurrent weights: [nb_neurons, nb_neurons]
          Initialization: N(0, rec_weight_scale/sqrt(nb_neurons))
        - Both require gradients for learning
        """
        self.ff_weights = torch.empty((self.nb_neurons, self.nb_inputs),
                                      device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.fwd_weight_scale / (self.nb_inputs ** 0.5))
        self.rec_weights = torch.empty((self.nb_neurons, self.nb_neurons),
                                       device=self.device, dtype=self.dtype, requires_grad=True)
        torch.nn.init.normal_(self.rec_weights, mean=0.0,
                              std=self.rec_weight_scale / (self.nb_neurons ** 0.5))

    def compute_activity(self, input_activity, nb_steps, lower_bound=None):
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
        - Reset mechanism: multiplicative (voltage * (1 - spike))
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
        if self.ref_per is not None:
            self.reset_refractory_perdiod_counter()

        mem_rec = []
        spk_rec = []

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + \
                torch.einsum("ab,bc->ac", (out, self.rec_weights.t()))
            mthr = mem-1.0
            # Use surrogate gradient for BPTT compatibility
            if self.use_eprop:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient
                out = spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            if self.ref_per is not None:
                self.update_refractory_perdiod_counter(rst)
                # only update the membrane potential if not in refractory period
                # take care of last batch
                mask = self.ref_per_tensor[:syn.shape[0], :syn.shape[1]] == 0.0
                new_syn = self.alpha * syn
                new_syn[mask] = (self.alpha*syn[mask] + h1[mask])
            else:
                new_syn = self.alpha*syn + h1

            if self.use_linear_decay:
                new_mem = ((mem-torch.sign(mem)*self.beta) + syn)*(1.0-rst)
            else:
                new_mem = (self.beta*mem + syn)*(1.0-rst)

            if lower_bound:
                # clamp membrane potential
                new_mem[new_mem < lower_bound] = lower_bound

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


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
        - 'use_random_tie_breaking' : bool
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
    >>> params = {'use_random_tie_breaking': True}
    >>> summed_spikes, neuron_idc = compute_winning_neuron(spikes, params)
    >>> print(summed_spikes)  # spike counts: tensor([[1, 2]])
    >>> print(neuron_idc)  # prediction: tensor([1])  # neuron 1 has more spikes
    """

    summed_spikes = torch.sum(spk_rec_readout,
                              dim=1)  # sum over time: [batch, output_neurons]

    # Select winner based on spike counts
    max_nb_spikes, neuron_idc = torch.max(
        summed_spikes, dim=1)  # argmax over output units
    if params['use_random_tie_breaking']:
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
    in the training loop based on params["use_eprop"] setting.

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
    rec_layer, ff_layer = layers

    if params["nb_input_copies"] > 1:
        h1 = torch.einsum(
            "abc,cd->abd", (inputs.tile((params["nb_input_copies"],)), rec_layer.ff_weights.t()))
    else:
        h1 = torch.einsum(
            "abc,cd->abd", inputs, rec_layer.ff_weights.t())

    spk_rec_hidden, mem_rec_hidden = rec_layer.compute_activity(
        h1, params['data_steps'], params["lower_bound"])

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec_hidden, ff_layer.ff_weights.t()))

    spk_rec_readout, mem_rec_readout = ff_layer.compute_activity(
        h2, params['data_steps'], params["lower_bound"])

    other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout]

    return spk_rec_readout, other_recs
