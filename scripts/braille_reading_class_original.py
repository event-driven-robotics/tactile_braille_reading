import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch, argparse
import torch.nn as nn
from matplotlib.gridspec import GridSpec  # can be used for nice subplot layout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import pickle

torch.cuda.empty_cache()
#torch.autograd.set_detect_anomaly(True)

global default_device
default_device ='cuda:0'

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# set variables
use_seed = True
threshold = 2  # possible values are: 1, 2, 5, 10

# create folder to safe figures later
path = './figures'
isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

neg_capacitance = torch.arange(255, -1, -1)

pos_capacitance = torch.arange(1, 257)
diff_cap = pos_capacitance - neg_capacitance
diff_cap = diff_cap.to(torch.device(default_device))
q = 1/256

factor = 10 ** 3

int_values = (diff_cap * factor * q).to(torch.int64)  # this avoids float rounding errors check what to do
possible_weight = int_values.to(torch.float64) / factor

def default_parser(parser):
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--time_bin_size', type=int, default=3, help='time bin size in ms (default: 3)')
    parser.add_argument('--max_time', type=int, default=3501, help='max time in ms (default: 3501)')
    parser.add_argument('--device', type=str, default=default_device, help='device to use change to cpu if no gpu available')
    parser.add_argument('--dtype', type=str, default='torch.float', help='data type to use (default: floa)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 4)')
    parser.add_argument('--batch_size_test', type=int, default=128, help='input batch size for test (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0015, help='learning rate (default:0.0008)')
    parser.add_argument('--tau_mem', type=float, default=0.06, help='membrane time constant (default: 0.06)')
    parser.add_argument('--tau_syn', type=float, default=0.0, help='synaptic time constant (default: 0.0)')
    parser.add_argument('--lower_bound', type=float, default=-1.0, help='lower bound for membrane potential (default: -1.0)')
    parser.add_argument('--ref_per_timesteps', type=int, default=1, help='refractory period in time steps (default: 1)')
    parser.add_argument('--eprop', type=bool, default=False, help='use event propagation as alghorithm (default: False)')
    parser.add_argument('--tau_trace', type=float, default=0.08, help='trace decay (default: 0.08)')
    parser.add_argument('--tau_trace_rec', type=float, default=0.105, help='trace decay (default: 0.105)')
    parser.add_argument('--fwd_weight_scale', type=float, default=1, help='forward weight scale for initialization (default: 1)')
    parser.add_argument('--weight_scale_factor', type=float, default=0.02, help='recurrent weight scale for initialization (default: 0.02)')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma factor for surrogate gradient (default: 0.3)')
    parser.add_argument('--reg_spikes', type=float, default=0.0015, help='regularization factor for spikes (default: 0.0015)')
    parser.add_argument('--reg_neurons', type=float, default=0.0, help='regularization factor for neurons (default: 0.0)')
    parser.add_argument('--firing_threshold', type=float, default=1.0, help='firing threshold (default: 1.0)')
    parser.add_argument('--quantization', type=bool, default=False, help='use quantization of weights (default: False)')

parser = argparse.ArgumentParser(description='Training a RSNN on BRAILLE')
default_parser(parser)
global dict_args
dict_args = vars(parser.parse_args())

dict_args.update({"eprop": True})
dict_args.update({"batch_size": 4})
dict_args.update({"lr": 0.0008})
dict_args.update({"quantization": True})
dict_args.update({"batch_size_test": 128})

# use fixed seed for reproducable results
if use_seed:
    seed = 42  # "Answer to the Ultimate Question of Life, the Universe, and Everything"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")


def load_and_extract(params, file_name, taxels=None, letter_written=letters):

    max_time = int(params['max_time'])  # ms
    time_bin_size = int(params['time_bin_size'])  # so far from laoded file, but can be set manually here

    data_steps = len(range(0, max_time, time_bin_size))

    global delayed_output
    delayed_output = data_steps

    data_dict = pd.read_pickle(file_name)
    data_dict = pd.DataFrame(data_dict)
    # Extract data
    data = []
    labels = []
    bins = 1000  # ms conversion
    nchan = len(data_dict['events'][1])  # number of channels per sensor
    # loop over all trials
    for i, sample in enumerate(data_dict['events']):
        events_array = np.zeros(
            [nchan, round((max_time/time_bin_size)+0.5), 2])
        # loop over sensors (taxel)
        for taxel in range(len(sample)):
            # loop over On and Off channels
            for event_type in range(len(sample[taxel])):
                if sample[taxel][event_type]:
                    indx = bins*(np.array(sample[taxel][event_type]))
                    indx = np.array((indx/time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[
                                      :, taxels, :], (events_array.shape[1], -1))
            selected_chans = 2*len(taxels)
        else:
            events_array = np.reshape(np.transpose(
                events_array, (1, 0, 2)), (events_array.shape[1], -1))
            selected_chans = 2*nchan
        data.append(events_array)
        labels.append(letter_written.index(data_dict['letter'][i]))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)

    # Print average number of spikes for each letter
    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    # create 70/20/10 train/test/validation split
    # first create 70/30 train/(test + validation)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.30, shuffle=True, stratify=labels)
    # split test and validation 2/1
    x_test, x_validation, y_test, y_validation = train_test_split(
        x_test, y_test, test_size=0.33, shuffle=True, stratify=y_test)

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
    ds_validation = TensorDataset(x_validation, y_validation)

    return ds_train, ds_test, ds_validation, labels, selected_chans, data_steps


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


class CuBaLIF:
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

class CuBaRLIF:
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



class SRNN:

    def __init__(self, nb_inputs, nb_hidden, nb_output, dict_args):
        """
        Initialize the SRNN model with parameters and layers.
        All entries from dict_args are set as attributes.
        """

        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output

        for k, v in dict_args.items():
            if hasattr(self, k):
                raise ValueError(f"Cannot override existing attribute {k!r}")
            setattr(self, k, v)

        if self.eprop:
            self.train = self.train_eprop
        else:
            self.train = self.train_bptt

        time_step = self.time_bin_size / 1000.0

        if self.tau_syn == 0.0:
            self.alpha = 0.0
        else:
            self.alpha = float(np.exp(-time_step / self.tau_syn))

        self.device = torch.device(dict_args['device'])
        self.dtype = torch.float

        self.beta = float(np.exp(-time_step / self.tau_mem))
        self.beta_trace = float(np.exp(-time_step / self.tau_trace))
        self.beta_trace_rec = float(np.exp(-time_step / self.tau_trace_rec))

        with open('test_init_weight.pkl', 'rb') as f:
            layers = pickle.load(f)

        self.ff_layer = CuBaLIF(batch_size=self.batch_size, nb_inputs=self.nb_hidden, nb_neurons=self.nb_output,
                            fwd_scale=self.fwd_weight_scale, alpha=self.alpha, firing_threshold=self.firing_threshold,
                            beta=self.beta, device=self.device, dtype=self.dtype, lower_bound=self.lower_bound,
                            ref_per_timesteps=self.ref_per_timesteps,weights=layers[1], requires_grad=True)

        self.rec_layer = CuBaRLIF(batch_size=self.batch_size, nb_inputs=self.nb_inputs, nb_neurons=self.nb_hidden,
                           fwd_scale=self.fwd_weight_scale, rec_scale=self.weight_scale_factor, alpha=self.alpha,
                           firing_threshold=self.firing_threshol, beta=self.beta, device=self.device, dtype=self.dtype,
                           lower_bound=self.lower_bound, ref_per_timesteps=self.ref_per_timesteps, weights=[layers[0], layers[2]],
                           requires_grad=True)


    def forward(self, input, weights):
        nb_steps = self.max_time // self.time_bin_size
        bs = input.shape[0]
        rec_layer_ff_weight, ff_layer_weights, rec_layer_rec_weight = weights

        layers_update = weights
        # Reset from previous batch
        self.rec_layer.syn = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.mem = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.rst = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.ref_per_counter = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.firing_threshold = self.rec_layer.theta * torch.ones((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.out = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)

        self.ff_layer.syn = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.mem = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.rst = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.ref_per_counter = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.firing_threshold = self.ff_layer.theta * torch.ones((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.n_spike = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)

        # add them as rsnn attribute?
        rec_spk_tot = torch.zeros((bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_syn_tot = torch.zeros((bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_syn_tot = torch.zeros((bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_mem_tot = torch.zeros((bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)

        ff_spk_tot = torch.zeros((bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_syn_tot = torch.zeros((bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_mem_tot = torch.zeros((bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_nb_spk_tot = torch.zeros((bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)

        h = torch.einsum("abc,cd->abd", input, rec_layer_ff_weight)

        for t in range(nb_steps):
            rec_spk, rec_syn, rec_mem = self.rec_layer.step(h[:,t,:], rec_layer_rec_weight)
            rec_spk_tot[:,t,:] = rec_spk
            rec_syn_tot[:,t,:] = rec_syn
            rec_mem_tot[:,t,:] = rec_mem

        h1 = torch.einsum("abc,cd->abd", rec_spk_tot, ff_layer_weights)

        for t in range(nb_steps):
            ff_spk, ff_syn, ff_mem, ff_nb_spk = self.ff_layer.step(h1[:,t,:])

            ff_spk_tot[:,t,:] = ff_spk
            ff_syn_tot[:,t,:] = ff_syn
            ff_mem_tot[:,t,:] = ff_mem
            ff_nb_spk_tot[:,t,:] = ff_nb_spk

        return[rec_spk_tot, rec_syn_tot, rec_mem_tot], [ff_spk_tot, ff_syn_tot, ff_mem_tot, ff_nb_spk_tot], layers_update

    def train_bptt(self, dataset_train, dataset_test):
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2)
        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                            total=self.epochs, leave=False)
        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(self.device), y_local.to(self.device)
                recs, ff, layers_update = self.forward(x_local)
                if self.quantization:
                    layers_update = [ste_fn(layer, possible_weight) for layer in layers_update]
                    layers_update = [layer.to(self.dtype) for layer in layers_update]
                spk_rec_readout = ff[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
                spk_rec_hidden = recs[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                #print("m: ", m.sum())
                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes*torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                #print("reg_loss: ", reg_loss)
                reg_loss += self.reg_neurons * \
                    torch.mean(torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                #print(f"{loss_val:.15f}")  # prints 10 decimal places

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
                max_val, am = torch.max(m, 1)     # argmax over output units
                '''
                # This is a workaround to randomly select one of the neurons that have the maximum spikes
                max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

                is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

                rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
                rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

                _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
                '''
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(dataset_test, layers_update)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]*100, "Test acc", accs_hist[1][-1]*100)

        return loss_hist, accs_hist, best_acc_layers



    def train_eprop(self, dataset_train, dataset_test):
        # The log softmax function across output units
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2)

        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        # The optimization loop
        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                            total=self.epochs, leave=False)

        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(self.device), y_local.to(self.device)
                # reset refractory period counter for each batch
                optimizer.zero_grad()
                one_hot_encoded = torch.nn.functional.one_hot(y_local, num_classes=len(np.unique(labels)))

                rec, ff, layers_update = self.forward(x_local, layers)
                if self.quantization:
                    layers_update = [ste_fn(layer, possible_weight) for layer in layers_update]
                    layers_update = [layer.to(self.dtype) for layer in layers_update]
                ff[3][:] = ff[3][:, -1:, :].expand_as(ff[3])
                _, am = torch.max(ff[3], 2)

                '''
                # This is a workaround to randomly select one of the neurons that have the maximum spikes
                max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

                is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

                rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
                rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

                _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
                '''


                yo = torch.nn.functional.one_hot(am, num_classes=len(np.unique(labels))).to(self.device)

                spk_rec_hidden= rec[0]
                spk_rec_readout = ff[0]

                self.grads_batch(x_local.permute(1,0,2), yo.permute(1,0,2), one_hot_encoded, rec[2].permute(1,0,2), rec[0].permute(1,0,2))

                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes * torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                reg_loss += self.reg_neurons * \
                    torch.mean(torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                optimizer.step()
                local_loss.append(loss_val.item())

                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            # mean_accs: mean training accuracy of current epoch (average over all batches)
            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(
                    dataset_test, layers_update)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]*100, "Test acc", accs_hist[1][-1]*100)
        return loss_hist, accs_hist, best_acc_layers

    def grads_batch(self, x, yo, yt, v, z):

        if self.ff_layer.ff_weights.grad is None:
            self.ff_layer.ff_weights.grad = torch.zeros_like(self.ff_layer.ff_weights)
        if self.rec_layer.ff_weights.grad is None:
            self.rec_layer.ff_weights.grad = torch.zeros_like(self.rec_layer.ff_weights)
        if self.rec_layer.rec_weights.grad is None:
            self.rec_layer.rec_weights.grad = torch.zeros_like(self.rec_layer.rec_weights)

        # Surrogate derivatives
        h = self.gamma * torch.max(torch.zeros_like(v), 1 - torch.abs((v - self.firing_threshold) / self.firing_threshold))

        err = torch.zeros_like(yo)

        # Eligibility traces convolution
        beta_conv     = torch.tensor([self.beta_trace_rec ** (data_steps - i - 1) for i in range(data_steps)]).float().view(1, 1, -1).to(self.device)
        beta_rec_conv = torch.tensor([self.beta_trace ** (data_steps - i - 1) for i in range(data_steps)]).float().view(1, 1, -1).to(self.device)

        # Convoluzione Input eligibility traces
        trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(nb_inputs, -1, -1), padding=data_steps, groups=nb_inputs)[:, :, 1:data_steps+1]
        trace_in = trace_in.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
        trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

        trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, :data_steps]
        trace_rec = trace_rec.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
        trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

        # Output eligibility vector
        trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, 1:data_steps+1]

        trace_in = F.conv1d(trace_in.reshape(self.batch_size, nb_inputs * nb_hidden, data_steps),
                            beta_conv.expand(nb_inputs * nb_hidden, -1, -1),
                            padding=data_steps, groups=nb_inputs * nb_hidden)[:, :, 1:data_steps+1]
        trace_in = trace_in.reshape(self.batch_size, nb_hidden, nb_inputs, data_steps)

        trace_rec = F.conv1d(trace_rec.reshape(self.batch_size, nb_hidden * nb_hidden, data_steps),
                            beta_conv.expand(nb_hidden * nb_hidden, -1, -1),
                            padding=data_steps, groups=nb_hidden * nb_hidden)[:, :, 1:data_steps+1]
        trace_rec = trace_rec.reshape(self.batch_size, nb_hidden, nb_hidden, data_steps)

        for i in range(yo.shape[0]):
            err[i,:,:] = yo[i,:,:] - yt
        err = err.to(self.dtype)
        # Learning signal
        L = torch.einsum('tbo,or->brt', err, self.ff_layer.ff_weights.t())

        # Weight gradient updates
        self.rec_layer.ff_weights.grad += (torch.sum(L.unsqueeze(2).expand(-1, -1, nb_inputs, -1) * trace_in, dim=(0, 3))).t()
        self.rec_layer.rec_weights.grad += (torch.sum(L.unsqueeze(2).expand(-1, -1, nb_hidden, -1) * trace_rec, dim=(0, 3))).t()
        self.ff_layer.ff_weights.grad += (torch.einsum('tbo,brt->or', err, trace_out)).t()

    def compute_classification_accuracy(self, dataset, weights):
        """ Computes classification accuracy on supplied data in batches. """

        generator = DataLoader(dataset=dataset, batch_size=self.batch_size_test, pin_memory=True,
                            shuffle=False, num_workers=2)
        accs = []

        for x_local, y_local in generator:
            x_local, y_local = x_local.to(self.device), y_local.to(self.device)
            with torch.no_grad():
                rec, ff, _ = self.forward(x_local, weights)

            spk_rec_readout = ff[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
            m = torch.sum(spk_rec_readout, 1)  # sum over time

            max_val, am = torch.max(m, 1)     # argmax over output units
            '''
            # This is a workaround to randomly select one of the neurons that have the maximum spikes
            max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

            is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

            rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
            rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

            _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
            '''

            # compare to labels
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        return np.mean(accs)


# Load data and parameters
file_dir_data = './data/'
file_type = 'data_braille_letters_100Hz_th'
file_thr = str(threshold)
file_name = file_dir_data + file_type + file_thr + '.pkl'

if __name__ == '__main__':

    ds_train, ds_test, ds_validation, labels, nb_channels, data_steps = load_and_extract(
        dict_args, file_name, letter_written=letters)

    nb_inputs = nb_channels
    nb_hidden = 450
    nb_outputs = len(np.unique(labels))

    a = SRNN(nb_inputs=nb_inputs,nb_hidden=nb_hidden,nb_output=nb_outputs,dict_args=dict_args)
    b, c, best_layers = a.train(ds_train, ds_test)

    torch.save(best_layers, './model/best_model_th'+str(threshold)+'_quant.pt')


