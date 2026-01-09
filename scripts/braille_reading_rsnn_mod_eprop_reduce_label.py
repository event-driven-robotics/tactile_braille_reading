"""
Here we train a minimal model which can discriminate between three braille letters using a recurrent network.
The code will find the minimum size of the recurrent layer we need to reach good performance.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec  # can be used for nice subplot layout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# or make permanent by adding to bashrc "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"


torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

dtype = torch.float

letters = ['A', 'B']  # difficult: a vs. b, c, e, i; easy: a vs. p, q, y
letters = ['J', 'U']  # difficult: a vs. b, c, e, i; easy: a vs. p, q, y
create_validation = False

# set variables
use_seed = True
# set the number of epochs you want to train the network (default = 300)
epochs = 20  # TODO bring back to 50

# TODO double check those params


class experimantal_params:
    def __init__(self):
        # self.epochs = 50  # TODO remove the  one above for consistency and use this instead
        self.threshold = 2  # possible values are: 1, 2, 5, 10
        self.time_bin_size = 1  # ms
        self.nb_input_copies = 1
        self.batch_size = 128  # default: 128
        self.learning_rate = 0.004
        self.gamma = 0.3  # used for the surrogate gradient
        self.lower_bound = -1.0
        self.no_synapse = True
        self.use_linear_decay = False
        self.ref_per_timesteps = 3
        self.time_bin_size = 1
        self.tau_mem = 0.05
        self.tau_ratio = 10
        self.fwd_weight_scale = 10
        self.weight_scale_factor = 0.2
        self.reg_spikes = 0.0015
        self.reg_neurons = 0.0
        self.tau_mem = 0.06  # params['tau_mem']  # ms
        # global tau_mem_rec
        self.tau_mem_rec = 0.06  # params['tau_mem'] #ms
        # global tau_trace
        self.tau_trace = 0.14
        self.tau_trace_out = 0.14
        self.use_eprop = False  # if False, use BPTT (backpropagation through time)
        self.use_mechanoreceptor_encoding = True  # if True, use mechanoreceptor encoding, if False sigma-delta encoding
        if self.use_mechanoreceptor_encoding:
            self.max_time = 3700
        else:
            self.max_time = 3501


params = experimantal_params().__dict__

# global batch_size
# batch_size = 128
# global lr
# lr = 0.004  # 0.0008
# print("Learning rate: ", lr)
# global gamma
# gamma = 0.3

# global lower_bound
# lower_bound = -1.0  # set to None to disable
# global no_synapse
# no_synapse = True
# global use_linear_decay
# use_linear_decay = False
# global ref_per_timesteps
# # refractory period is set in simulation time steps for now; set to None to disable
# ref_per_timesteps = 3

# some options for plotting
NB_BATCHES_TO_PLOT = 1
NB_TRIALS_TO_PLOT = 1

# create folder to safe figures later
path = './figures'
isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

device = torch.device("cuda:0")

neg_capacitance = torch.arange(255, -1, -1)

pos_capacitance = torch.arange(1, 257)
diff_cap = pos_capacitance - neg_capacitance
diff_cap = diff_cap.to(device)
q = 1/256

possible_weight = diff_cap * q


factor = 10 ** 3
possible_weight = torch.floor(possible_weight * factor) / factor
# print(possible_weight)


# use fixed seed for reproducable results
if use_seed:
    seed = 42  # "Answer to the Ultimate Question of Life, the Universe, and Everything"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")


def load_and_extract(params: dict, file_name: str, taxels=None, letter_written=letters, create_validation=False) -> tuple:
    """
    Load and preprocess braille letter tactile sensor data from a pickle file.

    This function loads event-based tactile sensor data, bins it into time windows,
    optionally filters by specific taxels (tactile sensors) and letters, and creates
    train/test (and optionally validation) datasets.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters:
        - 'max_time' : int
            Maximum time duration in milliseconds
        - 'time_bin_size' : int
            Size of time bins in milliseconds
    file_name : str
        Path to the pickle file containing braille letter data
    taxels : list or None, optional
        List of specific taxel indices to use. If None, all taxels are used (default: None)
    letter_written : list, optional
        List of letter labels to include in the dataset (default: letters global variable)
    create_validation : bool, optional
        If True, creates a 70/20/10 train/test/validation split.
        If False, creates an 80/20 train/test split (default: False)

    Returns
    -------
    tuple
        If create_validation is True:
            (ds_train, ds_test, ds_validation, labels, selected_chans, data_steps)
        If create_validation is False:
            (ds_train, ds_test, labels, selected_chans, data_steps)

        Where:
        - ds_train, ds_test, ds_validation : TensorDataset
            PyTorch datasets containing (data, labels) pairs
        - labels : torch.Tensor
            All label values in the dataset
        - selected_chans : int
            Number of selected channels (2 * number of taxels for ON/OFF events)
        - data_steps : int
            Number of time steps in the data

    Notes
    -----
    - Events are binned into time windows of size time_bin_size
    - Each taxel has two channels (ON and OFF events)
    - Data is converted to torch tensors with appropriate dtypes
    - Splits are stratified to maintain class balance
    """

    max_time = params["max_time"]  # ms
    # int(params['time_bin_size'])  # so far from laoded file, but can be set manually here
    time_bin_size = params["time_bin_size"]
    time = range(0, max_time, time_bin_size)

    params["time_step"] = time_bin_size*0.001  # ms
    data_steps = len(time)
    # params["delayed_output"] = data_steps
    params["delayed_output"] = 0  # data_steps

    data_dict = pd.read_pickle(file_name)
    # data_dict_2 = pd.read_pickle('./data/data_braille_letters_th_2.pkl')
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
        if data_dict['letter'][i] in letter_written:
            data.append(events_array)
            labels.append(letter_written.index(data_dict['letter'][i]))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)
    # print(labels)
    data = torch.tensor(data, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    if create_validation:
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

    else:
        # create 80/20 train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.20, shuffle=True, stratify=labels)

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)

        return ds_train, ds_test, labels, selected_chans, data_steps


def grads_batch(x: torch.Tensor, yo: torch.Tensor, yt: torch.Tensor, gamma: float, thr: int, v: torch.Tensor, z: torch.Tensor, w_in: torch.Tensor, w_rec: torch.Tensor, w_out: torch.Tensor, beta_trace: float, beta_trace_out: float) -> None:
    """
    Compute weight gradients using e-prop (eligibility propagation) for spiking neural networks.

    This function implements the e-prop learning algorithm for recurrent spiking neural networks.
    It computes eligibility traces for input, recurrent, and output connections, then uses these
    traces along with the error signal to calculate weight gradients. Gradients are accumulated
    directly into the .grad attributes of the weight tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input spike trains with shape [time, batch, input_features]
    yo : torch.Tensor
        Network output (predicted labels) with shape [time, batch, output_units]
    yt : torch.Tensor
        Target labels (one-hot encoded) with shape [batch, output_units]
    gamma : float
        Surrogate gradient scaling factor for the spike function derivative approximation
    thr : int
        Firing threshold for neurons (typically 1)
    v : torch.Tensor
        Membrane potential traces of hidden neurons with shape [time, batch, hidden_neurons]
    z : torch.Tensor
        Spike trains of hidden neurons with shape [time, batch, hidden_neurons]
    w_in : torch.Tensor
        Input-to-hidden weight matrix with shape [hidden_neurons, input_features]
        Gradients are accumulated in w_in.grad
    w_rec : torch.Tensor
        Recurrent (hidden-to-hidden) weight matrix with shape [hidden_neurons, hidden_neurons]
        Gradients are accumulated in w_rec.grad
    w_out : torch.Tensor
        Hidden-to-output weight matrix with shape [output_units, hidden_neurons]
        Gradients are accumulated in w_out.grad
    beta_trace : float
        Decay factor for eligibility traces of hidden layer (exp(-dt/tau_trace))
    beta_trace_out : float
        Decay factor for output eligibility traces (exp(-dt/tau_trace_out))

    Returns
    -------
    None
        This function modifies weight gradients in-place (w_in.grad, w_rec.grad, w_out.grad)

    Notes
    -----
    - Implements the e-prop algorithm for online learning in spiking neural networks
    - Uses surrogate gradients to approximate the non-differentiable spike function
    - Eligibility traces track the causal relationship between weight changes and neuron spikes
    - Convolutions are used to efficiently compute eligibility traces over time
    - The delayed_output parameter from params dict controls which time steps contribute to gradients
    - All computations are performed on the device specified by the global 'device' variable

    References
    ----------
    Bellec et al. (2020). "A solution to the learning dilemma for recurrent networks 
    of spiking neurons." Nature Communications.
    """
    nb_inputs = x.shape[-1]

    if w_in.grad is None:
        w_in.grad = torch.zeros_like(w_in)
    if w_rec.grad is None:
        w_rec.grad = torch.zeros_like(w_rec)
    if w_out.grad is None:
        w_out.grad = torch.zeros_like(w_out)
    # Surrogate derivatives
    h = gamma * torch.max(torch.zeros_like(v), 1 - torch.abs((v - thr) / thr))

    # Crea una variabile di errore vuota con le stesse dimensioni di yo
    err = torch.zeros_like(yo)

    # Eligibility traces convolution
    beta_conv = torch.tensor([beta_trace_out ** (data_steps - i - 1)
                             for i in range(data_steps)]).float().view(1, 1, -1).to(device)
    beta_rec_conv = torch.tensor([beta_trace ** (data_steps - i - 1)
                                 for i in range(data_steps)]).float().view(1, 1, -1).to(device)

    # Convoluzione Input eligibility traces
    trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(
        nb_inputs, -1, -1), padding=data_steps, groups=nb_inputs)[:, :, 1:data_steps+1]
    trace_in = trace_in.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

    # Convoluzione Recurrent eligibility traces
    trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(
        nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, :data_steps]
    trace_rec = trace_rec.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
    trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)
    
    # Free h as it's no longer needed
    del h

    # Output eligibility vector
    trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(
        nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, 1:data_steps+1]

    # Ottimizzazione convoluzioni batch-wise
    trace_in = F.conv1d(trace_in.reshape(x.shape[1], nb_inputs * nb_hidden, data_steps),
                        beta_conv.expand(nb_inputs * nb_hidden, -1, -1),
                        padding=data_steps, groups=nb_inputs * nb_hidden)[:, :, 1:data_steps+1]
    trace_in = trace_in.reshape(
        x.shape[1], nb_hidden, nb_inputs, data_steps)

    trace_rec = F.conv1d(trace_rec.reshape(x.shape[1], nb_hidden * nb_hidden, data_steps),
                         beta_conv.expand(nb_hidden * nb_hidden, -1, -1),
                         padding=data_steps, groups=nb_hidden * nb_hidden)[:, :, 1:data_steps+1]
    trace_rec = trace_rec.reshape(
        x.shape[1], nb_hidden, nb_hidden, data_steps)

    for i in range(yo.shape[0]):
        err[i, :, :] = yo[i, :, :] - yt
    err = err.to(dtype)

    L = torch.einsum('tbo,or->brt', err, w_out)

    if params["delayed_output"] != 0:
        L = L[:, :, -params["delayed_output"]:]
        err = err[-params["delayed_output"]:, :, :]
        trace_in = trace_in[:, :, :, -params["delayed_output"]:]
        trace_rec = trace_rec[:, :, :, -params["delayed_output"]:]
        trace_out = trace_out[:, :, -params["delayed_output"]:]

    # Weight gradient updates
    w_in.grad += torch.sum(L.unsqueeze(2).expand(-1, -1,
                           nb_inputs, -1) * trace_in, dim=(0, 3))
    
    # Free trace_in immediately after use
    del trace_in
    
    w_rec.grad += torch.sum(L.unsqueeze(2).expand(-1, -1,
                            nb_hidden, -1) * trace_rec, dim=(0, 3))
    
    # Free trace_rec immediately after use
    del trace_rec
    w_out.grad += torch.einsum('tbo,brt->or', err, trace_out)
    
    # Free remaining large tensors
    del trace_out, L, err


def run_snn(inputs: torch.Tensor, layers: list) -> tuple:
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

    Returns
    -------
    tuple
        (spk_rec_readout, other_recs) where:
        - spk_rec_readout : torch.Tensor
            Output layer spike recordings [batch, time, output_neurons]
        - other_recs : list
            [mem_rec_hidden, spk_rec_hidden, mem_rec_readout] recordings from layers

    Notes
    -----
    - This function only performs the forward pass
    - Gradient computation (e-prop or BPTT) is handled in the train() function
    - For e-prop mode: uses hard threshold spike generation (no gradient through spikes)
    - For BPTT mode: uses surrogate gradient function for differentiable spike generation
    - Supports input replication via params["nb_input_copies"] for increased input representation
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
        h1, data_steps, params["lower_bound"])

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec_hidden, ff_layer.ff_weights.t()))

    spk_rec_readout, mem_rec_readout = ff_layer.compute_activity(
        h2, data_steps, params["lower_bound"])

    other_recs = [mem_rec_hidden, spk_rec_hidden, mem_rec_readout]

    return spk_rec_readout, other_recs


def build_and_train(params: dict, ds_train: TensorDataset, ds_test: TensorDataset, nb_hidden=20, epochs=epochs) -> tuple:
    """
    Build and train a recurrent spiking neural network for braille letter classification.

    This function constructs a two-layer spiking neural network (recurrent hidden layer and
    feedforward readout layer), trains it using e-prop, and reports the best performance metrics.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters including:
        - 'nb_input_copies' : int
            Number of copies for each input channel
        - 'tau_mem', 'tau_mem_rec', 'tau_trace', 'tau_trace_out' : float
            Time constants for membrane potentials and eligibility traces
        - 'batch_size', 'learning_rate' : int, float
            Training hyperparameters
        - 'no_synapse', 'use_linear_decay', 'ref_per_timesteps' : bool/int
            Neuron model configuration
        - 'fwd_weight_scale', 'weight_scale_factor' : float
            Weight initialization scales
    ds_train : TensorDataset
        Training dataset containing (input_data, labels) pairs
    ds_test : TensorDataset
        Test dataset containing (input_data, labels) pairs
    nb_hidden : int, optional
        Number of neurons in the recurrent hidden layer (default: 20)
    epochs : int, optional
        Number of training epochs (default: global epochs variable)

    Returns
    -------
    tuple
        (loss_hist, accs_hist, best_layers, vars_eprop) where:
        - loss_hist : list
            Training loss values per epoch
        - accs_hist : list of lists
            [[train_accuracies], [test_accuracies]] per epoch
        - best_layers : list
            Weight matrices of the best performing model
        - vars_eprop : list
            [beta_trace, beta_trace_out] eligibility trace decay factors

    Notes
    -----
    - Computes and prints best training/test accuracies and their corresponding epochs
    - Uses exponential or linear decay for membrane potential dynamics
    - Supports optional synaptic dynamics and refractory periods
    - Saves the model with best training accuracy during training
    """

    # Num of spiking neurons used to encode each channel
    nb_input_copies = params['nb_input_copies']
    # print("Number of input copies ", nb_input_copies)
    # Network parameters
    nb_inputs = nb_channels*nb_input_copies
    nb_outputs = len(np.unique(labels))
    nb_hidden = nb_hidden
    # print("Number of hidden neurons ", nb_hidden)

    # tau_mem = 0.06  # params['tau_mem']  # ms
    # # global tau_mem_rec
    # tau_mem_rec = 0.06  # params['tau_mem'] #ms
    # # global tau_trace
    # tau_trace = 0.14
    # tau_trace_out = 0.14
    tau_syn = params['tau_mem']/params['tau_ratio']
    # print("tau_mem: ", params['tau_mem'], "tau_mem recurrent: ", params['tau_mem_rec'],
    #       "tau trace out: ", params['tau_trace_out'], "tau trace: ", params['tau_trace'])
    if params["no_synapse"]:
        alpha = 0.0  # here we disable synapse dynamics
    else:
        alpha = float(np.exp(-params["time_step"]/tau_syn))

    if params["use_linear_decay"]:
        beta = 0.005  # 0.05 < 0.01 says how much to lose
    else:
        # says how much to keep
        beta = float(np.exp(-params["time_step"]/params['tau_mem']))
        beta_rec = float(np.exp(-params["time_step"]/params['tau_mem_rec'])
                         )  # says how much to keep
        beta_trace = float(np.exp(-params["time_step"]/params['tau_trace']))
        beta_trace_out = float(
            np.exp(-params["time_step"]/params['tau_trace_out']))

    vars_eprop = [beta_trace, beta_trace_out]
    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = fwd_weight_scale*params['weight_scale_factor']

    # Spiking network
    # recurrent layer
    rec_layer = recurrent_layer(
        batch_size=params["batch_size"], nb_inputs=nb_inputs, nb_neurons=nb_hidden, fwd_scale=fwd_weight_scale, rec_scale=rec_weight_scale, alpha=alpha, beta=beta_rec, ref_per=params["ref_per_timesteps"])

    # readout layer
    ff_layer = feedforward_layer(
        batch_size=params["batch_size"], nb_inputs=nb_hidden, nb_neurons=nb_outputs, fwd_weight_scale=fwd_weight_scale, alpha=alpha, beta=beta, ref_per=params["ref_per_timesteps"])

    layers = [rec_layer, ff_layer]
    
    # # Debug: Check initial weight statistics
    # print(f"\nInitial weight statistics for {nb_hidden} hidden neurons:")
    # print(f"  Recurrent layer input weights: mean={rec_layer.ff_weights.data.mean():.4f}, std={rec_layer.ff_weights.data.std():.4f}")
    # print(f"  Recurrent layer recurrent weights: mean={rec_layer.rec_weights.data.mean():.4f}, std={rec_layer.rec_weights.data.std():.4f}")
    # print(f"  Output layer weights shape: {ff_layer.ff_weights.shape}")
    # print(f"  Output layer weights per neuron:")
    # for neuron_idx in range(ff_layer.ff_weights.shape[0]):
    #     w = ff_layer.ff_weights[neuron_idx]
    #     print(f"    Neuron {neuron_idx}: mean={w.data.mean():.4f}, std={w.data.std():.4f}, min={w.data.min():.4f}, max={w.data.max():.4f}")
    # print()

    # a fixed learning rate is already defined within the train function, that's why here it is omitted
    loss_hist, accs_hist, best_layers = train(
        params=params, dataset=ds_train, layers=layers, vars_eprop=vars_eprop, lr=params["learning_rate"], nb_epochs=epochs, dataset_test=ds_test)

    # best training and test at best training
    acc_best_train = np.max(accs_hist[0])  # returns max value
    acc_best_train = acc_best_train*100

    idx_best_train = np.argmax(accs_hist[0])  # returns index of max value
    acc_test_at_best_train = accs_hist[1][idx_best_train]*100

    # best test and training at best test
    acc_best_test = np.max(accs_hist[1])
    acc_best_test = acc_best_test*100
    idx_best_test = np.argmax(accs_hist[1])
    acc_train_at_best_test = accs_hist[0][idx_best_test]*100

    # TODO track time constants!!!
    # print("Final results: ")
    # print("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(
    #     acc_best_train, acc_test_at_best_train, idx_best_train+1))
    # print("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
    #     acc_best_test, acc_train_at_best_test, idx_best_test+1))
    # print("------------------------------------------------------------------------------------\n")

    return loss_hist, accs_hist, best_layers, vars_eprop


def copy_layers(layers: list) -> list:
    """
    Create deep copies of layer objects by recreating them with copied weights.

    This function manually copies layer objects to avoid deepcopy issues with
    PyTorch tensors that have gradients attached.

    Parameters
    ----------
    layers : list
        List of [recurrent_layer, feedforward_layer] objects to copy

    Returns
    -------
    list
        New list of copied layer objects with detached and cloned weights
    """
    rec_layer, ff_layer = layers

    # Create new recurrent layer instance
    new_rec_layer = recurrent_layer(
        batch_size=rec_layer.batch_size,
        nb_inputs=rec_layer.nb_inputs,
        nb_neurons=rec_layer.nb_neurons,
        fwd_scale=rec_layer.fwd_scale,
        rec_scale=rec_layer.rec_scale,
        alpha=rec_layer.alpha,
        beta=rec_layer.beta,
        ref_per=rec_layer.ref_per
    )
    # Copy weights (detached and cloned to break gradient connection)
    new_rec_layer.ff_weights.data = rec_layer.ff_weights.data.detach().clone()
    new_rec_layer.rec_weights.data = rec_layer.rec_weights.data.detach().clone()

    # Create new feedforward layer instance
    new_ff_layer = feedforward_layer(
        batch_size=ff_layer.batch_size,
        nb_inputs=ff_layer.nb_inputs,
        nb_neurons=ff_layer.nb_neurons,
        fwd_weight_scale=ff_layer.scale,
        alpha=ff_layer.alpha,
        beta=ff_layer.beta,
        ref_per=ff_layer.ref_per
    )
    # Copy weights (detached and cloned)
    new_ff_layer.ff_weights.data = ff_layer.ff_weights.data.detach().clone()

    return [new_rec_layer, new_ff_layer]


def train(params: dict, dataset: TensorDataset, layers: list, vars_eprop: list, lr=0.0015, nb_epochs=300, dataset_test=None) -> tuple:
    """
    Train a spiking neural network using e-prop and evaluate on test data.

    This function implements the training loop for a spiking neural network using the e-prop
    algorithm. It processes data in batches, computes gradients, updates weights, and tracks
    training/test performance over epochs.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters including:
        - 'batch_size' : int
            Number of samples per batch
        - 'reg_spikes', 'reg_neurons' : float
            L1 and L2 regularization coefficients for spike activity
        - 'delayed_output' : int
            Number of timesteps to use for output computation
    dataset : TensorDataset
        Training dataset containing (input_data, labels) pairs
    layers : list
        List containing [recurrent_layer, feedforward_layer] layer objects
    vars_eprop : list
        List containing [beta_trace, beta_trace_out] decay factors for eligibility traces
    lr : float, optional
        Learning rate for Adamax optimizer (default: 0.0015)
    nb_epochs : int, optional
        Number of training epochs (default: 300)
    dataset_test : TensorDataset, optional
        Test dataset for evaluation after each epoch (default: None)

    Returns
    -------
    tuple
        (loss_hist, accs_hist, best_acc_layers) where:
        - loss_hist : list
            Training loss values per epoch
        - accs_hist : list of lists
            [[train_accuracies], [test_accuracies]] per epoch
        - best_acc_layers : list
            Weight matrices of the best performing model

    Notes
    -----
    - Uses Adamax optimizer with betas=(0.9, 0.995)
    - Applies weight quantization via Straight-Through Estimator
    - Implements NLL loss with optional spike regularization (currently disabled)
    - Handles ties in output predictions via random selection
    - Tracks average spike counts for hidden and output layers
    - Saves model with best training accuracy
    - Displays progress via tqdm progress bars
    """

    weights = [layers[0].ff_weights,
               layers[0].rec_weights, layers[1].ff_weights]
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=True, num_workers=4)

    # The optimization loop
    loss_hist = []
    accs_hist = [[], []]
    best_test_acc = 0.0
    best_acc_layers = []

    pbar_training = tqdm(range(nb_epochs), position=1,
                         total=nb_epochs, leave=False)
    average_spike_recurrent = torch.zeros(nb_epochs, device=device)
    average_spike_output = torch.zeros(nb_epochs, device=device)
    count_epoch = 0
    for _ in pbar_training:
        # learning rate decreases over epochs
        optimizer = torch.optim.Adamax(weights, lr=lr, betas=(0.9, 0.995))
        # if e > nb_epochs/2:
        #     lr = lr * 0.9
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        pbar_batches = tqdm(generator, position=2,
                            total=len(generator), leave=False)
        for x_local, y_local in pbar_batches:
            x_local, y_local = x_local.to(device), y_local.to(device)

            optimizer.zero_grad()

            spk_rec_readout, recs = run_snn(inputs=x_local, layers=layers)
            # weight quantization - apply directly to layer weights
            layers[0].ff_weights.data = ste_fn(
                layers[0].ff_weights, possible_weight).to(dtype)
            layers[0].rec_weights.data = ste_fn(
                layers[0].rec_weights, possible_weight).to(dtype)
            layers[1].ff_weights.data = ste_fn(
                layers[1].ff_weights, possible_weight).to(dtype)

            average_spike_output[count_epoch] = torch.mean(
                torch.sum(spk_rec_readout, 1))
            average_spike_recurrent[count_epoch] = torch.mean(
                torch.sum(recs[1], 1))

            _, spk_rec_hidden, _ = recs

            # Use only the delayed_output window for spike counting (consistent with test evaluation)
            m = torch.sum(spk_rec_readout[:, -params["delayed_output"]:, :], dim=1)

            # cross entropy loss on the active read-out layer
            log_p_y = log_softmax_fn(m)

            # Here we can set up our regularizer loss
            # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
            # L1 loss on total number of spikes (hidden layer 1)
            reg_loss = params['reg_spikes'] * \
                torch.mean(torch.sum(spk_rec_hidden, 1))
            # L1 loss on total number of spikes (output layer)
            # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
            # print("L1: ", reg_loss)
            # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
            # L2 loss on spikes per neuron (hidden layer 1)
            reg_loss += params['reg_neurons'] * \
                torch.mean(
                    torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
            # L2 loss on spikes per neuron (output layer)
            # reg_loss += params['reg_neurons'] * \
            #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
            # print("L1 + L2: ", reg_loss)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local)  # + reg_loss
            
            # Select winner based on spike counts (used for both e-prop and accuracy)
            max_val, am = torch.max(m, 1)     # argmax over output units
            # Handle ties: if multiple neurons have the same max spike count, select randomly
            mask = torch.sum(m == max_val.unsqueeze(-1), dim=-1) > 1
            if mask.any():
                true_indices = torch.nonzero(mask, as_tuple=True)
                for i in true_indices:
                    candidates = torch.nonzero(
                        m[i] == max_val[i].unsqueeze(-1), as_tuple=True)[0]
                    am[i] = candidates[torch.randint(0, len(candidates), (1,))]
            
            # Compute gradients based on selected learning algorithm
            if params["use_eprop"]:
                one_hot_encoded = torch.nn.functional.one_hot(
                    y_local, num_classes=len(np.unique(labels)))
                # E-prop: manual gradient computation via eligibility traces
                # Create one-hot encoded predictions from selected winners
                yo = torch.nn.functional.one_hot(
                    am, num_classes=len(np.unique(labels)))
                # Expand to match temporal dimension for e-prop
                yo = yo.unsqueeze(1).expand(-1, spk_rec_readout.shape[1], -1)
                
                # Compute e-prop gradients
                mem_rec_hidden, spk_rec_hidden, _ = recs
                grads_batch(x_local.permute(1, 0, 2), yo.permute(1, 0, 2), one_hot_encoded, 
                           params["gamma"], 1, mem_rec_hidden.permute(1, 0, 2), 
                           spk_rec_hidden.permute(1, 0, 2), layers[0].ff_weights, 
                           layers[0].rec_weights, layers[1].ff_weights, 
                           vars_eprop[0], vars_eprop[1])
            else:
                # BPTT: standard backpropagation
                loss_val.backward()
            
            optimizer.step()

            local_loss.append(loss_val.item())

            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)
            
            # Debug: Print first batch of first epoch to check predictions
            # if count_epoch == 0 and len(accs) == 1:
            #     print(f"\nDebug - First batch:")
            #     print(f"  True labels (y_local): {y_local[:10].cpu().numpy()}")
            #     print(f"  Predictions (am): {am[:10].cpu().numpy()}")
            #     print(f"  Spike counts (m): {m[:10].detach().cpu().numpy()}")
            #     print(f"  Label distribution: 0={torch.sum(y_local==0).item()}, 1={torch.sum(y_local==1).item()}")
            #     print(f"  Prediction distribution: 0={torch.sum(am==0).item()}, 1={torch.sum(am==1).item()}")
            #     print(f"  Batch accuracy: {tmp:.4f}")
                
            #     # Check input data
            #     print(f"\n  Input data statistics:")
            #     print(f"    Shape: {x_local.shape}")
            #     print(f"    Total input spikes (first 10 samples): {torch.sum(x_local[:10], dim=(0,1)).cpu().numpy()}")
            #     print(f"    Input spike rate (mean): {x_local.mean().item():.6f}")
                
            #     # Check hidden layer
            #     print(f"\n  Hidden layer statistics:")
            #     hidden_spike_counts = torch.sum(spk_rec_hidden, dim=0)  # sum over time: [batch, neurons]
            #     print(f"    Spike count distribution (mean across batch): {hidden_spike_counts.mean(dim=0).cpu().numpy()}")
            #     print(f"    Spike count (min, max): ({hidden_spike_counts.min().item()}, {hidden_spike_counts.max().item()})")
            #     print(f"    First 10 samples total spikes: {torch.sum(spk_rec_hidden[:10], dim=(0,1)).cpu().numpy()}")
                
            #     # Check output layer
            #     print(f"\n  Output layer statistics:")
            #     print(f"    Weights shape: {layers[1].ff_weights.shape}")
            #     print(f"    Weights mean per neuron: {torch.mean(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
            #     print(f"    Weights std per neuron: {torch.std(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
            #     h2_sample = torch.einsum("abc,cd->abd", (spk_rec_hidden[:10], layers[1].ff_weights.t()))
            #     print(f"    Input current (h2) mean per output neuron: {h2_sample.mean(dim=(0,1)).detach().cpu().numpy()}")
            #     print(f"    Number of positive weights: neuron0={torch.sum(layers[1].ff_weights[0] > 0).item()}, neuron1={torch.sum(layers[1].ff_weights[1] > 0).item()}")
            #     print()

            # Free up memory by deleting large intermediate tensors
            del spk_rec_readout, recs, spk_rec_hidden, m, log_p_y, reg_loss, max_val, am, mask

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)
        '''
        with open("average_spike_eprop_new_data.txt", "ab") as f:
            np.savetxt(f, np.array([average_spike_recurrent[count_epoch].detach().cpu().numpy(), average_spike_output[count_epoch].detach().cpu().numpy()]).reshape((1,2)))
        '''
        count_epoch = count_epoch + 1

        # Calculate test accuracy in each epoch
        test_acc = compute_classification_accuracy(dataset=dataset_test, layers=layers)
        accs_hist[1].append(test_acc)
        if np.max(test_acc) >= best_test_acc:
            best_test_acc = np.max(test_acc)
            # Save copies of the layer objects using our custom copy function
            best_acc_layers = copy_layers(layers)

        pbar_training.set_description("{:.2f}%, {:.2f}%".format(
            accs_hist[0][-1]*100, accs_hist[1][-1]*100))
        # print("Train acc: ", accs_hist[0][-1]*100, "Test acc",
        #       accs_hist[1][-1]*100, 'Loss: ', loss_hist[-1])

    return loss_hist, accs_hist, best_acc_layers


def compute_classification_accuracy(dataset: TensorDataset, layers: list) -> float:
    """
    Compute classification accuracy on a dataset using a trained spiking neural network.

    This function evaluates network performance by running inference on all samples in the dataset
    (processed in batches) and comparing predicted labels to ground truth labels.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset containing (input_data, labels) pairs for evaluation
    layers : list
        List containing [recurrent_layer, feedforward_layer] trained layer objects
    vars_eprop : list
        List containing [beta_trace, beta_trace_out] decay factors (used for consistency)

    Returns
    -------
    float
        Mean classification accuracy across all samples (range: 0.0 to 1.0)

    Notes
    -----
    - Predictions are made by counting output spikes during the delayed_output window
    - Winner selection uses random tie-breaking when multiple neurons have equal spike counts
    - All computations are performed with torch.no_grad() for efficiency
    - Uses the global params['delayed_output'] to determine which timesteps to consider
    """

    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=False, num_workers=4)
    accs = []

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, _ = run_snn(inputs=x_local, layers=layers)

        # with output spikes
        # sum over time
        m = torch.sum(spk_rec_readout[:, -params["delayed_output"]:, :], dim=1)
        max_val, am = torch.max(m, 1)     # argmax over output units

        # with output spikes
        mask = torch.sum(m == max_val.unsqueeze(-1), dim=-1) > 1
        # If multiple neurons emit the highest number of spikes, select one of them randomly
        if mask.any():
            # print("Multiple maxima detected. It happened: ", mask.sum().item(), " times.")
            # compare to labels
            true_indices = torch.nonzero(mask, as_tuple=True)
            # am[true_indices] = torch.randint(0, len(letters), (len(true_indices),), device=device)
            for i in true_indices:
                candidates = torch.nonzero(
                    m[i] == max_val[i].unsqueeze(-1), as_tuple=True)[0]
                am[i] = candidates[torch.randint(0, len(candidates), (1,))]

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
    
    mean_acc = np.mean(accs)
    # Optionally print for debugging (can comment out after verification)
    # print(f"Test mean accuracy: {mean_acc:.4f}")
    return mean_acc


def plot_training_perfromance(path: str, acc_train: np.ndarray, acc_test: np.ndarray, loss_train: np.ndarray) -> None:
    """
    Visualize training performance with accuracy and loss plots across one or multiple runs.

    Creates a two-panel figure showing training/test accuracies and training loss over epochs.
    For single runs, displays only that run. For multiple runs, displays mean ± standard deviation
    and highlights the best trial.

    Parameters
    ----------
    path : str
        File path (without extension) where the PDF figure will be saved
    acc_train : np.ndarray
        Training accuracies with shape [n_runs, n_epochs] or [1, n_epochs] for single run
    acc_test : np.ndarray
        Test accuracies with shape [n_runs, n_epochs] or [1, n_epochs] for single run
    loss_train : np.ndarray
        Training loss values with shape [n_runs, n_epochs] or [1, n_epochs] for single run

    Returns
    -------
    None
        Saves figure to {path}.pdf and closes the figure

    Notes
    -----
    - Top panel: Training and test accuracy (%) with shaded standard deviation regions
    - Bottom panel: Training loss with shaded standard deviation region
    - Best trial is highlighted with solid lines (trial with maximum test accuracy)
    - Mean across trials shown with dashed lines
    - Figure size: 8x12 inches, saved as PDF
    """
    # calc mean and std
    acc_mean_train, acc_std_train = np.mean(
        acc_train, axis=0), np.std(acc_train, axis=0)
    acc_mean_test, acc_std_test = np.mean(
        acc_test, axis=0), np.std(acc_test, axis=0)
    best_trial, best_val_idx = np.where(np.max(acc_test) == acc_test)
    best_trial, best_val_idx = best_trial[0], best_val_idx[0]
    loss_train_mean, loss_train_std = np.mean(
        loss_train, axis=0), np.std(loss_train, axis=0)

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(2, 1, 1)
    ax.fill_between(range(1, len(acc_mean_train)+1), 100*(acc_mean_train+acc_std_train), 100*(
        acc_mean_train-acc_std_train), color='cornflowerblue')
    ax.fill_between(range(1, len(acc_mean_test)+1), 100*(
        acc_mean_test+acc_std_test), 100*(acc_mean_test-acc_std_test), color='sandybrown')
    # plot mean and std
    ax.plot(range(1, len(acc_mean_train)+1),
            100*np.array(acc_mean_train), color='blue', linestyle='dashed')
    ax.plot(range(1, len(acc_mean_test)+1), 100 *
            np.array(acc_mean_test), color='orangered', linestyle='dashed')
    # highlight best trial
    ax.plot(range(1, len(acc_train[best_trial])+1), 100*np.array(
        acc_train[best_trial]), color='blue')
    ax.plot(range(1, len(acc_test[best_trial])+1), 100*np.array(
        acc_test[best_trial]), color='orangered')
    # ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim((0, 105))
    ax.set_title("Accuracy")
    ax.legend(["Training std", "Test std", r"$\overline{\mathrm{Training}}$",
              r"$\overline{\mathrm{Test}}$", "Training @ best test", "Best test"], loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    ax.fill_between(range(1, len(loss_train_mean)+1), loss_train_mean +
                    loss_train_std, loss_train_mean-loss_train_std, color='cornflowerblue')
    ax.plot(range(1, len(loss_train_mean)+1),
            loss_train_mean, color='blue', linestyle='dashed')
    ax.plot(range(1, len(loss_train[best_trial])+1),
            loss_train[best_trial], color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim((0, None))
    ax.legend(
        ["Training std", r"$\overline{\mathrm{Training}}$", "Training loss @ best test"])
    ax.set_title("Training loss")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def plot_confusion_matrix(path: str, dataset: TensorDataset, layers: list, labels: list) -> None:
    """
    Generate and save a normalized confusion matrix for network predictions.

    Runs the trained network on all samples in the dataset, collects predictions and ground truth
    labels, then creates a heatmap visualization of the confusion matrix.

    Parameters
    ----------
    path : str
        File path (without extension) where the PDF figure will be saved
    dataset : TensorDataset
        Dataset containing (input_data, labels) pairs for evaluation
    layers : list
        List containing [recurrent_layer, feedforward_layer] trained layer objects
    labels : list
        List of label names (e.g., ['A', 'B']) for axis tick labels
    vars_eprop : list
        List containing [beta_trace, beta_trace_out] decay factors (used for consistency)

    Returns
    -------
    None
        Saves confusion matrix heatmap to {path}.pdf

    Notes
    -----
    - Confusion matrix is normalized by true labels (rows sum to 1.0)
    - Uses seaborn heatmap with YlGnBu colormap
    - Figure size: 12x9 inches
    - Predictions are based on total spike counts per output neuron
    """
    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=False, num_workers=4)
    accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, _ = run_snn(inputs=x_local, layers=layers)

        # with output spikes
        m = torch.sum(spk_rec_readout, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())

    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
                         jj for jj in labels])
    plt.figure(figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    plt.savefig(
        f"{path}.pdf")


def get_network_activity(dataset: TensorDataset, layers: list) -> tuple:
    """
    Record network activity (spike trains) for all samples in a dataset.

    Runs the trained network in inference mode on all samples and collects spike recordings
    from both hidden and output layers for subsequent analysis or visualization.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset containing (input_data, labels) pairs for evaluation
    layers : list
        List containing [recurrent_layer, feedforward_layer] trained layer objects
    vars_eprop : list
        List containing [beta_trace, beta_trace_out] decay factors

    Returns
    -------
    tuple
        (accs, spk_rec_readout_list, spk_rec_hidden_list) where:
        - accs : list
            Classification accuracy for each batch
        - spk_rec_readout_list : list of numpy arrays
            Output layer spike trains [batch][samples, timesteps, neurons]
        - spk_rec_hidden_list : list of numpy arrays
            Hidden layer spike trains [batch][samples, timesteps, neurons]

    Notes
    -----
    - All computations performed with torch.no_grad() for efficiency
    - Returns data as numpy arrays (moved from GPU to CPU)
    - Predictions based on total spike counts per output neuron
    - Useful for creating raster plots and analyzing network dynamics
    """

    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=False, num_workers=4)
    accs = []
    spk_rec_readout_list = []
    spk_rec_hidden_list = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        with torch.no_grad():
            spk_rec_readout, recs = run_snn(
                inputs=x_local, layers=layers)

        _, spk_rec_hidden, _ = recs

        # with output spikes
        m = torch.sum(spk_rec_readout, 1)  # sum over time
        _, am = torch.max(m, 1)     # argmax over output units

        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        spk_rec_readout_list.append(spk_rec_readout.detach().cpu().numpy())
        spk_rec_hidden_list.append(spk_rec_hidden.detach().cpu().numpy())

    return accs, spk_rec_readout_list, spk_rec_hidden_list


def plot_network_activity(spr_recs: list, layer_names: list, figname: str = './figures') -> None:
    """
    Create and save raster plots visualizing spike activity across network layers.

    Generates a multi-panel figure showing spike raster plots for each layer, where each
    spike is plotted as a vertical line at its occurrence time for the corresponding neuron.

    Parameters
    ----------
    spr_recs : list
        List of spike recordings for each layer, where each element is a numpy array
        with shape [timesteps, neurons]. Each array contains binary spike data (0 or 1)
    layer_names : list
        List of layer names (e.g., ['Hidden layer', 'Readout layer']) for subplot titles
    figname : str, optional
        File path (without extension) where the PDF figure will be saved (default: './figures')

    Returns
    -------
    None
        Saves figure to {figname}.pdf and closes the figure

    Notes
    -----
    - Creates one subplot per layer in a vertical arrangement
    - Spike times are converted from timesteps to seconds using params['time_bin_size']
    - X-axis: Time in seconds
    - Y-axis: Neuron ID (one row per neuron)
    - Uses matplotlib.eventplot for efficient raster visualization
    - Line width is set to 0.3 for visibility
    - Figure is automatically sized and saved as PDF
    - Prints debug information about number of neurons and spike lists
    """
    nb_layers = len(layer_names)
    fig = plt.figure()
    for counter, name in enumerate(layer_names):
        # TODO plot hidden layer activity
        spk_per_layer = spr_recs[counter]
        num_neurons = spk_per_layer.shape[1]
        ax = fig.add_subplot(nb_layers, 1, counter+1)

        spikes_per_neuron = []
        for neuron_idx in range(spk_per_layer.shape[-1]):
            spk_times_per_neuron = np.where(spk_per_layer[:, neuron_idx])[0]
            spk_times_per_neuron = spk_times_per_neuron * \
                0.001*int(params['time_bin_size'])
            spikes_per_neuron.append(spk_times_per_neuron)

        # # TODO possible optimization
        # # Find the indices of spikes (value 1)
        # spike_times, neuron_ids = np.where(spk_per_layer == 1)
        # # Sort by neuron id and then by spike time
        # # sorted_indices = np.lexsort((spike_times, neuron_ids))
        # # # Get the sorted spike times and neuron ids
        # # sorted_spike_times = spike_times[sorted_indices]  # contains the neuron IDs
        # # sorted_neuron_ids = neuron_ids[sorted_indices]  # contains the according spike times
        # # # Group indices by neuron
        # # spikes_per_neuron = {neuron: sorted_spike_times[sorted_neuron_ids == neuron] for neuron in np.unique(sorted_neuron_ids)}

        # # Sort by neuron id and then by spike time
        # sorted_indices = np.lexsort((spike_times, neuron_ids))

        # # Get the sorted spike times and neuron ids
        # sorted_spike_times = spike_times[sorted_indices]
        # sorted_neuron_ids = neuron_ids[sorted_indices]

        # # Get the total number of neurons
        # num_neurons = spk_per_layer.shape[1]

        # # Include empty lists for neurons with no spikes
        # spikes_per_neuron = {neuron: sorted_spike_times[sorted_neuron_ids == neuron].tolist() for neuron in range(num_neurons)}

        # TODO possible colorcode by nb spikes
        # print(len(spikes_per_neuron))
        # print(len(range(num_neurons)))
        ax.eventplot(spikes_per_neuron, orientation="horizontal",
                     lineoffsets=range(num_neurons), linewidth=0.3, colors="k")
        ax.set_xlim(0, params["max_time"] * 0.001)
        ax.set_ylabel("Neuron ID")
        ax.set_title(f"{name} activity")
    ax.set_xlabel("Time [sec]")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{figname}.pdf")
    plt.close(fig)


# Load data and parameters
file_dir_data = './data/100Hz/'
if params["use_mechanoreceptor_encoding"]:
    file_name = file_dir_data + 'mechanoreceptor_encoded.pkl'
else:
    file_type = 'data_braille_letters_100Hz_th'
    file_thr = str(params["threshold"])
    file_name = file_dir_data + file_type + file_thr + '.pkl'


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


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient function for spike generation.
    Forward pass: Hard threshold (Heaviside step function)
    Backward pass: Smooth surrogate gradient (derivative of fast sigmoid)
    """
    @staticmethod
    def forward(ctx, input, gamma):
        """
        Forward pass: Generate spikes using hard threshold
        input: membrane potential minus threshold (mthr = mem - threshold)
        gamma: surrogate gradient scale factor
        """
        ctx.save_for_backward(input)
        ctx.gamma = gamma
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use surrogate gradient
        Approximates derivative with: gamma * max(0, 1 - |input|)
        """
        input, = ctx.saved_tensors
        gamma = ctx.gamma
        # Surrogate gradient: piecewise linear approximation
        grad_input = grad_output.clone()
        # Only compute gradient where surrogate is non-zero
        surrogate = gamma * torch.clamp(1 - torch.abs(input), min=0)
        grad_input = grad_input * surrogate
        return grad_input, None


spike_fn = SurrogateSpike.apply


class feedforward_layer:
    """
    Spiking feedforward layer with Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a fully-connected feedforward layer of spiking neurons with
    optional synaptic dynamics, exponential or linear membrane potential decay, and
    optional refractory period. Supports both e-prop and BPTT learning via surrogate gradients.

    Attributes
    ----------
    batch_size : int
        Maximum batch size for pre-allocation
    nb_inputs : int
        Number of input features/channels
    nb_neurons : int
        Number of neurons in this layer
    scale : float
        Weight initialization scale factor
    alpha : float
        Synaptic current decay factor (0 for no synapse, or exp(-dt/tau_syn))
    beta : float
        Membrane potential decay factor (exp(-dt/tau_mem) or linear decay rate)
    ref_per : int or None
        Refractory period duration in timesteps (None to disable)
    ff_weights : torch.Tensor
        Feedforward weight matrix [nb_neurons, nb_inputs]
    ref_per_tensor : torch.Tensor
        Tracks remaining refractory period timesteps per neuron [batch_size, nb_neurons]

    Notes
    -----
    - Weights initialized with Gaussian distribution and positive constraint applied
    - Spike generation uses hard threshold for e-prop, surrogate gradient for BPTT
    - Refractory period prevents neurons from spiking during cooldown
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_weight_scale, alpha, beta, ref_per=None):
        """
        Initialize feedforward spiking layer.

        Parameters
        ----------
        batch_size : int
            Maximum batch size for memory pre-allocation
        nb_inputs : int
            Number of input features/channels
        nb_neurons : int
            Number of neurons in this layer
        fwd_weight_scale : float
            Weight initialization scale (weights drawn from N(0, scale/sqrt(nb_inputs)))
        alpha : float
            Synaptic current decay factor (0.0 to disable synaptic dynamics)
        beta : float
            Membrane potential decay factor (exp(-dt/tau_mem) for exponential decay)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
        """
        self.batch_size = batch_size
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.scale = fwd_weight_scale
        self.alpha = alpha
        self.beta = beta
        self.ref_per = ref_per
        if ref_per is not None or ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=device)
        self.create_layer()

    def reset_refractory_perdiod_counter(self):
        """
        Reset all refractory period counters to zero.

        Called at the start of each forward pass to clear refractory states
        from previous batches.
        """
        self.ref_per_tensor = torch.zeros_like(self.ref_per_tensor)

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

        Creates and initializes the weight matrix with Gaussian distribution
        and ensures all weights are positive.

        Notes
        -----
        - Feedforward weights: [nb_neurons, nb_inputs]
          Initialization: N(0, scale/sqrt(nb_inputs))
        - Positive constraint applied to ensure excitatory connections only
        - Requires gradient for learning
        """
        self.ff_weights = torch.empty((self.nb_neurons, self.nb_inputs),
                                      device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.scale/np.sqrt(self.nb_inputs))

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
                          device=device, dtype=dtype)
        new_syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                              device=device, dtype=dtype)
        mem = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=device, dtype=dtype)
        out = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=device, dtype=dtype)

        # always reset the refractory period counter at the beginning of a new forward pass
        if self.ref_per is not None:
            self.reset_refractory_perdiod_counter()

        mem_rec = []
        spk_rec = []
        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            # Use surrogate gradient for BPTT compatibility
            if params["use_eprop"]:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient
                out = spike_fn(mthr, params["gamma"])
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

            if params["use_linear_decay"]:
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
    batch_size : int
        Maximum batch size for pre-allocated tensors
    nb_inputs : int
        Number of input features/channels
    nb_neurons : int
        Number of recurrent neurons in this layer
    fwd_scale : float
        Feedforward weight initialization scale factor
    rec_scale : float
        Recurrent weight initialization scale factor
    alpha : float
        Synaptic current decay factor (0 for no synapse, or exp(-dt/tau_syn))
    beta : float
        Membrane potential decay factor (exp(-dt/tau_mem) or linear decay rate)
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

    def __init__(self, batch_size, nb_inputs, nb_neurons, fwd_scale, rec_scale, alpha, beta, ref_per=None):
        """
        Initialize recurrent spiking layer.

        Parameters
        ----------
        batch_size : int
            Maximum batch size for memory pre-allocation
        nb_inputs : int
            Number of input features/channels
        nb_neurons : int
            Number of recurrent neurons in this layer
        fwd_scale : float
            Feedforward weight initialization scale (N(0, fwd_scale/sqrt(nb_inputs)))
        rec_scale : float
            Recurrent weight initialization scale (N(0, rec_scale/sqrt(nb_neurons)))
        alpha : float
            Synaptic current decay factor (0.0 to disable synaptic dynamics)
        beta : float
            Membrane potential decay factor (exp(-dt/tau_mem) for exponential decay)
        ref_per : int or None, optional
            Refractory period duration in timesteps (default: None, disabled)
        """
        self.batch_size = batch_size
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.fwd_scale = fwd_scale
        self.rec_scale = rec_scale
        self.alpha = alpha
        self.beta = beta
        self.ref_per = ref_per
        if ref_per is not None or ref_per > 0:
            self.ref_per_tensor = torch.zeros(
                (batch_size, nb_neurons), device=device)
        self.create_layer()

    def reset_refractory_perdiod_counter(self):
        """
        Reset all refractory period counters to zero.

        Called at the start of each forward pass to clear refractory states
        from previous batches.
        """
        self.ref_per_tensor = torch.zeros_like(self.ref_per_tensor)

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
          Initialization: N(0, fwd_scale/sqrt(nb_inputs))
        - Recurrent weights: [nb_neurons, nb_neurons]
          Initialization: N(0, rec_scale/sqrt(nb_neurons))
        - Both require gradients for learning
        """
        self.ff_weights = torch.empty((self.nb_neurons, self.nb_inputs),
                                      device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.ff_weights, mean=0.0,
                              std=self.fwd_scale/np.sqrt(self.nb_inputs))
        self.rec_weights = torch.empty((self.nb_neurons, self.nb_neurons),
                                       device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(self.rec_weights, mean=0.0,
                              std=self.rec_scale/np.sqrt(self.nb_neurons))

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
                          device=device, dtype=dtype)
        new_syn = torch.zeros((input_activity.shape[0], self.nb_neurons),
                              device=device, dtype=dtype)
        mem = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=device, dtype=dtype)
        out = torch.zeros((input_activity.shape[0], self.nb_neurons),
                          device=device, dtype=dtype)

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
            if params["use_eprop"]:
                # For e-prop, use hard threshold (no gradient needed through spikes)
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1
            else:
                # For BPTT, use surrogate gradient
                out = spike_fn(mthr, params["gamma"])
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

            if params["use_linear_decay"]:
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


if __name__ == '__main__':
    # Print learning algorithm being used
    print(f"\n{'='*60}")
    print(f"Training with: {'e-prop' if params['use_eprop'] else 'BPTT (Backpropagation Through Time)'}")
    print(f"{'='*60}\n")
    
    # here we can set how often we wat to run the training to get some statistics
    min_nb_neurons = 50
    max_nb_neurons = 100
    nb_neurons_step = 10
    best_acc = 0.0
    
    # Define results file path
    results_file = f"./results/braille_reading_rsnn_eprop_reduce_label_{min_nb_neurons}_to_{max_nb_neurons}_neurons_{letters[0]}_vs_{letters[1]}_th_{params['threshold']}_{params['ref_per_timesteps']}_ref_per.npz"
    
    # Load existing results if available
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        loaded = np.load(results_file, allow_pickle=True)
        acc_train_dict = loaded['acc_train'].item() if loaded['acc_train'].ndim == 0 else {}
        acc_test_dict = loaded['acc_test'].item() if loaded['acc_test'].ndim == 0 else {}
        loss_train_dict = loaded['loss_train'].item() if loaded['loss_train'].ndim == 0 else {}
        nb_hidden_list = loaded.get('nb_hidden_list', [])
        if nb_hidden_list.ndim == 0:
            nb_hidden_list = list(nb_hidden_list.item())
        else:
            nb_hidden_list = list(nb_hidden_list)
        print(f"Found existing data for {len(nb_hidden_list)} neuron counts: {nb_hidden_list}")
    else:
        print("No existing results found, starting fresh")
        acc_train_dict = {}
        acc_test_dict = {}
        loss_train_dict = {}
        nb_hidden_list = []

    # always use the same data split
    if create_validation:
        ds_train, ds_test, ds_validation, labels, nb_channels, data_steps = load_and_extract(
            params, file_name, letter_written=letters, create_validation=create_validation)
    else:
        ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract(
            params, file_name, letter_written=letters, create_validation=create_validation)

    pbar_nb_neurons = tqdm(range(min_nb_neurons, max_nb_neurons+1, nb_neurons_step),
                           position=0, total=max_nb_neurons, leave=True)
    for nb_hidden in pbar_nb_neurons:
        # Clear GPU cache at the start of each iteration
        torch.cuda.empty_cache()

        pbar_nb_neurons.set_description(
            f"{nb_hidden}/{max_nb_neurons}")
        # load data for each repetition indepoently to get different splits
        if nb_hidden == min_nb_neurons:
            print("Number of training data %i." % len(ds_train))
            print("Number of testing data %i." % len(ds_test))
            if create_validation:
                print("Number of validation data %i." % len(ds_validation))
            print("Number of outputs %i." % len(np.unique(labels)))
            print("Number of timesteps %i." % data_steps)
            print("Delayed output ", params["delayed_output"])
            if params["no_synapse"]:
                print(f"No synapse dynamics.")
            if params["lower_bound"]:
                print(
                    f"Clamp membrane voltage to: {params['lower_bound']}.")
            if params["use_linear_decay"]:
                print(f"Use linear decay.")
            else:
                print(f"Use exponential decay.")
            if params["ref_per_timesteps"]:
                print(
                    f"Refractory period set to {params['ref_per_timesteps']} simulation timesteps.")
            print("Input duration %fs" % (data_steps*params["time_step"]))
            print("---------------------------\n")

        # initialize and train network
        loss_hist, acc_hist, best_layers, vars_eprop = build_and_train(
            params, ds_train, ds_test, nb_hidden=nb_hidden, epochs=epochs)

        # get validation results
        if create_validation:
            val_acc = compute_classification_accuracy(
                dataset=ds_validation, layers=best_layers)

        # save the best layer
        torch.save(
            best_layers, f'./model/best_model_{letters[0]}_vs_{letters[1]}_{nb_hidden}_neurons_th_{params["threshold"]}_{params["ref_per_timesteps"]}_ref_per.pt')

        # Store the training histories in dictionaries
        acc_train_dict[nb_hidden] = acc_hist[0]
        acc_test_dict[nb_hidden] = acc_hist[1]
        loss_train_dict[nb_hidden] = loss_hist
        
        # Update nb_hidden list if this is a new entry
        if nb_hidden not in nb_hidden_list:
            nb_hidden_list.append(nb_hidden)
            nb_hidden_list.sort()  # Keep sorted for consistency
        
        # Save results after each iteration to prevent data loss
        np.savez(results_file,
                 acc_train=acc_train_dict,
                 acc_test=acc_test_dict,
                 loss_train=loss_train_dict,
                 nb_hidden_list=np.array(nb_hidden_list))
        print(f"Results saved to {results_file}")

        # ### Lets plot the training curve and the confusion matrix for this specific number of neurons
        # Plot only the current run's data (last element in the list)
        plot_training_perfromance(
            path=f"./figures/best_model_{letters[0]}_vs_{letters[1]}_{nb_hidden}_neurons_th_{params['threshold']}_{params['ref_per_timesteps']}_ref_per_training_performance", acc_train=np.array([acc_hist[0]]), acc_test=np.array([acc_hist[1]]), loss_train=np.array([loss_hist]))
        # plotting the confusion matrix
        plot_confusion_matrix(
            path=f"./figures/best_model_{letters[0]}_vs_{letters[1]}_{nb_hidden}_neurons_th_{params['threshold']}_{params['ref_per_timesteps']}_ref_per_confusion_matrix", dataset=ds_test, layers=best_layers, labels=letters, vars_eprop=vars_eprop)

        #####################################
        ### Lets create some raster plots ###
        #####################################

        # plotting the network activity
        accs, spk_rec_readout_array, spk_rec_hidden_array = get_network_activity(
            ds_test, layers=best_layers, vars_eprop=vars_eprop)

        layer_names = ["Hidden layer", "Readout layer"]
        nb_layers = len(layer_names)

        total_nb_batches = len(accs)

        # select the batches to plot
        if NB_BATCHES_TO_PLOT > total_nb_batches:
            # print(
            #     f"WARNING: Not enough batches to plot. Will plot all {total_nb_batches} batches instead of the asked {NB_BATCHES_TO_PLOT}. Lower the number to avoid this warning.")
            batch_selection = range(NB_BATCHES_TO_PLOT)
        elif NB_BATCHES_TO_PLOT == total_nb_batches:
            # print(f"Plotting all {total_nb_batches} batches.")
            batch_selection = range(NB_BATCHES_TO_PLOT)
        else:
            # print(
            #     f"Plotting {NB_BATCHES_TO_PLOT} random batches (out of {total_nb_batches}).")
            found_unique = False
            while not found_unique:
                batch_selection = np.random.choice(
                    total_nb_batches, NB_BATCHES_TO_PLOT)
                if len(np.unique(batch_selection)) == NB_BATCHES_TO_PLOT:
                    found_unique = True

        for batch_idx in batch_selection:
            batch_acc = accs[batch_idx]
            # [trials, timesteps, neurons]
            spk_rec_readout_batch = spk_rec_readout_array[batch_idx]
            # [trials, timesteps, neurons]
            spk_rec_hidden_batch = spk_rec_hidden_array[batch_idx]
            # select random trials to plot
            total_nb_trials = len(spk_rec_readout_batch)
            if NB_TRIALS_TO_PLOT > total_nb_trials:
                # print(
                #     f"WARNING: Not enough trials to plot. Will plot all {total_nb_trials} trials instead of the asked {NB_TRIALS_TO_PLOT}. Lower the number to avoid this warning.")
                trial_selection = range(NB_BATCHES_TO_PLOT)
            elif NB_TRIALS_TO_PLOT == total_nb_trials:
                # print(f"Plotting all {total_nb_trials} trials.")
                trial_selection = range(NB_TRIALS_TO_PLOT)
            else:
                # print(
                #     f"Plotting {NB_TRIALS_TO_PLOT} random trials (out of {total_nb_trials}).")
                found_unique = False
                while not found_unique:
                    trial_selection = np.random.choice(
                        total_nb_trials, NB_TRIALS_TO_PLOT)
                    if len(np.unique(trial_selection)) == NB_TRIALS_TO_PLOT:
                        found_unique = True

            for trial_idx in trial_selection:
                spr_recs = [spk_rec_hidden_batch[trial_idx],
                            spk_rec_readout_batch[trial_idx]]
                # TODO include more specifics into the figure name
                plot_network_activity(
                    spr_recs, layer_names, figname=f"./figures/best_model_{letters[0]}_vs_{letters[1]}_{nb_hidden}_neurons_th_{params['threshold']}_{params['ref_per_timesteps']}_ref_per_network_activity")

        # Clean up large variables to free memory
        del loss_hist, acc_hist, best_layers, accs, spk_rec_readout_array, spk_rec_hidden_array
        torch.cuda.empty_cache()

    print(f"\nTraining complete! Final results saved to {results_file}")
    print(f"Completed training for {len(nb_hidden_list)} neuron counts: {nb_hidden_list}")
