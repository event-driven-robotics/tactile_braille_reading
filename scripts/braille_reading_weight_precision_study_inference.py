import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from utils.data_manager import create_splits, load_and_extract
from utils.train_manager import build, check_cuda, train, validate_model
from utils.visualizer import ConfusionMatrix, NetworkActivity

# TODO change name according to precision tested!
datetime = str(datetime.datetime.now())
logger_name = f'./logs/{datetime.split(" ")[0]}_{datetime.split(" ")[1].split(".")[0]}_braille_reading_weight_precision_study.log'
logging.basicConfig(filename=f'{logger_name}',
                    filemode='a+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
LOG = logging.getLogger(f'{logger_name}')

# set variables
use_seed = False
threshold = 2  # possible values are: 1, 2, 5, 10
bit_resolution_list = ["baseline", 16, 14, 12,
                       10, 8, 6, 4, 2, 1]  # possible bit resolutions
dynamic_clamping = False  # if True, the weights are clamped to the range after training

# weight range extracted from unconstrained training
weight_range = [-0.5, 0.5]
max_repetitions = 5

use_trainable_out = False
use_trainable_tc = False
use_dropout = False
batch_size = 128
lr = 0.0015

dtype = torch.float  # float32

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# create folder to safe figures later
path = './figures'
isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

# check for available GPU and distribute work
device = check_cuda()

# use fixed seed for reproducable results
if use_seed:
    seed = 42  # "Answer to the Ultimate Question of Life, the Universe, and Everything"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LOG.debug("Seed set to {}".format(seed))
else:
    LOG.debug("Shuffle data randomly")

# Load data and parameters
file_dir_data = '/mnt/disk1/data/tactile_braille/old_40Hz/'  # old_40Hz, new_100Hz
file_type = 'data_braille_letters_th_'
file_thr = str(threshold)
file_name = file_dir_data + file_type + file_thr + '.pkl'

file_dir_params = './parameters/'
param_filename = 'parameters_th' + str(threshold) + '.txt'
file_name_parameters = file_dir_params + param_filename
params = {}
with open(file_name_parameters) as file:
    for line in file:
        (key, value) = line.split()
        if key == 'time_bin_size' or key == 'nb_input_copies':
            params[key] = int(value)
        else:
            params[key] = np.double(value)


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


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = params['scale']

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
        out[input > 0] = 1.0
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

# load data
data, labels, nb_channels, data_steps, time_step = load_and_extract(
    params=params, file_name=file_name, letter_written=letters)
ds_total = TensorDataset(data, labels)

for bit_resolution in bit_resolution_list:
    if bit_resolution != "baseline":
        # calculate possible weight values
        # determines in how many increments we seperate values between min and max (inlc. both)
        number_of_increments = 2**bit_resolution
        possible_weight_values = np.linspace(
            weight_range[0], weight_range[1], number_of_increments)
        possible_weight_values = torch.as_tensor(
            possible_weight_values, device=device, dtype=dtype)
    else:
        possible_weight_values = bit_resolution

    for repetition in range(max_repetitions):
        if repetition == 0:
            LOG.debug("Number of data %i" % len(ds_total))
            LOG.debug("Number of outputs %i" % len(np.unique(labels)))
            LOG.debug("Number of timesteps %i" % data_steps)
            LOG.debug("Input duration %fs" % (data_steps*time_step))
            LOG.debug("---------------------------\n")

        # build the network
        _, time_constants = build(params=params, nb_channels=nb_channels, ste_fn=ste_fn, nb_hidden=450, nb_outputs=len(
            np.unique(labels)), time_step=time_step, possible_weight_values=possible_weight_values, device=device, logger=LOG)

        # load the baseline network
        layers = torch.load(
            f'./model/best_model_th{threshold}_baseline_bit_resolution_run_{repetition+1}.pt')

        if dynamic_clamping:
            clamp_max, clamp_min = np.max([torch.max(w1).detach().cpu().numpy(), torch.max(w2).detach().cpu().numpy(), torch.max(v1).detach().cpu(
            ).numpy()]), np.min([torch.min(w1).detach().cpu().numpy(), torch.min(w2).detach().cpu().numpy(), torch.min(v1).detach().cpu().numpy()])
            clamp_max, clamp_min = 1.2*clamp_max, 1.2*clamp_min  # add some margin
            # calculate possible weight values
            # determines in how many increments we seperate values between min and max (inlc. both)
            number_of_increments = 2**bit_resolution
            possible_weight_values = np.linspace(
                clamp_min, clamp_max, number_of_increments)
            possible_weight_values = torch.as_tensor(
                possible_weight_values, device=device, dtype=dtype)
        else:
            # calculate possible weight values
            # determines in how many increments we seperate values between min and max (inlc. both)
            number_of_increments = 2**bit_resolution
            possible_weight_values = np.linspace(
                -0.5, 0.5, number_of_increments)
            possible_weight_values = torch.as_tensor(
                possible_weight_values, device=device, dtype=dtype)

        # get test results
        val_acc, trues, preds, activity_record = validate_model(dataset=ds_total, layers=layers, time_constants=time_constants, batch_size=batch_size, spike_fn=spike_fn, nb_input_copies=params[
            'nb_input_copies'], device=device, dtype=torch.float, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)

        # plotting the confusion matrix
        ConfusionMatrix(out_path=path, trues=trues, preds=preds, labels=letters, threshold=threshold, bit_resolution=bit_resolution,
                        use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, repetition=repetition+1)

        # visualize network activity of the best perfoming batch
        NetworkActivity(out_path=path, spk_recs=activity_record[np.argmax(val_acc)], threshold=threshold, bit_resolution=bit_resolution,
                        use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, repetition=repetition+1)

        # free memory
        torch.clear_autocast_cache()

    LOG.debug("*************************")
    LOG.debug("\n\n\n")
