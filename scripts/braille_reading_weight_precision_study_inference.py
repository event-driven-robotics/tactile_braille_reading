import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset

from utils.data_manager import create_folder, load_and_extract
from utils.train_manager import build, check_cuda, validate_model
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
bit_resolution_list = [16, 15, 14, 13, 12, 11,
                       10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # possible bit resolutions
dynamic_clamping = False  # if True, the weights are clamped to the range after training

# weight range extracted from unconstrained training
weight_range = [-0.5, 0.5]
folds = 5
kfold = KFold(n_splits=5, shuffle=True)

use_trainable_out = False
use_trainable_tc = False
use_dropout = False
batch_size = 128
lr = 0.0015

dtype = torch.float  # float32

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# create folder to safe figures later
# create folders to safe everything
if dynamic_clamping:
    study_type = 'dynamic_clamping'
else:
    study_type = 'static_clamping'
path = f'./figures/inference/{study_type}'
fig_path = f'./figures/inference/{study_type}'
create_folder(fig_path)
results_path = f'./results/inference/{study_type}'
create_folder(results_path)

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
# ds_total = TensorDataset(data, labels)
max_repetitions = 5
total_mean_list = []
total_std_list = []
for bit_resolution in bit_resolution_list:
    print(f"Testing bit resolution {bit_resolution}...")
    # acc_per_split, trues_over_splits, preds_over_splits = [], [], []
    results_dict = {
        "bit_resolution": bit_resolution,
        "nb_repetitions": max_repetitions,
        "acc_per_split": [],
        "trues_over_splits": [],
        "preds_over_splits": [],
    }
    for repetition in range(max_repetitions):
        for _, test_idc in kfold.split(data, labels):
            ds_split = TensorDataset(data[test_idc], labels[test_idc])
            # build the network (never use quntization during build, applied later)
            _, time_constants = build(params=params, nb_channels=nb_channels, ste_fn=ste_fn, nb_hidden=450, nb_outputs=len(
                np.unique(labels)), time_step=time_step, bit_resolution='baseline', dynamic_clamping=dynamic_clamping, device=device, logger=LOG)

            # load the baseline network
            layers = torch.load(
                f'./model/best_model_th{threshold}_baseline_bit_resolution_run_{repetition+1}.pt')

            if dynamic_clamping:
                clamp_max, clamp_min = np.max([torch.max(layers[0]).detach().cpu().numpy(), torch.max(layers[1]).detach().cpu().numpy(), torch.max(layers[2]).detach().cpu(
                ).numpy()]), np.min([torch.min(layers[0]).detach().cpu().numpy(), torch.min(layers[1]).detach().cpu().numpy(), torch.min(layers[2]).detach().cpu().numpy()])
                clamp_max, clamp_min = 1.2*clamp_max, 1.2*clamp_min  # add some margin
            else:
                clamp_max, clamp_min = weight_range[1], weight_range[0]
            # calculate possible weight values
            # determines in how many increments we seperate values between min and max (inlc. both)
            number_of_increments = 2**bit_resolution
            possible_weight_values = torch.as_tensor(np.linspace(
                clamp_min, clamp_max, number_of_increments), device=device, dtype=dtype)

            # use the STE function for quantization
            for layer_idx in range(len(layers)):
                try:
                    # fast but memory intensive
                    layers[layer_idx].data.copy_(
                        ste_fn(layers[layer_idx].data, possible_weight_values))
                except:
                    # slower but memory efficient
                    for neuron_idx in range(len(layers[layer_idx])):
                        layers[layer_idx][neuron_idx].data.copy_(
                            ste_fn(layers[layer_idx][neuron_idx].data, possible_weight_values))
                        
            # get test results
            val_acc, trues, preds, activity_record = validate_model(dataset=ds_split, layers=layers, time_constants=time_constants, batch_size=batch_size, spike_fn=spike_fn, nb_input_copies=params[
                'nb_input_copies'], device=device, dtype=torch.float, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)

            # TODO write results to variable to plot later
            results_dict["acc_per_split"].extend(val_acc)
            results_dict["trues_over_splits"].extend(trues)
            results_dict["preds_over_splits"].extend(preds)

            # visualize network activity of the best perfoming batch
            NetworkActivity(out_path=path, spk_recs=activity_record[np.argmax(val_acc)], threshold=threshold, bit_resolution=bit_resolution,
                            use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, repetition=repetition+1)

            # free memory
            torch.clear_autocast_cache()

    # plotting the confusion matrix
    ConfusionMatrix(out_path=path, trues=results_dict["trues_over_splits"], preds=results_dict["preds_over_splits"], labels=letters, threshold=threshold, bit_resolution=bit_resolution,
                    use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out, repetition="all")
    total_mean_list.append(np.mean(results_dict["acc_per_split"]))
    total_std_list.append(np.std(results_dict["acc_per_split"]))

LOG.debug("*************************")
LOG.debug("\n\n\n")

# save results
torch.save(
    results_dict, f'{results_path}/results_th{threshold}_{bit_resolution}_bit_resolution.pt')
