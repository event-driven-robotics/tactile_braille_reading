import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.data_manager import load_and_extract
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
# set the number of epochs you want to train the network (default = 300)
epochs = 100
# bit_resolution_list = [16, 14, 12, 10, 8, 6, 4, 2, 1]  # possible bit resolutions
bit_resolution_list = ["baseline"]  # possible bit resolutions
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

for bit_resolution in bit_resolution_list:
    best_acc = 0.0
    # for actual_weight_precision in weight_precision:
    results_dict = {
        "bit_resolution": bit_resolution,
        "nb_repetitions": max_repetitions,
        "nb_epochs": epochs,
        "training_results": [],
        "validation_results": [],
        "test_results": [],
    }
    if bit_resolution != "baseline":
        # calculate possible weight values
        # determines in how many increments we seperate values between min and max (inlc. both)
        number_of_increments = 2**bit_resolution
        weight_range = [-1, 1]  # weight range
        possible_weight_values = np.linspace(
            weight_range[0], weight_range[1], number_of_increments)
        possible_weight_values = torch.as_tensor(
            possible_weight_values, device=device, dtype=dtype)
    else:
        possible_weight_values = bit_resolution

    for repetition in range(max_repetitions):
        # load data
        ds_train, ds_validation, ds_test, labels, nb_channels, data_steps, time_step = load_and_extract(
            params=params, file_name=file_name, letter_written=letters)
        if repetition == 0:
            LOG.debug("Number of training data %i" % len(ds_train))
            LOG.debug("Number of validation data %i" % len(ds_validation))
            LOG.debug("Number of testing data %i" % len(ds_test))
            LOG.debug("Number of outputs %i" % len(np.unique(labels)))
            LOG.debug("Number of timesteps %i" % data_steps)
            LOG.debug("Input duration %fs" % (data_steps*time_step))
            LOG.debug("---------------------------\n")

        # build the network
        # TODO hand over logger!
        layers, time_constants = build(params=params, nb_channels=nb_channels, ste_fn=ste_fn, nb_hidden=450, nb_outputs=len(
            np.unique(labels)), time_step=time_step, possible_weight_values=possible_weight_values, device=device, logger=LOG)

        # train the network
        # a fixed learning rate is already defined within the train function, that's why here it is omitted
        loss_hist, accs_hist, best_layers = train(params=params, spike_fn=spike_fn, ste_fn=ste_fn, dataset_train=ds_train, batch_size=batch_size, lr=lr, nb_epochs=epochs,
                                                  layers=layers, time_constants=time_constants, dataset_validation=ds_validation, possible_weight_values=possible_weight_values, device=device, dtype=dtype, logger=LOG)

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

        LOG.debug("Final results: ")
        LOG.debug("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(
            acc_best_train, acc_test_at_best_train, idx_best_train+1))
        LOG.debug("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
            acc_best_test, acc_train_at_best_test, idx_best_test+1))
        LOG.debug(
            "------------------------------------------------------------------------------------\n")

        # get test results
        val_acc, trues, preds, activity_record = validate_model(dataset=ds_test, layers=best_layers, time_constants=time_constants, batch_size=batch_size, spike_fn=spike_fn, nb_input_copies=params[
            'nb_input_copies'], device=device, dtype=torch.float, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)

        # safe overall best layer
        if np.mean(val_acc) > best_acc:
            very_best_layer = best_layers
            best_acc = np.mean(val_acc)
            best_trues = trues
            best_preds = preds
            best_activity_record = activity_record

        results_dict["training_results"].append(accs_hist[0])
        results_dict["validation_results"].append(accs_hist[1])
        results_dict["test_results"].append(np.mean(val_acc))

        # free memory
        # del ds_train, ds_validation, ds_test
        torch.clear_autocast_cache()

    LOG.debug("*************************")
    LOG.debug("* Best: {:.2f} %         *" .format(best_acc*100))
    LOG.debug("*************************")

    # save the best layer
    torch.save(very_best_layer,
               f'./model/best_model_th{threshold}_{bit_resolution}_bit_resolution.pt')

    # save results
    torch.save(
        results_dict, f'./results/results_th{threshold}_{bit_resolution}_bit_resolution.pt')

    # calc mean and std
    acc_mean_train = np.mean(
        results_dict["training_results"], axis=0)
    acc_std_train = np.std(
        results_dict["training_results"], axis=0)
    acc_mean_test = np.mean(
        results_dict["validation_results"], axis=0)
    acc_std_test = np.std(results_dict["validation_results"], axis=0)
    # find best validation trial and epoch
    best_trial, best_epoch = np.where(np.max(
        results_dict["validation_results"]) == results_dict["validation_results"])
    best_trial, best_epoch = best_trial[0], best_epoch[0]

    plt.figure()
    # plot mean and std
    plt.plot(range(1, len(acc_mean_train)+1), 100 *
             np.array(acc_mean_train), color='blue')
    plt.plot(range(1, len(acc_mean_test)+1), 100 *
             np.array(acc_mean_test), color='orangered')
    plt.fill_between(range(1, len(acc_mean_train)+1), 100*(acc_mean_train +
                     acc_std_train), 100*(acc_mean_train-acc_std_train), color='cornflowerblue')
    plt.fill_between(range(1, len(acc_mean_test)+1), 100*(acc_mean_test +
                     acc_std_test), 100*(acc_mean_test-acc_std_test), color='sandybrown')
    # plot best trial
    plt.plot(range(1, len(results_dict["training_results"][best_trial])+1), 100*np.array(
        results_dict["training_results"][best_trial]), color='blue', linestyle='dashed')
    plt.plot(range(1, len(results_dict["validation_results"][best_trial])+1), 100*np.array(
        results_dict["validation_results"][best_trial]), color='orangered', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim((0, 105))
    plt.legend(["Training", "Test"], loc='lower right')
    plt.savefig(
        f"{path}/rsnn_thr_{threshold}_{bit_resolution}_bit_resolution_acc.png", dpi=300)

    # get test results
    test_acc, trues, preds, activity_record = validate_model(dataset=ds_test, layers=very_best_layer, time_constants=time_constants, batch_size=batch_size, spike_fn=spike_fn, nb_input_copies=params[
        'nb_input_copies'], device=device, dtype=dtype, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)

    # find best test
    idx_best_test = np.argmax(results_dict["test_results"])

    # plotting the confusion matrix
    ConfusionMatrix(out_path=path, trues=trues[idx_best_test], preds=preds[idx_best_test], labels=letters, threshold=threshold, bit_resolution=bit_resolution,
                    use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out)

    # plotting the network activity
    NetworkActivity(out_path=path, spk_recs=activity_record[idx_best_test], threshold=threshold, bit_resolution=bit_resolution,
                    use_trainable_tc=use_trainable_tc, use_trainable_out=use_trainable_out)
