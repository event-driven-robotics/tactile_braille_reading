import argparse
import os

import numpy as np
import torch

from utils.data_management import load_and_extract
from utils.network_definition import SRNN, SRNN_OG

torch.cuda.empty_cache()
# torch.autograd.set_detect_anomaly(True)

global default_device
default_device = 'cuda:0'

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
possible_weight = diff_cap * q

factor = 10 ** 3
possible_weight = torch.floor(possible_weight * factor) / factor


def default_parser(parser):
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train')
    parser.add_argument('--time_bin_size', type=int, default=3,
                        help='time bin size in ms (default: 3)')
    parser.add_argument('--max_time', type=int, default=3501,
                        help='max time in ms (default: 3501)')
    parser.add_argument('--device', type=str, default=default_device,
                        help='device to use change to cpu if no gpu available')
    parser.add_argument('--dtype', type=str, default='torch.float',
                        help='data type to use (default: floa)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--batch_size_test', type=int, default=128,
                        help='input batch size for test (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0015,
                        help='learning rate (default:0.0008)')
    parser.add_argument('--n_rec_alif', type=int, default=0,
                        help='number of recurrent ALIF neurons (default: 0)')
    parser.add_argument('--tau_mem', type=float, default=0.06,
                        help='membrane time constant (default: 0.06)')
    parser.add_argument('--tau_syn', type=float, default=0.0,
                        help='synaptic time constant (default: 0.0)')
    parser.add_argument('--lower_bound', type=float, default=-1.0,
                        help='lower bound for membrane potential (default: -1.0)')
    parser.add_argument('--ref_per_timesteps', type=int, default=1,
                        help='refractory period in time steps (default: 1)')
    parser.add_argument('--eprop', type=bool, default=False,
                        help='use event propagation as alghorithm (default: False)')
    parser.add_argument('--tau_trace', type=float,
                        default=0.08, help='trace decay (default: 0.08)')
    parser.add_argument('--tau_trace_rec', type=float,
                        default=0.105, help='trace decay (default: 0.105)')
    parser.add_argument('--tau_adaptive_thr', type=float, default=0.07,
                        help='adaptive threshold decay (default: 0.07)')
    parser.add_argument('--dump_thr', type=float, default=0.1,
                        help='dumping threshold factor for alif neuron(default: 0.1)')
    parser.add_argument('--fwd_weight_scale', type=float, default=1,
                        help='forward weight scale for initialization (default: 1)')
    parser.add_argument('--weight_scale_factor', type=float, default=0.02,
                        help='recurrent weight scale for initialization (default: 0.02)')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='gamma factor for surrogate gradient (default: 0.3)')
    parser.add_argument('--reg_spikes', type=float, default=0.0015,
                        help='regularization factor for spikes (default: 0.0015)')
    parser.add_argument('--reg_neurons', type=float, default=0.0,
                        help='regularization factor for neurons (default: 0.0)')
    parser.add_argument('--firing_threshold', type=float,
                        default=1.0, help='firing threshold (default: 1.0)')


parser = argparse.ArgumentParser(description='Training a RSNN on BRAILLE')
default_parser(parser)
global dict_args
dict_args = vars(parser.parse_args())

#dict_args.update({"eprop": False})
#dict_args.update({"batch_size": 4})
#dict_args.update({"lr": 0.0008})

# use fixed seed for reproducable results
if use_seed:
    seed = 42  # "Answer to the Ultimate Question of Life, the Universe, and Everything"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")

# Load data and parameters
file_dir_data = './data/100Hz/'
# file_type = 'data_braille_letters_th_new'
file_type = 'data_braille_letters_100Hz_th'
file_thr = str(threshold)
file_name = file_dir_data + file_type + file_thr + '.pkl'

test_og = True

if __name__ == '__main__':

    ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract(
        dict_args, file_name, letter_written=letters)

    nb_inputs = nb_channels
    nb_hidden = 450
    nb_outputs = len(np.unique(labels))
    if test_og:
        # Use the original SRNN class
        rsnn = SRNN_OG(nb_inputs=nb_inputs, nb_hidden=nb_hidden,
                          nb_output=nb_outputs, dict_args=dict_args)
        loss_hist, accs_hist, best_acc_layers = rsnn.train(ds_train, ds_test, labels, possible_weight)
    else:
        rsnn = SRNN(nb_inputs=nb_inputs, nb_hidden=nb_hidden,
                    nb_output=nb_outputs, dict_args=dict_args)
        result_rsnn = rsnn.train_bptt(ds_train, ds_test)
