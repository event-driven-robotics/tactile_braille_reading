import numpy as np
import matplotlib.pyplot as plt
import torch


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

for bit_resolution in bit_resolution_list:
    # I evaluate study
    results_dict = torch.load(f'./results/results_th{threshold}_{bit_resolution}_bit_resolution.pt')

    # II evaluate weight distribution for unconstained case
    very_best_layer = torch.load(f'./model/best_model_th{threshold}_{bit_resolution}_bit_resolution.pt')  # w1, w2, v1
    # extract max and min values
    weights_max_min = []
    for layer in very_best_layer:
        weights_max_min.append([torch.max(layer).detach().cpu().numpy(), torch.min(layer).detach().cpu().numpy()])
    weights_max_min = np.array(weights_max_min)
    total_max = np.max(weights_max_min[:, 0])
    total_min = np.min(weights_max_min[:, 1])
    total_max_std = np.std(weights_max_min[:, 0])
    total_min_std = np.std(weights_max_min[:, 1])
    print("Absolute max: {:.3f} +- {:.3f}" .format(total_max, total_max_std))
    print("Absolute min: {:.3f} +- {:.3f}" .format(total_min, total_min_std))

    print("STOP")
