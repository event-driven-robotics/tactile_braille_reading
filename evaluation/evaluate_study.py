import numpy as np
import matplotlib.pyplot as plt
import torch


# set variables
use_seed = False
threshold = 2  # possible values are: 1, 2, 5, 10
# set the number of epochs you want to train the network (default = 300)
epochs = 100
bit_resolution_list = [16, 14, 12, 10, 8, 6, 4, 2, 1]  # possible bit resolutions
# bit_resolution_list = ["baseline"]  # possible bit resolutions
max_repetitions = 5

use_trainable_out = False
use_trainable_tc = False
use_dropout = False
batch_size = 128
lr = 0.0015

dtype = torch.float  # float32

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# get baseline
training_means_list = []
training_stds_list = []
validation_means_list = []
validation_stds_list = []
test_means_list = []
test_stds_list = []

results_baseline = torch.load(f'./results/results_th{threshold}_baseline_bit_resolution.pt')
layers_baseline = []
for repetition in range(max_repetitions):
    layers_baseline.append(torch.load(f'./model/best_model_th{threshold}_baseline_bit_resolution_run_{repetition+1}.pt'))  # w1, w2, v1

training_acc_baseline = np.array(results_baseline["training_results"])
training_acc_baseline_max = np.max(training_acc_baseline, axis=1)
training_means_list.append(np.mean(training_acc_baseline_max))
training_stds_list.append(np.std(training_acc_baseline_max))

validation_acc_baseline = np.array(results_baseline["validation_results"])
validation_acc_baseline_max = np.max(validation_acc_baseline, axis=1)
validation_means_list.append(np.mean(validation_acc_baseline_max))
validation_stds_list.append(np.std(validation_acc_baseline_max))

test_acc_baseline = np.array(results_baseline["test_results"])
test_means_list.append(np.mean(test_acc_baseline))
test_stds_list.append(np.std(test_acc_baseline))

label = ['b', '16', '14', '12', '10', '8', '6', '4', '2', '1']
for study_type in ["dynamic_clamping"]:  # "static_clamping", 
    for bit_resolution in bit_resolution_list:
            # I evaluate study
            results = torch.load(f'./results/{study_type}/results_th{threshold}_{bit_resolution}_bit_resolution.pt')

            training_acc = np.array(results["training_results"])
            training_acc_max = np.max(training_acc, axis=1)
            training_means_list.append(np.mean(training_acc_max))
            training_stds_list.append(np.std(training_acc_max))

            validation_acc = np.array(results["validation_results"])
            validation_acc_max = np.max(validation_acc, axis=1)
            validation_means_list.append(np.mean(validation_acc_max))
            validation_stds_list.append(np.std(validation_acc_max))

            test_acc = np.array(results["test_results"])
            test_means_list.append(np.mean(test_acc))
            test_stds_list.append(np.std(test_acc))


            # print("STOP")
            # II evaluate weight distribution
            # for repetition in range(max_repetitions):
                # layers = torch.load(f'./model/best_model_th{threshold}_{bit_resolution}_bit_resolution.pt')  # w1, w2, v1
                # extract max and min values
                # weights_max_min = []
                # for layer in layers:
                #     weights_max_min.append([torch.max(layer).detach().cpu().numpy(), torch.min(layer).detach().cpu().numpy()])
                # weights_max_min = np.array(weights_max_min)
                # total_max = np.max(weights_max_min[:, 0])
                # total_min = np.min(weights_max_min[:, 1])
                # total_max_std = np.std(weights_max_min[:, 0])
                # total_min_std = np.std(weights_max_min[:, 1])
                # print("Absolute max: {:.3f} +- {:.3f}" .format(total_max, total_max_std))
                # print("Absolute min: {:.3f} +- {:.3f}" .format(total_min, total_min_std))

    # create plots
    fig, ax = plt.subplots()
    ax.errorbar(label, training_means_list, yerr=training_stds_list, label="Training")
    ax.errorbar(label, validation_means_list, yerr=validation_stds_list, label="Validation")
    ax.errorbar(label, test_means_list, yerr=test_stds_list, label="Test")
    ax.set_xlabel("Bit resolution")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs. bit resolution for threshold {threshold}")
    ax.legend()
    plt.savefig(f"./figures/{study_type}/accuracy_vs_bit_resolution_th{threshold}.png")
    plt.close()

    print("STOP")