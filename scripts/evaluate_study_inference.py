import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.data_manager import create_folder


def main():
    # set variables
    threshold = 2  # possible values are: 1, 2, 5, 10

    bit_resolution_list = [16, 15, 14, 13, 12, 11, 10, 9,
                           8, 7, 6, 5, 4, 3, 2, 1]  # possible bit resolutions
    max_repetitions = 5

    label = ['b', '16', '15', '14', '13', '12', '11',
             '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
    layer_names = ['w1', 'v1', 'w2']
    for study_type in ["dynamic_clamping", "static_clamping"]:
        # create out folder
        fig_path = f'./figures/inference/{study_type}'
        create_folder(fig_path)
        # set data path
        model_path = f'./model/inference/{study_type}'
        results_path = f'./results/inference/{study_type}'
        study_name = study_type.split('_')
        test_means_list = []
        test_stds_list = []

        results_baseline = torch.load(
            f'./results/inference/{study_type}/results_th{threshold}_baseline_bit_resolution.pt')
        layers_baseline = []
        for repetition in range(max_repetitions):
            layers_baseline.append(torch.load(
                f'./model/inference/{study_type}/best_model_th{threshold}_baseline_bit_resolution_run_{repetition+1}.pt'))  # w1, w2, v1

        training_acc_baseline = np.array(results_baseline["acc_per_split"])
        # training_acc_baseline_max = np.max(training_acc_baseline)
        test_means_list.append(np.mean(training_acc_baseline))
        test_stds_list.append(np.std(training_acc_baseline))

        for bit_resolution in bit_resolution_list:
            print(f"Start {study_name[0]} clamping with {bit_resolution}bit")
            # I evaluate study
            results = torch.load(
                f'{results_path}/results_th{threshold}_{bit_resolution}_bit_resolution.pt')

            test_acc = np.array(results["acc_per_split"])
            test_means_list.append(np.mean(test_acc))
            test_stds_list.append(np.std(test_acc))

            # II evaluate weight distribution
            for repetition in range(max_repetitions):
                # weights_max_min = []
                fig, ax = plt.subplots(3, 1)
                plt.suptitle(
                    f"Weight distribution for {study_name[0]} clamping, {bit_resolution}bit, run {repetition}")
                layers = torch.load(
                    f'{model_path}/best_model_th{threshold}_{bit_resolution}_bit_resolution_run_{repetition+1}.pt')  # w1, w2, v1
                for layer_count, layer in enumerate(layers):
                    # extract max and min values
                    # weights_max_min.append([torch.max(layer).detach().cpu(
                    # ).numpy(), torch.min(layer).detach().cpu().numpy()])
                    # check if layers have activity, else write no acitvity
                    if np.sum(np.isnan(layer.detach().cpu().numpy())) > 0:
                        print(np.sum(np.isnan(layer.detach().cpu().numpy())) > 0)
                        print(np.sum(np.isnan(layer.detach().cpu().numpy())))
                        print('No weights. Check')

                        ax[layer_count].set_title(layer_names[layer_count])
                        ax[layer_count].text(0.5, 0.5, 'No weights found')
                        ax[layer_count].set_ylabel("Frequency")
                    else:
                        # create histograms of weight distribution per layer
                        layer = layer.detach().cpu().numpy()
                        nb_bins = len(np.unique(layer))
                        if nb_bins > 1000:
                            nb_bins = 1000
                        print(len(np.unique(layer)))
                        ax[layer_count].set_title(layer_names[layer_count])
                        ax[layer_count].hist(
                            layer.flatten(), bins=nb_bins, label=layer_names[layer_count])
                        ax[layer_count].axvline(
                            x=np.mean(layer), color='r', linestyle='dashed')
                        ax[layer_count].axvline(x=np.median(layer), color='k',
                                                linestyle='dashdot')
                        ax[layer_count].legend(["_hist", f"mean: {np.mean(layer):.4f}",
                                                f"median: {np.median(layer):.4f}"], loc="upper left")
                        ax[layer_count].set_ylabel("Frequency")

                ax[layer_count].set_ylabel("Frequency")
                ax[layer_count].set_xlabel("Weight value")
                fig.align_ylabels()
                plt.tight_layout()
                plt.savefig(
                    f"{fig_path}/weight_distribution_th{threshold}_{bit_resolution}_bit_resolution_run_{repetition+1}_{study_type}.png")
                plt.close()

                # weights_max_min = np.array(weights_max_min)
                # total_max = np.max(weights_max_min[:, 0])
                # total_min = np.min(weights_max_min[:, 1])
                # total_max_std = np.std(weights_max_min[:, 0])
                # total_min_std = np.std(weights_max_min[:, 1])
                # print(
                #     "Absolute max: {:.3f} +- {:.3f}" .format(total_max, total_max_std))
                # print(
                #     "Absolute min: {:.3f} +- {:.3f}" .format(total_min, total_min_std))

        # create plots
        fig, ax = plt.subplots()
        ax.errorbar(label, test_means_list, yerr=test_stds_list, label="Test")
        ax.set_xlabel("Bit resolution")
        ax.set_ylabel("Accuracy")
        ax.set_title(
            f"Accuracy vs. bit resolution for {study_name[0]} clamping")
        ax.legend()
        plt.savefig(
            f"./figures/inference/accuracy_vs_bit_resolution_th{threshold}_{study_type}.png")
        plt.close()

        print(f"Finished {study_name[0]} clamping")


if __name__ == "__main__":
    main()
