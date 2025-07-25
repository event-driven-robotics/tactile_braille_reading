import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_performance(path, acc_train, acc_test, loss_train):
    """
    Plot training and test accuracy and training loss over epochs.

    Calculates mean and standard deviation across trials for training accuracy,
    test accuracy, and training loss. Highlights the best test-set trial and
    saves the resulting figure as a PDF.

    Args:
        path (str):
            File path prefix (without “.pdf”) where the figure will be saved.
        acc_train (array-like, shape (n_trials, n_epochs)):
            Training accuracy values for each epoch and each trial.
        acc_test (array-like, shape (n_trials, n_epochs)):
            Test accuracy values for each epoch and each trial.
        loss_train (array-like, shape (n_trials, n_epochs)):
            Training loss values for each epoch and each trial.

    Returns:
        None
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


def plot_confusion_matrix_seaborn(trues, preds, labels):
    """
    Compute, plot, and save a normalized confusion matrix as a heatmap.

    Args:
        trues (array-like): Ground‐truth class labels.
        preds (array-like): Predicted class labels.
        labels (list of str): Class names, used to label the rows and columns of the matrix.

    Returns:
        None
    """
    cm = confusion_matrix(trues, preds, normalize='true')
    plt.figure(figsize=(12, 9))
    sn.heatmap(cm,
               annot=True,
               fmt='.1g',
               cbar=False,
               xticklabels=labels,
               yticklabels=labels,
               cmap="YlGnBu",
               square=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=0)
    plt.savefig("./figures/confusion_matrix_eprop_def.pdf")
    plt.close()


def plot_confusion_matrix_sklearn(trues, preds, labels):
    """
    Compute, plot, and save a normalized confusion matrix using sklearn’s ConfusionMatrixDisplay.

    Args:
        trues (array-like): Ground‐truth class labels.
        preds (array-like): Predicted class labels.
        labels (list of str): Class names, used to label the rows and columns of the matrix.

    Returns:
        None
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        trues,
        preds,
        display_labels=labels,
        normalize='true',
        cmap="YlGnBu",
        values_format='.1g'
    )
    disp.figure_.set_size_inches(12, 9)
    plt.xticks(rotation=0)
    disp.figure_.savefig("./figures/confusion_matrix_eprop_def.pdf")
    plt.close(disp.figure_)


def plot_network_activity(spr_recs: list, layer_names: list, time_bin_size: int, figname='./figures'):
    """
    Create and save raster plots of spike activity for each network layer.

    For each layer, this function converts binary spike recordings into
    event times (scaled by time_bin_size) and produces a horizontal raster
    plot. All layer plots are stacked vertically and saved as a single PDF.

    Args:
        spr_recs (list of numpy.ndarray):
            List of spike recording arrays, one per layer. Each array has shape
            (timesteps, num_neurons), with 1 indicating a spike and 0 no spike.
        layer_names (list of str):
            Names of the network layers; used as subplot titles.
        time_bin_size (int):
            Duration of a single time bin (in milliseconds). Spike indices are
            multiplied by this value to convert to time.
        figname (str, optional):
            File path prefix (without extension) where the figure will be saved.
            The output file will be '<figname>.pdf'. Defaults to './figures'.

    Returns:
        None
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
                0.001*time_bin_size
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
        print(len(spikes_per_neuron))
        print(len(range(num_neurons)))
        ax.eventplot(spikes_per_neuron, orientation="horizontal",
                     lineoffsets=range(num_neurons), linewidth=0.3, colors="k")
        ax.set_ylabel("Neuron ID")
        ax.set_title(f"{name} activity")
    ax.set_xlabel("Time [sec]")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{figname}.pdf")
    plt.close(fig)
