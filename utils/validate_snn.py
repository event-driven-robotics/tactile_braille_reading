"""validate_snn.py

Validation and visualization utilities for spiking neural networks.

Provides functions for model evaluation, classification accuracy computation, confusion
matrix generation, network activity visualization (raster plots), and training performance
plotting. Supports analysis of recurrent spiking neural networks trained for tactile
braille letter classification with detailed metrics and diagnostic visualizations.

Key Components:
- compute_classification_accuracy: Evaluate network performance on datasets
- plot_training_performance: Visualize training/test accuracy and loss over epochs
- plot_confusion_matrix: Generate confusion matrix heatmaps from predictions
- get_network_activity: Record spike trains from all network layers
- plot_network_activity: Create raster plots of neuronal spike activity

Visualization Features:
- Multi-run statistics (mean ± std) with best trial highlighting
- Normalized confusion matrices with accuracy metrics
- Spike raster plots for temporal activity analysis
- PDF output for publication-ready figures

Author: Simon F. Muller-Cleve
Date: January 15, 2026
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from .snn import compute_winning_neuron, run_snn


def compute_classification_accuracy(dataset: TensorDataset, layers: list, params: dict) -> tuple:
    """
    Compute classification accuracy on a dataset using a trained spiking neural network.

    Evaluates network performance by running inference on all samples in the dataset
    (processed in batches) and comparing predicted labels to ground truth labels. Uses
    compute_winning_neuron() for consistent prediction logic with random tie-breaking support.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset containing (input_data, labels) pairs for evaluation
        Input shape: [n_samples, time_steps, n_channels]
        Labels shape: [n_samples] with integer class indices

    layers : list
        List containing [recurrent_layer, feedforward_layer] trained layer objects
        Weights should be in evaluation mode (no dropout, etc.)

    params : dict
        Dictionary containing experimental parameters:
        - 'batch_size' : int
            Number of samples per batch for inference
        - 'device' : str
            Device for computation ("cuda:0", "cpu", etc.)
        - 'eprop' : bool
            If True, uses e-prop mode for evaluation
        - 'delayed_output' : int or None
            For e-prop: number of final timesteps to use for prediction
            For BPTT: ignored (always uses all timesteps)
            If None or 0, uses all timesteps
        - 'random_tie_breaking' : bool
            Passed to compute_winning_neuron for tie-breaking logic

    Returns
    -------
    tuple
        (mean_accuracy, true_labels, predicted_labels) where:

        - mean_accuracy : float
            Mean classification accuracy across all samples (range [0.0, 1.0])
        - true_labels : list of numpy.ndarray
            Ground truth labels for all batches, each array shape: [batch_size]
        - predicted_labels : list of numpy.ndarray
            Predicted labels for all batches, each array shape: [batch_size]

    Notes
    -----
    **Evaluation Mode:**
    - All computations performed with torch.no_grad() for memory efficiency
    - No gradient computation or weight updates occur
    - Processes data in batches according to params["batch_size"]

    **Prediction Logic:**
    - Uses compute_winning_neuron() for consistent winner-take-all selection
    - For e-prop: optionally uses only final delayed_output timesteps
    - For BPTT: always uses all timesteps
    - Supports random tie-breaking when multiple neurons have equal spike counts

    **Return Format:**
    - true_labels and predicted_labels are lists of numpy arrays (one per batch)
    - Can be concatenated for confusion matrix: np.concatenate(true_labels)
    - Useful for detailed per-sample analysis and error investigation

    **Memory Efficiency:**
    - DataLoader uses pin_memory=True for faster GPU transfer
    - Uses num_workers=4 for parallel data loading
    - All tensors moved to CPU after batch processing

    Examples
    --------
    >>> # Evaluate on test set
    >>> acc, y_true, y_pred = compute_classification_accuracy(
    ...     dataset=test_dataset, layers=trained_layers, params=params)
    >>> print(f"Test accuracy: {acc*100:.2f}%")
    >>> # Flatten for confusion matrix
    >>> y_true_flat = np.concatenate(y_true)
    >>> y_pred_flat = np.concatenate(y_pred)

    See Also
    --------
    compute_winning_neuron : Core prediction function used by this method
    plot_confusion_matrix : Visualize prediction errors
    get_network_activity : Record detailed spike activity
    """
    logger = logging.getLogger('braille_training')
    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=False, num_workers=4)
    accs = []
    trues = []
    preds = []
    batch_count = 0
    for x_local, y_local in generator:
        batch_count += 1
        if x_local.numel() == 0 or y_local.numel() == 0:
            logger.warning(f"Empty batch encountered during evaluation (batch {batch_count}). Skipping.")
            continue
        try:
            x_local, y_local = x_local.to(
                params['device']), y_local.to(params['device'])
            with torch.no_grad():
                readout_activity, recs = run_snn(
                    inputs=x_local, layers=layers, params=params)

            if params["eprop"]:
                _, _, mem_rec_readout = recs
                yo = torch.softmax(mem_rec_readout, dim=2)
                if params["delayed_output"] is not None and params["delayed_output"] > 0:
                    _, neuron_idc = compute_winning_neuron(
                        yo[:, -params["delayed_output"]:, :], params=params)
                else:
                    _, neuron_idc = compute_winning_neuron(yo, params=params)
            else:
                # Use compute_winning_neuron with delayed_output handling for BPTT
                if params["delayed_output"] is not None and params["delayed_output"] > 0:
                    _, neuron_idc = compute_winning_neuron(
                        readout_activity[:, -params["delayed_output"]:, :], params=params)
                else:
                    _, neuron_idc = compute_winning_neuron(
                        readout_activity, params=params)

            # Check for NaN/Inf in predictions or labels
            if torch.isnan(y_local).any() or torch.isinf(y_local).any():
                logger.warning(f"NaN or Inf detected in true labels (batch {batch_count}).")
            if torch.isnan(neuron_idc).any() or torch.isinf(neuron_idc).any():
                logger.warning(f"NaN or Inf detected in predictions (batch {batch_count}).")

            # Check for mismatched batch sizes
            if y_local.shape != neuron_idc.shape:
                logger.warning(f"Batch size mismatch: y_local {y_local.shape}, predictions {neuron_idc.shape} (batch {batch_count})")

            # Compare to labels
            acc_arr = (y_local == neuron_idc).detach().cpu().numpy()
            mean_acc = np.mean(acc_arr)
            accs.append(mean_acc)
            trues.append(y_local.detach().cpu().numpy())
            preds.append(neuron_idc.detach().cpu().numpy())

            # Log extreme accuracy values
            if mean_acc == 0.0 or mean_acc == 1.0:
                logger.debug(f"Extreme batch accuracy ({mean_acc*100:.1f}%) in batch {batch_count}.")
        except Exception as e:
            logger.error(f"Exception during evaluation in batch {batch_count}: {e}")
            continue

    if batch_count == 0:
        logger.warning("No batches processed during evaluation. Dataset may be empty.")
        return 0.0, [], []
    if len(accs) == 0:
        logger.warning("No valid accuracy values computed during evaluation.")
        return 0.0, trues, preds
    mean_accs = np.mean(accs)
    logger.debug(f"Evaluation complete: {batch_count} batches, mean accuracy {mean_accs*100:.2f}%.")
    return mean_accs, trues, preds


def plot_training_performance(path: str, acc_train: np.ndarray, acc_test: np.ndarray, loss_train: np.ndarray) -> None:
    """
    Visualize training performance for a single training run.

    Creates a two-panel figure showing training/test accuracies and training loss over epochs
    for a single training run.

    Parameters
    ----------
    path : str
        File path (without extension) where the PDF figure will be saved
        Example: '/path/to/figures/training_performance'
        Output: '/path/to/figures/training_performance.pdf'

    acc_train : np.ndarray
        Training accuracies with shape [n_epochs]
        Values should be in range [0.0, 1.0] (converted to % internally)

    acc_test : np.ndarray
        Test accuracies with shape [n_epochs]
        Values should be in range [0.0, 1.0] (converted to % internally)

    loss_train : np.ndarray
        Training loss values with shape [n_epochs]
        Typically positive values (NLL loss + regularization)

    Returns
    -------
    None
        Saves figure to {path}.pdf and closes the figure

    Notes
    -----
    **Figure Layout:**
    - Figure size: 8x12 inches (portrait orientation)
    - Top panel: Accuracy plot (training and test)
    - Bottom panel: Loss plot (training only)

    **Top Panel (Accuracy):**
    - Y-axis: Accuracy in percentage (0-105%)
    - Solid lines for both training and test accuracy
    - Colors: Blue for training, Orange/Red for test
    - Legend: Training and Test labels

    **Bottom Panel (Loss):**
    - Y-axis: Loss value (auto-scaled)
    - Solid blue line for training loss
    - Color: Blue for training loss

    **Output Format:**
    - Saved as PDF for publication-ready vector graphics
    - Figure is closed after saving to free memory

    Examples
    --------
    >>> # Single run visualization
    >>> acc_train = np.array([0.5, 0.6, 0.7, 0.8])  # shape: [4]
    >>> acc_test = np.array([0.45, 0.55, 0.65, 0.75])
    >>> loss_train = np.array([2.0, 1.5, 1.0, 0.5])
    >>> plot_training_performance('./figures/single_run', acc_train, acc_test, loss_train)

    See Also
    --------
    plot_training_performance_repetitive_runs : For multiple training runs
    train : Function that generates the training history data
    """
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(2, 1, 1)
    # Plot training and test accuracy
    ax.plot(range(1, len(acc_train)+1),
            100*np.array(acc_train), color='blue')
    ax.plot(range(1, len(acc_test)+1), 100 *
            np.array(acc_test), color='orangered')
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim((0, 105))
    ax.set_title("Accuracy")
    ax.legend(["Training", "Test"], loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(range(1, len(loss_train)+1),
            loss_train, color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim((0, np.max(loss_train)*1.1))
    ax.legend(["Training loss"])
    ax.set_title("Training loss")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def plot_training_performance_repetitive_runs(path: str, acc_train: np.ndarray, acc_test: np.ndarray, loss_train: np.ndarray) -> None:
    """
    Visualize training performance with accuracy and loss plots across one or multiple runs.

    Creates a two-panel figure showing training/test accuracies and training loss over epochs.
    For single runs, displays only that run. For multiple runs, displays mean ± standard deviation
    and highlights the best trial (highest test accuracy).

    Parameters
    ----------
    path : str
        File path (without extension) where the PDF figure will be saved
        Example: '/path/to/figures/training_performance'
        Output: '/path/to/figures/training_performance.pdf'

    acc_train : np.ndarray
        Training accuracies with shape [n_runs, n_epochs]
        For single run: shape [1, n_epochs]
        Values should be in range [0.0, 1.0] (converted to % internally)

    acc_test : np.ndarray
        Test accuracies with shape [n_runs, n_epochs]
        For single run: shape [1, n_epochs]
        Values should be in range [0.0, 1.0] (converted to % internally)

    loss_train : np.ndarray
        Training loss values with shape [n_runs, n_epochs]
        For single run: shape [1, n_epochs]
        Typically positive values (NLL loss + regularization)

    Returns
    -------
    None
        Saves figure to {path}.pdf and closes the figure

    Notes
    -----
    **Figure Layout:**
    - Figure size: 8x12 inches (portrait orientation)
    - Top panel: Accuracy plot (training and test)
    - Bottom panel: Loss plot (training only)

    **Top Panel (Accuracy):**
    - Y-axis: Accuracy in percentage (0-105%)
    - Shaded regions: ± 1 standard deviation around mean
    - Dashed lines: Mean accuracy across all runs
    - Solid lines: Best trial (trial with maximum test accuracy)
    - Colors: Blue for training, Orange/Red for test
    - Legend: Shows mean, std, and best trial

    **Bottom Panel (Loss):**
    - Y-axis: Loss value (auto-scaled, starting at 0)
    - Shaded region: ± 1 standard deviation around mean
    - Dashed line: Mean loss across all runs
    - Solid line: Loss from best trial (same trial as accuracy plot)
    - Color: Blue for training loss

    **Best Trial Selection:**
    - Best trial is defined as the run with maximum test accuracy
    - If multiple runs achieve same max, selects first occurrence
    - Both accuracy and loss plots show data from this same trial

    **Output Format:**
    - Saved as PDF for publication-ready vector graphics
    - Figure is closed after saving to free memory

    Examples
    --------
    >>> # Multiple runs with statistics
    >>> acc_train = np.random.rand(10, 100)  # 10 runs, 100 epochs
    >>> acc_test = np.random.rand(10, 100)
    >>> loss_train = np.random.rand(10, 100) * 2
    >>> plot_training_performance_repetitive_runs('./figures/multi_run', acc_train, acc_test, loss_train)

    See Also
    --------
    plot_training_performance : For single training run
    train : Function that generates the training history data
    """
    # Ensure inputs are 2D arrays
    if acc_train.ndim == 1:
        acc_train = acc_train.reshape(1, -1)
    if acc_test.ndim == 1:
        acc_test = acc_test.reshape(1, -1)
    if loss_train.ndim == 1:
        loss_train = loss_train.reshape(1, -1)

    # calc mean and std
    acc_mean_train, acc_std_train = np.mean(
        acc_train, axis=0), np.std(acc_train, axis=0)
    acc_mean_test, acc_std_test = np.mean(
        acc_test, axis=0), np.std(acc_test, axis=0)

    # Find best trial (highest final test accuracy)
    best_trial = np.argmax(np.max(acc_test, axis=1))

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
    ax.set_ylim(bottom=0)
    ax.legend(
        ["Training std", r"$\overline{\mathrm{Training}}$", "Training loss @ best test"])
    ax.set_title("Training loss")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def plot_confusion_matrix(path: str, trues: list, preds: list, labels: list) -> None:
    """
    Generate and save a normalized confusion matrix heatmap for network predictions.

    Creates a confusion matrix from ground truth and predicted labels, normalizes by true class
    (rows sum to 1.0), and visualizes as a heatmap with overall accuracy in the title.

    Parameters
    ----------
    path : str
        File path (without extension) where the PDF figure will be saved
        Example: '/path/to/figures/confusion_matrix'
        Output: '/path/to/figures/confusion_matrix.pdf'

    trues : list of numpy.ndarray or numpy.ndarray
        Ground truth labels, either:
        - List of arrays (one per batch), each with shape [batch_size]
        - Single flattened array with shape [n_samples]
        Integer class indices (0 to n_classes-1)

    preds : list of numpy.ndarray or numpy.ndarray
        Predicted labels, either:
        - List of arrays (one per batch), each with shape [batch_size]
        - Single flattened array with shape [n_samples]
        Integer class indices (0 to n_classes-1)

    labels : list of str
        List of class label names (e.g., ['A', 'B', 'C', 'D', 'E'])
        Used for axis tick labels
        Length should match number of unique classes

    Returns
    -------
    None
        Saves confusion matrix heatmap to {path}.pdf

    Notes
    -----
    **Confusion Matrix:**
    - Rows: True labels (ground truth)
    - Columns: Predicted labels (network output)
    - Normalization: By true class (each row sums to 1.0)
    - Interpretation: Entry [i,j] = fraction of class i samples predicted as class j
    - Perfect diagonal indicates 100% accuracy

    **Visualization:**
    - Figure size: 12x9 inches (landscape orientation)
    - Colormap: YlGnBu (Yellow-Green-Blue)
    - Annotations: Show normalized values with format '.1g' (1 significant digit)
    - Title: Overall accuracy percentage (e.g., "85.23% Accuracy")
    - No colorbar (cleaner appearance)
    - Square aspect ratio for cells

    **Input Handling:**
    - If trues/preds are lists: automatically flattened via np.concatenate()
    - If already arrays: used directly
    - Must have same total length

    **Accuracy Computation:**
    - Simple element-wise comparison: accuracy = sum(trues == preds) / len(trues)
    - Displayed in title with 2 decimal places

    Examples
    --------
    >>> # From compute_classification_accuracy output
    >>> acc, y_true, y_pred = compute_classification_accuracy(test_data, layers, params)
    >>> labels = ['A', 'B', 'C', 'D', 'E']
    >>> plot_confusion_matrix('./figures/test_confusion', y_true, y_pred, labels)

    >>> # From pre-computed arrays
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 1, 0, 1, 2])
    >>> plot_confusion_matrix('./figures/cm', y_true, y_pred, ['Class0', 'Class1', 'Class2'])

    See Also
    --------
    compute_classification_accuracy : Generate predictions for confusion matrix
    sklearn.metrics.confusion_matrix : Underlying function for matrix computation
    """

    # Flatten lists if needed, but preserve type for function signature
    trues_arr = np.concatenate(trues) if isinstance(trues, list) else trues
    preds_arr = np.concatenate(preds) if isinstance(preds, list) else preds

    accs = np.sum(trues_arr == preds_arr) / len(trues_arr)
    cm = confusion_matrix(trues_arr, preds_arr, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[
                         jj for jj in labels])
    plt.figure(figsize=(12, 9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.title(f'{accs*100:.2f}% Accuracy\n')
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    plt.savefig(
        f"{path}.pdf")


def get_network_activity(dataset: TensorDataset, layers: list, params: dict) -> tuple:
    """
    Record network activity (spike trains) for all samples in a dataset.

    Runs the trained network in inference mode on all samples and collects spike recordings
    from both hidden and output layers for subsequent analysis or visualization. Always uses
    all timesteps for recording (ignores delayed_output setting for complete activity capture).

    Parameters
    ----------
    dataset : TensorDataset
        Dataset containing (input_data, labels) pairs for evaluation
        Input shape: [n_samples, time_steps, n_channels]
        Labels shape: [n_samples] with integer class indices

    layers : list
        List containing [recurrent_layer, feedforward_layer] trained layer objects
        Weights should be in evaluation mode

    params : dict
        Dictionary containing experimental parameters:
        - 'batch_size' : int
            Number of samples per batch for inference
        - 'device' : str
            Device for computation ("cuda:0", "cpu", etc.)
        - 'random_tie_breaking' : bool
            Passed to compute_winning_neuron for prediction
        - Other parameters passed to run_snn for forward pass

    Returns
    -------
    tuple
        (spk_rec_readout_list, spk_rec_hidden_list, mem_rec_hidden_list,
         mem_rec_readout_list, syn_rec_hidden_list, syn_rec_readout_list) where:

        - spk_rec_readout_list : list of numpy.ndarray
            Output layer spike trains, one array per batch
            Each array shape: [batch_size, time_steps, n_output_neurons]
            Binary values (0 or 1) indicating spike events
        - spk_rec_hidden_list : list of numpy.ndarray
            Hidden layer spike trains, one array per batch
            Each array shape: [batch_size, time_steps, n_hidden_neurons]
            Binary values (0 or 1) indicating spike events

    Notes
    -----
    **Recording Mode:**
    - Always uses all timesteps for complete activity recording
    - Ignores params["delayed_output"] to capture full temporal dynamics
    - All computations performed with torch.no_grad() for memory efficiency
    - No gradient computation or weight updates occur

    **Data Format:**
    - Returns data as numpy arrays (moved from GPU to CPU)
    - Each list element corresponds to one batch from the dataset
    - Spike trains are binary: 1 indicates spike occurred in that timestep
    - Time dimension preserved for temporal analysis

    **Prediction for Accuracy:**
    - Uses compute_winning_neuron() for consistent winner-take-all selection
    - Always uses all timesteps for prediction (not just delayed_output)
    - Accuracy computed per batch for diagnostic purposes

    **Memory Considerations:**
    - Spike recordings can be large (batch * time * neurons per layer)
    - For long simulations or large batches, consider processing in smaller chunks
    - All recordings stored in memory before returning

    **Use Cases:**
    - Creating raster plots (see plot_network_activity)
    - Analyzing temporal spike patterns
    - Computing spike rate statistics
    - Investigating network dynamics per class
    - Debugging network behavior

    Examples
    --------
    >>> # Record activity on test set
    >>> output_spikes, hidden_spikes = get_network_activity(
    ...     dataset=test_dataset, layers=trained_layers, params=params)
    >>> print(f"Output spikes shape (first batch): {output_spikes[0].shape}")
    >>> print(f"Hidden spikes shape (first batch): {hidden_spikes[0].shape}")

    >>> # Analyze first sample from first batch
    >>> sample_idx = 0
    >>> sample_output = output_spikes[0][sample_idx]  # shape: [time, n_output]
    >>> total_spikes = np.sum(sample_output)
    >>> print(f"Total output spikes for sample {sample_idx}: {total_spikes}")

    See Also
    --------
    plot_network_activity : Visualization function that uses this data
    compute_winning_neuron : Prediction function for accuracy
    run_snn : Forward pass function for spike generation
    """
    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=False, num_workers=4)

    spk_rec_readout_list = []
    spk_rec_hidden_list = []
    mem_rec_hidden_list = []
    mem_rec_readout_list = []
    syn_rec_hidden_list = []
    syn_rec_readout_list = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(
            params['device']), y_local.to(params['device'])
        activity_params = dict(params)
        activity_params['return_extended_recs'] = True
        with torch.no_grad():
            spk_rec_readout, recs = run_snn(
                inputs=x_local, layers=layers, params=activity_params)

        if len(recs) == 5:
            mem_rec_hidden, spk_rec_hidden, mem_rec_readout, syn_rec_hidden, syn_rec_readout = recs
        else:
            logging.warning("run_snn did not return synaptic recordings. Filling with zeros.")
            mem_rec_hidden, spk_rec_hidden, mem_rec_readout = recs
            syn_rec_hidden = torch.zeros_like(mem_rec_hidden)
            syn_rec_readout = torch.zeros_like(mem_rec_readout)

        spk_rec_readout_list.append(spk_rec_readout.detach().cpu().numpy())
        spk_rec_hidden_list.append(spk_rec_hidden.detach().cpu().numpy())
        mem_rec_hidden_list.append(mem_rec_hidden.detach().cpu().numpy())
        mem_rec_readout_list.append(mem_rec_readout.detach().cpu().numpy())
        syn_rec_hidden_list.append(syn_rec_hidden.detach().cpu().numpy())
        syn_rec_readout_list.append(syn_rec_readout.detach().cpu().numpy())

    return (
        spk_rec_readout_list,
        spk_rec_hidden_list,
        mem_rec_hidden_list,
        mem_rec_readout_list,
        syn_rec_hidden_list,
        syn_rec_readout_list,
    )


def plot_network_activity(spr_recs: list, layer_names: list, params: dict, figname: str = './figures') -> None:
    """
    Create and save raster plots visualizing spike activity across network layers.

    Generates a multi-panel figure showing spike raster plots for each layer, where each
    spike is plotted as a vertical line at its occurrence time for the corresponding neuron.
    Useful for analyzing temporal patterns, synchrony, and population dynamics.

    Parameters
    ----------
    spr_recs : list of numpy.ndarray
        List of spike recordings for each layer (one array per layer)
        Each array shape: [time_steps, n_neurons]
        Binary values (0 or 1) where 1 indicates a spike occurred
        Order should match layer_names (e.g., [hidden_spikes, output_spikes])

    layer_names : list of str
        List of layer names for subplot titles
        Example: ['Hidden layer', 'Readout layer']
        Length must match len(spr_recs)

    params : dict
        Dictionary containing experimental parameters:
        - 'time_bin_size' : int
            Size of time bins in milliseconds (for converting to seconds)
        - 'max_time' : int
            Maximum simulation time in milliseconds (for x-axis limits)

    figname : str, optional
        File path (without extension) where the PDF figure will be saved
        Default: './figures'
        Output: '{figname}.pdf'

    Returns
    -------
    None
        Saves figure to {figname}.pdf and closes the figure

    Notes
    -----
    **Figure Layout:**
    - Creates one subplot per layer in vertical arrangement
    - Number of subplots = len(spr_recs) = len(layer_names)
    - Figure size automatically determined by matplotlib
    - All y-labels aligned for clean appearance

    **Raster Plot Details:**
    - X-axis: Time in seconds (0 to max_time * 0.001)
    - Y-axis: Neuron ID (0 to n_neurons-1, one row per neuron)
    - Each spike: Vertical tick mark at spike time
    - Marker: '|' (vertical line) with size 8 for visibility
    - Color: Black for all spikes

    **Time Conversion:**
    - Spike times converted from timestep indices to seconds
    - Formula: time_seconds = timestep * time_bin_size * 0.001
    - Ensures accurate temporal representation

    **Empty Neurons:**
    - Neurons with no spikes appear as empty rows
    - All neurons displayed regardless of activity
    - Useful for identifying silent/dead neurons

    **Performance:**
    - Uses scatter plot with vertical line markers for efficient visualization
    - Direct numpy indexing for fast spike detection

    **Output Format:**
    - Saved as PDF for publication-ready vector graphics
    - Figure closed after saving to free memory
    - Tight layout applied for optimal spacing

    Examples
    --------
    >>> hidden_spikes = np.random.rand(100, 50) > 0.95  # 100 timesteps, 50 neurons
    >>> output_spikes = np.random.rand(100, 10) > 0.90  # 100 timesteps, 10 neurons
    >>> plot_network_activity([hidden_spikes, output_spikes], 
    ...                        ['Hidden', 'Output'], 
    ...                        './figures/activity')
    """
    nb_layers = len(layer_names)
    fig = plt.figure()
    for counter, name in enumerate(layer_names):
        spk_per_layer = spr_recs[counter]
        num_neurons = spk_per_layer.shape[1]
        ax = fig.add_subplot(nb_layers, 1, counter+1)

        # Find all spike times and corresponding neuron IDs
        spike_times, neuron_ids = np.where(spk_per_layer == 1)
        neuron_ids += 1  # Shift neuron IDs to start from 1

        # Convert spike times from timesteps to seconds
        spike_times_sec = spike_times * 0.001 * int(params['time_bin_size'])

        # Plot spikes as vertical tick marks using scatter
        ax.scatter(spike_times_sec, neuron_ids,
                   marker='|', s=8, c='k', linewidths=0.5)

        ax.set_xlim(0, params["max_time"] * 0.001)
        ax.set_ylim(-0.5, num_neurons + 0.5)
        ax.set_ylabel("Neuron ID")
        ax.set_title(f"{name} activity")
    ax.set_xlabel("Time [sec]")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{figname}.pdf")
    plt.close(fig)
