"""train_snn.py

Training procedures for spiking neural networks with e-prop and BPTT algorithms.

Implements network construction, training loops, weight initialization, and optimization
for recurrent spiking neural networks. Supports both e-prop (eligibility propagation) and
BPTT (backpropagation through time) learning rules with configurable neuron dynamics,
regularization, weight quantization, and training hyperparameters for tactile braille 
letter classification.

Key Components:
- build_and_train: High-level network construction and training orchestration
- train: Main training loop with batch processing and optimization
- grads_batch: E-prop gradient computation using eligibility traces
- copy_layers: Deep copy utility for saving best model weights

Training Features:
- Supports e-prop (online, memory-efficient) and BPTT (standard backprop)
- Adamax optimizer with configurable learning rates
- Optional weight quantization via straight-through estimator
- L1 and L2 spike regularization for controlling network activity
- Stratified train/test splits with progress monitoring
- Best model tracking based on test accuracy
- Detailed debug output for network diagnostics

Author: Simon F. Muller-Cleve
Date: January 13, 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .neuron_models import feedforward_layer, recurrent_layer, ste_fn
from .snn import compute_winning_neuron, run_snn
from .validate_snn import compute_classification_accuracy


def build_and_train(params: dict, ds_train: TensorDataset, ds_test: TensorDataset) -> tuple:
    """
    Build and train a recurrent spiking neural network for braille letter classification.

    This function constructs a two-layer spiking neural network (recurrent hidden layer and
    feedforward readout layer), initializes weights, configures neuron dynamics, and trains
    the network using either e-prop or BPTT. It tracks performance metrics and returns the
    best performing model along with training history.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters including:

        Network Architecture:
        - 'nb_input_copies' : int
            Number of times to replicate each input channel
        - 'nb_hidden' : int
            Number of neurons in the recurrent hidden layer
        - 'letters' : list of str
            List of output class labels (determines output layer size)
        - 'selected_channels' : list of int or None
            Taxel indices to use (determines input size: 2 * len(selected_channels))

        Neuron Dynamics:
        - 'tau_mem' : float
            Membrane time constant (seconds) for feedforward layers
        - 'tau_mem_rec' : float
            Membrane time constant (seconds) for recurrent layer
        - 'tau_ratio' : float
            Ratio of tau_mem to tau_syn (synaptic time constant)
        - 'time_step' : float
            Simulation timestep (seconds)
        - 'no_synapse' : bool
            If True, disables synaptic dynamics (alpha=0)
        - 'use_linear_decay' : bool
            If True, uses linear membrane decay; if False, uses exponential
        - 'ref_per_timesteps' : int or None
            Refractory period duration in timesteps
        - 'lower_bound' : float or None
            Minimum membrane potential (clamping threshold)

        E-prop Specific:
        - 'use_eprop' : bool
            If True, uses e-prop; if False, uses BPTT
        - 'tau_trace' : float
            Eligibility trace time constant for hidden layer (seconds)
        - 'tau_trace_out' : float
            Eligibility trace time constant for output layer (seconds)

        Weight Initialization:
        - 'fwd_weight_scale' : float
            Scale factor for feedforward weight initialization
        - 'weight_scale_factor' : float
            Multiplier for recurrent weights (rec_scale = fwd_scale * factor)

        Training:
        - 'batch_size' : int
            Number of samples per training batch
        - 'epochs' : int
            Number of training epochs
        - 'learning_rate' : float
            Initial learning rate for Adamax optimizer
        - 'use_weight_quantization' : bool
            If True, applies weight quantization via STE
        - 'possible_weights' : torch.Tensor
            Discrete weight values for quantization

        Regularization:
        - 'reg_spikes' : float
            L1 regularization coefficient for total spike count
        - 'reg_neurons' : float
            L2 regularization coefficient for per-neuron spike activity

        Other:
        - 'device' : str
            Device for computation ("cuda:0", "cpu", etc.)
        - 'dtype_torch' : torch.dtype
            Data type for tensors
        - 'debug' : bool
            If True, enables detailed diagnostic output

    ds_train : TensorDataset
        Training dataset containing (input_data, labels) pairs
        Input shape: [n_samples, time_steps, n_channels]

    ds_test : TensorDataset
        Test dataset containing (input_data, labels) pairs
        Input shape: [n_samples, time_steps, n_channels]

    Returns
    -------
    tuple
        (loss_hist_epochs, accs_hist_epochs, best_layers, vars_eprop) where:

        - loss_hist_epochs : list of float
            Training loss values per epoch (length: epochs)
        - accs_hist_epochs : list of lists
            [[train_accuracies], [test_accuracies]] per epoch
            Each sublist has length: epochs
            Accuracies are floats in range [0.0, 1.0]
        - best_layers : list
            [recurrent_layer, feedforward_layer] objects with weights from 
            the epoch with highest test accuracy
        - vars_eprop : list
            [beta_trace, beta_trace_out] decay factors for eligibility traces
            Values are float or None (None if use_eprop=False)

    Notes
    -----
    **Network Architecture:**
    - Input layer: 2 * len(selected_channels) * nb_input_copies channels
    - Hidden layer: nb_hidden recurrent LIF neurons
    - Output layer: len(letters) feedforward LIF neurons

    **Weight Initialization:**
    - Feedforward weights: N(0, fwd_weight_scale / sqrt(n_inputs))
    - Recurrent weights: N(0, rec_weight_scale / sqrt(n_neurons))
    - rec_weight_scale = fwd_weight_scale * weight_scale_factor

    **Neuron Dynamics:**
    - Exponential decay: beta = exp(-dt/tau_mem)
    - Linear decay: beta = 0.005 (fixed decay rate)
    - Synaptic dynamics: alpha = exp(-dt/tau_syn) or 0 if no_synapse=True

    **Training Process:**
    - Uses train() function for the main training loop
    - Tracks best model based on test accuracy (not training accuracy)
    - Returns weights from best epoch, not final epoch

    **Performance Tracking:**
    - Computes best training accuracy and corresponding test accuracy
    - Computes best test accuracy and corresponding training accuracy
    - Identifies epochs where best performance occurred

    See Also
    --------
    train : Main training loop implementation
    recurrent_layer : Recurrent spiking layer class
    feedforward_layer : Feedforward spiking layer class
    """

    # Num of spiking neurons used to encode each channel
    nb_input_copies = params['nb_input_copies']
    # print("Number of input copies ", nb_input_copies)
    # Network parameters
    nb_inputs = 2*len(params["selected_channels"])*nb_input_copies
    nb_outputs = len(params["letters"])
    nb_hidden = params['nb_hidden']
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

    if params["use_eprop"]:
        beta_trace = float(np.exp(-params["time_step"]/params['tau_trace']))
        beta_trace_out = float(
            np.exp(-params["time_step"]/params['tau_trace_out']))
        vars_eprop = [beta_trace, beta_trace_out]
    else:
        vars_eprop = [None, None]

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = fwd_weight_scale*params['weight_scale_factor']

    # Spiking network
    # recurrent layer
    rec_layer = recurrent_layer(batch_size=params["batch_size"], nb_inputs=nb_inputs, nb_neurons=nb_hidden, fwd_weight_scale=fwd_weight_scale, rec_weight_scale=rec_weight_scale, alpha=alpha,
                                beta=beta_rec, use_eprop=params["use_eprop"], use_linear_decay=params["use_linear_decay"], device=params["device"], dtype=params["dtype_torch"], ref_per=params["ref_per_timesteps"], gamma=params["gamma"])

    # readout layer
    ff_layer = feedforward_layer(batch_size=params["batch_size"], nb_inputs=nb_hidden, nb_neurons=nb_outputs, fwd_weight_scale=fwd_weight_scale, alpha=alpha, beta=beta_rec,
                                 use_eprop=params["use_eprop"], use_linear_decay=params["use_linear_decay"], device=params["device"], dtype=params["dtype_torch"], ref_per=params["ref_per_timesteps"], gamma=params["gamma"])

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
    loss_hist_epochs, accs_hist_epochs, best_layers = train(
        params=params, dataset=ds_train, layers=layers, vars_eprop=vars_eprop, dataset_test=ds_test)

    # best training and test at best training
    acc_best_train = np.max(accs_hist_epochs[0])  # returns max value
    acc_best_train = acc_best_train*100

    # returns index of max value
    idx_best_train = np.argmax(accs_hist_epochs[0])
    acc_test_at_best_train = accs_hist_epochs[1][idx_best_train]*100

    # best test and training at best test
    acc_best_test = np.max(accs_hist_epochs[1])
    acc_best_test = acc_best_test*100
    idx_best_test = np.argmax(accs_hist_epochs[1])
    acc_train_at_best_test = accs_hist_epochs[0][idx_best_test]*100

    # TODO track time constants!!!
    # print("Final results: ")
    # print("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(
    #     acc_best_train, acc_test_at_best_train, idx_best_train+1))
    # print("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(
    #     acc_best_test, acc_train_at_best_test, idx_best_test+1))
    # print("------------------------------------------------------------------------------------\n")

    return loss_hist_epochs, accs_hist_epochs, best_layers, vars_eprop


def train(params: dict, dataset: TensorDataset, layers: list, vars_eprop: list, dataset_test=None) -> tuple:
    """
    Train a spiking neural network and evaluate on test data.

    Implements the main training loop for a spiking neural network using either e-prop or BPTT.
    Processes data in batches, computes gradients (via eligibility traces for e-prop or standard
    backprop for BPTT), updates weights via Adamax optimizer, and tracks training/test performance
    over epochs. Supports optional weight quantization, spike regularization, and detailed debugging.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters including:

        Training Configuration:
        - 'batch_size' : int
            Number of samples per batch
        - 'epochs' : int
            Number of training epochs
        - 'learning_rate' : float
            Learning rate for Adamax optimizer

        Learning Algorithm:
        - 'use_eprop' : bool
            If True, uses e-prop (eligibility propagation);
            If False, uses BPTT (backpropagation through time)
        - 'delayed_output' : int or None
            For e-prop: number of final timesteps to use for gradient computation
            For BPTT: ignored (uses all timesteps)
            If None or 0, uses all timesteps for both algorithms
        - 'gamma' : float
            Surrogate gradient scale factor (typically 15.0)

        Regularization:
        - 'reg_spikes' : float
            L1 regularization coefficient for total spike count
            Applied to hidden layer only: reg_spikes * mean(sum(spikes_hidden))
        - 'reg_neurons' : float
            L2 regularization coefficient for per-neuron spike activity
            Applied to hidden layer: reg_neurons * mean(sum(sum(spikes_hidden, dim=time), dim=batch)^2)

        Weight Quantization:
        - 'use_weight_quantization' : bool
            If True, applies straight-through estimator (STE) after each forward pass
        - 'possible_weights' : torch.Tensor
            Discrete weight values for quantization (e.g., [-1, 0, 1])

        Network Parameters:
        - 'letters' : list of str
            Output class labels (determines number of output neurons)
        - 'data_steps' : int
            Number of simulation timesteps
        - 'device' : str
            Device for computation ("cuda:0", "cpu", etc.)
        - 'dtype_torch' : torch.dtype
            Data type for tensors

        Debugging:
        - 'debug' : bool
            If True, prints detailed diagnostics for first batch of first epoch
        - 'use_random_tie_breaking' : bool
            Passed to compute_winning_neuron for prediction logic

    dataset : TensorDataset
        Training dataset containing (input_data, labels) pairs
        Input shape: [n_samples, time_steps, n_channels]
        Labels shape: [n_samples] with integer class indices

    layers : list
        List containing [recurrent_layer, feedforward_layer] layer objects
        These objects are modified in-place during training

    vars_eprop : list
        List containing [beta_trace, beta_trace_out] decay factors for e-prop
        - beta_trace : float
            Eligibility trace decay for hidden layer (exp(-dt/tau_trace))
        - beta_trace_out : float  
            Eligibility trace decay for output layer (exp(-dt/tau_trace_out))
        - Both are None if use_eprop=False

    dataset_test : TensorDataset, optional
        Test dataset for evaluation after each epoch (default: None)
        If None, only training accuracy is tracked

    Returns
    -------
    tuple
        (loss_hist_epochs, accs_hist_epochs, best_acc_layers) where:

        - loss_hist_epochs : list of float
            Training loss values per epoch (length: epochs)
            Loss includes NLL loss + regularization (for BPTT only)
        - accs_hist_epochs : list of lists
            [[train_accuracies], [test_accuracies]] per epoch
            train_accuracies: list of float, length: epochs (range [0.0, 1.0])
            test_accuracies: list of float, length: epochs (range [0.0, 1.0])
            If dataset_test is None, test_accuracies will be empty lists
        - best_acc_layers : list
            [recurrent_layer, feedforward_layer] copied layer objects
            Contains weights from the epoch with highest test accuracy
            If dataset_test is None, returns layers from last epoch

    Notes
    -----
    **Training Algorithm:**

    For BPTT:
    - Uses negative log likelihood (NLL) loss: -log P(y_true | network_output)
    - Spike counts summed over time, then log-softmax applied
    - Adds L1 regularization on total spikes: reg_spikes * mean(sum(spikes))
    - Adds L2 regularization on per-neuron spikes: reg_neurons * mean((sum_per_neuron)^2)
    - Standard backpropagation via loss.backward()

    For E-prop:
    - Uses eligibility traces to compute gradients online
    - Calls grads_batch() to compute weight gradients manually
    - Error signal: (predicted_output - target) at each timestep
    - Eligibility traces filtered with exponential kernels (beta_trace, beta_trace_out)
    - Optionally uses only final delayed_output timesteps for gradient computation
    - No explicit loss function; gradients computed directly

    **Optimization:**
    - Adamax optimizer with betas=(0.9, 0.995)
    - Learning rate remains constant (no scheduling in this version)
    - Gradients zeroed at start of each batch
    - Weight quantization applied after forward pass but before backward pass

    **Weight Quantization:**
    - Applied to all weight matrices (ff_weights, rec_weights in both layers)
    - Uses straight-through estimator: forward uses quantized weights, backward uses continuous gradients
    - Quantization maps each weight to nearest value in possible_weights tensor

    **Performance Tracking:**
    - Training accuracy computed on full training set after each batch
    - Test accuracy computed on full test set after each epoch
    - Best model saved based on test accuracy (not training accuracy)
    - Progress displayed via nested tqdm bars (epochs and batches)

    **Prediction Logic:**
    - Uses compute_winning_neuron() for consistent winner-take-all selection
    - Sums spikes over time, selects neuron with highest count
    - Supports random tie-breaking if use_random_tie_breaking=True

    **Debug Output (first batch of first epoch only):**
    - True labels vs predicted labels (first 10 samples)
    - Spike counts and label distributions
    - Input data statistics (shape, spike counts, spike rate)
    - Hidden layer statistics (spike distributions, min/max counts)
    - Output layer statistics (weight distributions, input currents)
    - Training accuracy before and after weight update

    **Memory Management:**
    - Large intermediate tensors commented out for deletion (currently disabled)
    - Uses DataLoader with pin_memory=True and num_workers=4 for efficiency
    - GPU memory usage scales with batch_size and data_steps

    See Also
    --------
    grads_batch : E-prop gradient computation function
    compute_winning_neuron : Prediction function with tie-breaking
    compute_classification_accuracy : Accuracy computation on full dataset
    copy_layers : Function to save best model weights
    run_snn : Forward pass through the network
    """

    weights = [layers[0].ff_weights,
               layers[0].rec_weights, layers[1].ff_weights]
    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    generator = DataLoader(dataset=dataset, batch_size=params["batch_size"], pin_memory=True,
                           shuffle=True, num_workers=4)

    # The optimization loop
    loss_hist_epochs = []
    accs_hist_epochs = [[], []]
    best_test_acc_per_epoch = 0.0
    best_acc_layers = []

    pbar_training = tqdm(
        range(params['epochs']), position=1, total=params['epochs'], leave=False)
    count_epoch = 0
    for _ in pbar_training:
        # learning rate decreases over epochs
        optimizer = torch.optim.Adamax(
            weights, lr=params["learning_rate"], betas=(0.9, 0.995))
        # if e > nb_epochs/2:
        #     lr = lr * 0.9
        loss_hist_batches = []
        # accs: mean training accuracies for each batch
        accs_hist_batches = []
        pbar_batches = tqdm(generator, position=2,
                            total=len(generator), leave=False)
        for x_local, y_local in pbar_batches:
            x_local, y_local = x_local.to(
                params['device']), y_local.to(params['device'])

            optimizer.zero_grad()

            spk_rec_readout, recs = run_snn(
                inputs=x_local, layers=layers, params=params)
            # weight quantization - apply directly to layer weights
            if params["use_weight_quantization"]:
                layers[0].ff_weights.data = ste_fn(
                    layers[0].ff_weights, params['possible_weights']).to(params['dtype_torch'])
                layers[0].rec_weights.data = ste_fn(
                    layers[0].rec_weights, params['possible_weights']).to(params['dtype_torch'])
                layers[1].ff_weights.data = ste_fn(
                    layers[1].ff_weights, params['possible_weights']).to(params['dtype_torch'])

            # average_spike_output[count_epoch] = torch.mean(
            #     torch.sum(spk_rec_readout, 1))
            # average_spike_recurrent[count_epoch] = torch.mean(
            #     torch.sum(recs[1], 1))

            _, spk_rec_hidden, _ = recs

            # Use all timesteps for BPTT, optionally use delayed_output window for e-prop
            if params["use_eprop"] and params["delayed_output"] is not None and params["delayed_output"] > 0:
                summed_spikes, neuron_idc = compute_winning_neuron(
                    spk_rec_readout[:, -params["delayed_output"]:, :], params=params)
            else:
                summed_spikes, neuron_idc = compute_winning_neuron(
                    spk_rec_readout=spk_rec_readout, params=params)

            # Compute gradients based on selected learning algorithm
            if params["use_eprop"]:
                one_hot_encoded = torch.nn.functional.one_hot(
                    y_local, num_classes=len(params['letters']))
                # E-prop: manual gradient computation via eligibility traces
                # Create one-hot encoded predictions from selected winners
                yo = torch.nn.functional.one_hot(
                    neuron_idc, num_classes=len(params['letters']))
                # Expand to match temporal dimension for e-prop
                yo = yo.unsqueeze(1).expand(-1, spk_rec_readout.shape[1], -1)

                # Compute e-prop gradients
                mem_rec_hidden, spk_rec_hidden, _ = recs
                grads_batch(x_local.permute(1, 0, 2), yo.permute(1, 0, 2), one_hot_encoded,
                            params["gamma"], 1, mem_rec_hidden.permute(
                                1, 0, 2),
                            spk_rec_hidden.permute(
                                1, 0, 2), layers[0].ff_weights,
                            layers[0].rec_weights, layers[1].ff_weights,
                            vars_eprop[0], vars_eprop[1])
            else:
                log_p_y = log_softmax_fn(summed_spikes)

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
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                # BPTT: standard backpropagation
                loss_val.backward()

            optimizer.step()

            loss_hist_batches.append(loss_val.item())

            # Debug: Print first batch of first epoch to check predictions
            if params['debug'] and count_epoch == 0:
                tmp = np.mean((y_local == neuron_idc).detach().cpu().numpy())
                print(f"\nDebug - First batch:")
                print(
                    f"  True labels (y_local): {y_local[:min(10, len(y_local))].cpu().numpy()}")
                print(
                    f"  Predictions (neuron_idc): {neuron_idc[:min(10, len(y_local))].cpu().numpy()}")
                print(
                    f"  Spike counts (summed_spikes): {summed_spikes[:min(10, len(y_local))].detach().cpu().numpy()}")
                print(
                    f"  Label distribution: 0={torch.sum(y_local==0).item()}, 1={torch.sum(y_local==1).item()}")
                print(
                    f"  Prediction distribution: 0={torch.sum(neuron_idc==0).item()}, 1={torch.sum(neuron_idc==1).item()}")
                print(f"  Batch accuracy: {tmp:.4f}")

                # Check input data
                print(f"\n  Input data statistics:")
                print(f"    Shape: {x_local.shape}")
                print(
                    f"    Total input spikes (first 10 samples): {torch.sum(x_local[:min(10, len(y_local))], dim=(0,1)).cpu().numpy()}")
                print(
                    f"    Input spike rate (mean): {x_local.mean().item():.6f}")

                # Check hidden layer
                print(f"\n  Hidden layer statistics:")
                # sum over time: [batch, neurons]
                hidden_spike_counts = torch.sum(spk_rec_hidden, dim=0)
                print(
                    f"    Spike count distribution (mean across batch): {hidden_spike_counts.mean(dim=0).cpu().detach().numpy()}")
                print(
                    f"    Spike count (min, max): ({hidden_spike_counts.min().item()}, {hidden_spike_counts.max().item()})")
                print(
                    f"    First 10 samples total spikes: {torch.sum(spk_rec_hidden[:min(10, len(y_local))], dim=(0,1)).cpu().detach().numpy()}")

                # Check output layer
                print(f"\n  Output layer statistics:")
                print(f"    Weights shape: {layers[1].ff_weights.shape}")
                print(
                    f"    Weights mean per neuron: {torch.mean(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
                print(
                    f"    Weights std per neuron: {torch.std(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
                h2_sample = torch.einsum(
                    "abc,cd->abd", (spk_rec_hidden[:min(10, len(y_local))], layers[1].ff_weights.t()))
                print(
                    f"    Input current (h2) mean per output neuron: {h2_sample.mean(dim=(0,1)).detach().cpu().numpy()}")
                print(
                    f"    Number of positive weights: neuron0={torch.sum(layers[1].ff_weights[0] > 0).item()}, neuron1={torch.sum(layers[1].ff_weights[1] > 0).item()}")
                print()

            # Calculate train accuracy in each batch
            train_acc_per_batch, _, _ = compute_classification_accuracy(
                dataset=dataset, layers=layers, params=params)
            accs_hist_batches.append(train_acc_per_batch)

            if params['debug'] and count_epoch == 0:
                print(
                    f"Train acc after weight update: {train_acc_per_batch*100:.2f}%")

            # Update batch progress bar with current and running average accuracy
            current_acc = train_acc_per_batch * 100
            running_avg_acc = np.mean(accs_hist_batches) * 100
            pbar_batches.set_description(
                f"Batch acc: {current_acc:.1f}%, Running avg: {running_avg_acc:.1f}%")

            # Free up memory by deleting large intermediate tensors
            # del spk_rec_readout, recs, spk_rec_hidden, m, am
            # if not params["use_eprop"]:
            # del log_p_y, reg_loss
        mean_loss_per_epoch = np.mean(loss_hist_batches)
        loss_hist_epochs.append(mean_loss_per_epoch)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs_per_epoch = np.mean(accs_hist_batches)
        accs_hist_epochs[0].append(mean_accs_per_epoch)

        count_epoch = count_epoch + 1

        # Calculate test accuracy in each epoch
        test_acc_per_epoch, _, _ = compute_classification_accuracy(
            dataset=dataset_test, layers=layers, params=params)
        accs_hist_epochs[1].append(test_acc_per_epoch)
        if np.max(test_acc_per_epoch) >= best_test_acc_per_epoch:
            best_test_acc_per_epoch = np.max(test_acc_per_epoch)
            # Save copies of the layer objects using our custom copy function
            best_acc_layers = copy_layers(layers)

        pbar_training.set_description("Train {:.2f}%, Test {:.2f}%".format(
            accs_hist_epochs[0][-1]*100, accs_hist_epochs[1][-1]*100))
        # print("Train acc: ", accs_hist_epochs[0][-1]*100, "Test acc",
        #       accs_hist_epochs[1][-1]*100, 'Loss: ', loss_hist_epochs[-1])

    return loss_hist_epochs, accs_hist_epochs, best_acc_layers


def grads_batch(x: torch.Tensor, yo: torch.Tensor, yt: torch.Tensor, gamma: float, thr: int, v: torch.Tensor, z: torch.Tensor, w_in: torch.Tensor, w_rec: torch.Tensor, w_out: torch.Tensor, beta_trace: float, beta_trace_out: float, params: dict) -> None:
    """
    Compute weight gradients using e-prop (eligibility propagation) for spiking neural networks.

    This function implements the e-prop learning algorithm for recurrent spiking neural networks.
    It computes eligibility traces for input, recurrent, and output connections using efficient
    1D convolutions, then combines these traces with the error signal to calculate weight gradients.
    Gradients are accumulated directly into the .grad attributes of the weight tensors for subsequent
    optimization steps.

    Parameters
    ----------
    x : torch.Tensor
        Input spike trains with shape [time, batch, input_features]
        Binary spikes (0 or 1) representing input activity over time

    yo : torch.Tensor
        Network output (predicted labels, one-hot encoded) with shape [time, batch, output_units]
        Predicted class probabilities or spike counts processed through one-hot encoding

    yt : torch.Tensor
        Target labels (one-hot encoded) with shape [batch, output_units]
        Ground truth class labels in one-hot format

    gamma : float
        Surrogate gradient scaling factor for the spike function derivative approximation
        Typical value: 15.0 (controls steepness of surrogate gradient)

    thr : int
        Firing threshold for neurons (typically 1.0)
        Neurons spike when membrane potential exceeds this threshold

    v : torch.Tensor
        Membrane potential traces of hidden neurons with shape [time, batch, hidden_neurons]
        Continuous voltage values before spike thresholding

    z : torch.Tensor
        Spike trains of hidden neurons with shape [time, batch, hidden_neurons]
        Binary spikes (0 or 1) representing hidden layer activity

    w_in : torch.Tensor
        Input-to-hidden weight matrix with shape [hidden_neurons, input_features]
        Gradients are accumulated in w_in.grad (initialized to zeros if None)

    w_rec : torch.Tensor
        Recurrent (hidden-to-hidden) weight matrix with shape [hidden_neurons, hidden_neurons]
        Gradients are accumulated in w_rec.grad (initialized to zeros if None)

    w_out : torch.Tensor
        Hidden-to-output weight matrix with shape [output_units, hidden_neurons]
        Gradients are accumulated in w_out.grad (initialized to zeros if None)

    beta_trace : float
        Decay factor for eligibility traces of hidden layer connections
        Computed as exp(-dt/tau_trace) where dt is timestep and tau_trace is trace time constant
        Controls how long past spike events influence current weight updates

    beta_trace_out : float
        Decay factor for output eligibility traces
        Computed as exp(-dt/tau_trace_out) for output layer
        Typically equal to or larger than beta_trace

    params : dict
        Dictionary containing experimental parameters:
        - 'data_steps' : int
            Number of simulation timesteps (length of time dimension)
        - 'device' : str
            Device for computation ("cuda:0", "cpu", etc.)
        - 'dtype_torch' : torch.dtype
            Data type for tensors (float32, float64, etc.)
        - 'nb_hidden' : int
            Number of hidden neurons
        - 'use_eprop' : bool
            Should be True when calling this function
        - 'delayed_output' : int or None
            If set, only uses final delayed_output timesteps for gradient computation
            If None or 0, uses all timesteps

    Returns
    -------
    None
        This function modifies weight gradients in-place via:
        - w_in.grad : Gradients for input-to-hidden weights
        - w_rec.grad : Gradients for recurrent weights  
        - w_out.grad : Gradients for hidden-to-output weights

    Notes
    -----
    **E-prop Algorithm Overview:**

    E-prop computes gradients using eligibility traces that track the causal relationship
    between weight changes and neuron spikes. This allows for online, memory-efficient learning
    without storing full network activations over time (required for BPTT).

    **Eligibility Trace Computation:**

    1. Surrogate gradient: h = gamma * max(0, 1 - |v - thr| / thr)
       - Approximates derivative of spike function (which is zero almost everywhere)
       - Provides smooth gradients for optimization

    2. Input traces: trace_in[b,r,i,t] tracks influence of input i on hidden neuron r at time t
       - Filtered by beta_trace (exponential decay over time)
       - Modulated by surrogate gradient h

    3. Recurrent traces: trace_rec[b,r,j,t] tracks influence of hidden neuron j on neuron r
       - Captures recurrent dependencies in the network
       - Also filtered by beta_trace

    4. Output traces: trace_out[b,r,t] tracks influence of hidden neuron r on output
       - Filtered by beta_trace_out (typically slower decay)

    **Convolution-Based Implementation:**

    - Uses F.conv1d for efficient eligibility trace computation
    - Creates exponential kernels from beta values: [beta^(T-1), beta^(T-2), ..., beta^0]
    - Applies grouped convolutions to process all neurons/channels in parallel
    - Significantly faster than iterative for-loops over timesteps

    **Error Signal:**

    - Error: err = predicted_output - target (one-hot encoded)
    - Backpropagated through output weights: L = err @ w_out.T
    - Combined with eligibility traces to compute gradients

    **Gradient Accumulation:**

    - w_in.grad += sum over batch and time of: L * trace_in
    - w_rec.grad += sum over batch and time of: L * trace_rec  
    - w_out.grad += sum over time of: err @ trace_out

    **Delayed Output Option:**

    If params["delayed_output"] is set:
    - Only uses final delayed_output timesteps for gradient computation
    - Reduces temporal credit assignment window
    - Can improve learning stability in some cases

    **Memory Optimization:**

    - Intermediate tensors (h, trace_in, trace_rec) are large (batch * neurons * time)
    - Commented-out deletions can be enabled to free memory explicitly
    - Consider reducing batch_size or data_steps if memory is limited

    References
    ----------
    Bellec et al. (2020). "A solution to the learning dilemma for recurrent networks 
    of spiking neurons." Nature Communications, 11(1), 3625.

    Examples
    --------
    >>> # After forward pass through network with e-prop
    >>> grads_batch(x, yo, yt, gamma=15.0, thr=1, v=mem_rec, z=spk_rec,
    ...            w_in=rec_layer.ff_weights, w_rec=rec_layer.rec_weights,
    ...            w_out=ff_layer.ff_weights, beta_trace=0.9, beta_trace_out=0.95,
    ...            params=params)
    >>> # Now w_in.grad, w_rec.grad, w_out.grad contain accumulated gradients
    >>> optimizer.step()  # Apply gradients to weights
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
    beta_conv = torch.tensor([beta_trace_out ** (params['data_steps'] - i - 1)
                             for i in range(params['data_steps'])]).float().view(1, 1, -1).to(params['device'])
    beta_rec_conv = torch.tensor([beta_trace ** (params['data_steps'] - i - 1)
                                 for i in range(params['data_steps'])]).float().view(1, 1, -1).to(params['device'])

    # Convoluzione Input eligibility traces
    trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(
        nb_inputs, -1, -1), padding=params['data_steps'], groups=nb_inputs)[:, :, 1:params['data_steps']+1]
    trace_in = trace_in.unsqueeze(1).expand(-1, params['nb_hidden'], -1, -1)
    trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

    # Convoluzione Recurrent eligibility traces
    trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(
        params['nb_hidden'], -1, -1), padding=params['data_steps'], groups=params['nb_hidden'])[:, :, :params['data_steps']]
    trace_rec = trace_rec.unsqueeze(1).expand(-1, params['nb_hidden'], -1, -1)
    trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

    # Free h as it's no longer needed
    # del h

    # Output eligibility vector
    trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(
        params['nb_hidden'], -1, -1), padding=params['data_steps'], groups=params['nb_hidden'])[:, :, 1:params['data_steps']+1]

    # Ottimizzazione convoluzioni batch-wise
    trace_in = F.conv1d(trace_in.reshape(x.shape[1], nb_inputs * params['nb_hidden'], params['data_steps']),
                        beta_conv.expand(
                            nb_inputs * params['nb_hidden'], -1, -1),
                        padding=params['data_steps'], groups=nb_inputs * params['nb_hidden'])[:, :, 1:params['data_steps']+1]
    trace_in = trace_in.reshape(
        x.shape[1], params['nb_hidden'], nb_inputs, params['data_steps'])

    trace_rec = F.conv1d(trace_rec.reshape(x.shape[1], params['nb_hidden'] * params['nb_hidden'], params['data_steps']),
                         beta_conv.expand(
                             params['nb_hidden'] * params['nb_hidden'], -1, -1),
                         padding=params['data_steps'], groups=params['nb_hidden'] * params['nb_hidden'])[:, :, 1:params['data_steps']+1]
    trace_rec = trace_rec.reshape(
        x.shape[1], params['nb_hidden'], params['nb_hidden'], params['data_steps'])

    for i in range(yo.shape[0]):
        err[i, :, :] = yo[i, :, :] - yt
    err = err.to(params['dtype_torch'])

    L = torch.einsum('tbo,or->brt', err, w_out)

    if params["use_eprop"] and params["delayed_output"] is not None and params["delayed_output"] > 0:
        L = L[:, :, -params["delayed_output"]:]
        err = err[-params["delayed_output"]:, :, :]
        trace_in = trace_in[:, :, :, -params["delayed_output"]:]
        trace_rec = trace_rec[:, :, :, -params["delayed_output"]:]
        trace_out = trace_out[:, :, -params["delayed_output"]:]

    # Weight gradient updates
    w_in.grad += torch.sum(L.unsqueeze(2).expand(-1, -1,
                           nb_inputs, -1) * trace_in, dim=(0, 3))

    # Free trace_in immediately after use
    # del trace_in

    w_rec.grad += torch.sum(L.unsqueeze(2).expand(-1, -1,
                            params['nb_hidden'], -1) * trace_rec, dim=(0, 3))

    # Free trace_rec immediately after use
    # del trace_rec
    w_out.grad += torch.einsum('tbo,brt->or', err, trace_out)

    # Free remaining large tensors
    # del trace_out, L, err


def copy_layers(layers: list) -> list:
    """
    Create deep copies of layer objects by recreating them with copied weights.

    This function manually copies spiking neural network layer objects to avoid deepcopy
    issues with PyTorch tensors that have gradients attached. It creates new layer instances
    with the same architecture and copies over the learned weights (detached from the
    computational graph).

    Parameters
    ----------
    layers : list
        List of [recurrent_layer, feedforward_layer] objects to copy
        These should be instances of the recurrent_layer and feedforward_layer classes

    Returns
    -------
    list
        New list [new_recurrent_layer, new_feedforward_layer] with:
        - Same architecture (nb_inputs, nb_neurons, etc.) as original layers
        - Copied weights (detached and cloned to break gradient connections)
        - Fresh computational graph (no gradient history)

    Notes
    -----
    **Why Not Use copy.deepcopy():**
    - PyTorch tensors with gradients attached cause issues with deepcopy
    - Computational graph references can lead to memory leaks
    - Manual recreation ensures clean separation between original and copy

    **Attributes Copied:**
    For recurrent_layer:
    - ff_weights: input-to-hidden weights [hidden_neurons, input_features]
    - rec_weights: recurrent weights [hidden_neurons, hidden_neurons]

    For feedforward_layer:
    - ff_weights: hidden-to-output weights [output_neurons, hidden_neurons]

    **Attributes Preserved:**
    - batch_size, nb_inputs, nb_neurons: layer dimensions
    - fwd_scale, rec_scale: weight initialization scales
    - alpha, beta: neuron dynamics parameters
    - ref_per: refractory period settings

    **Use Case:**
    - Save best model during training without affecting ongoing optimization
    - Store multiple model checkpoints at different epochs
    - Compare different models with same architecture but different weights

    **Detach vs Clone:**
    - .detach(): removes tensor from computational graph (no gradients)
    - .clone(): creates new tensor with copied data
    - Together: creates independent copy safe for long-term storage

    Examples
    --------
    >>> # During training loop
    >>> if test_acc > best_test_acc:
    ...     best_test_acc = test_acc
    ...     best_layers = copy_layers(layers)  # Save best model
    >>> # Later, load best model for evaluation
    >>> layers = best_layers
    >>> test_accuracy = evaluate(layers, test_data)

    See Also
    --------
    recurrent_layer : Recurrent spiking layer class
    feedforward_layer : Feedforward spiking layer class
    """
    rec_layer, ff_layer = layers

    # Create new recurrent layer instance
    new_rec_layer = recurrent_layer(
        batch_size=rec_layer.batch_size,
        nb_inputs=rec_layer.nb_inputs,
        nb_neurons=rec_layer.nb_neurons,
        fwd_weight_scale=rec_layer.fwd_weight_scale,
        rec_weight_scale=rec_layer.rec_weight_scale,
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
        fwd_weight_scale=ff_layer.fwd_weight_scale,
        alpha=ff_layer.alpha,
        beta=ff_layer.beta,
        ref_per=ff_layer.ref_per
    )
    # Copy weights (detached and cloned)
    new_ff_layer.ff_weights.data = ff_layer.ff_weights.data.detach().clone()

    return [new_rec_layer, new_ff_layer]
