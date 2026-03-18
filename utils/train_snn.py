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
- Comprehensive logging for network diagnostics (DEBUG level)

Author: Simon F. Muller-Cleve
Date: January 15, 2026
"""

import logging
from typing import cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .neuron_models import feedforward_layer, recurrent_layer, ste_fn
from .snn import compute_winning_neuron, run_snn
from .validate_snn import compute_classification_accuracy

# Get logger instance
logger = logging.getLogger('braille_training')


def _normalize_eprop_mode(mode: str) -> str:
    alias_map = {
        'experimental': 'frenkel',
        'frenkel': 'frenkel',
        'traditional': 'bellec',
        'bellec': 'bellec',
    }
    normalized = alias_map.get(str(mode).lower())
    if normalized is None:
        raise ValueError(
            f"Unknown eprop_mode '{mode}'. Expected one of: "
            "frenkel, bellec (aliases: experimental, traditional)."
        )
    return normalized


def build_and_train(params: dict, ds_train: TensorDataset, ds_test: TensorDataset,
                    resume_weights: dict[str, np.ndarray] | None = None) -> tuple:
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
        - 'synapse' : bool
            If True, enables synaptic dynamics; if False, disables them (alpha=0)
        - 'linear_decay' : bool
            If True, uses linear membrane decay; if False, uses exponential
        - 'ref_per_timesteps' : int or None
            Refractory period duration in timesteps
        - 'lower_bound' : float or None
            Minimum membrane potential (clamping threshold)

        E-prop Specific:
        - 'eprop' : bool
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
        - 'quantize_weights' : bool
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

    resume_weights : dict or None
        Optional weight dictionary used to initialize the model before training.
        Expected keys: 'rec_ff_weights', 'rec_rec_weights', 'out_weights'.

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
            Values are float or None (None if eprop=False)

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
    - Synaptic dynamics: alpha = exp(-dt/tau_syn) or 0 if synapse=False

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
    logger.debug(f"Number of input copies {nb_input_copies}")
    # Network parameters
    nb_inputs = 2*len(params["selected_channels"])*nb_input_copies
    nb_outputs = len(params["letters"])
    nb_hidden = params['nb_hidden']
    logger.debug(f"Number of hidden neurons {nb_hidden}")

    tau_syn = params['tau_mem']/params['tau_ratio']
    if not params["synapse"]:
        alpha = 0.0  # here we disable synapse dynamics
    else:
        alpha = float(np.exp(-params["time_step"]/tau_syn))

    if params["linear_decay"]:
        # Linear decay uses a per-timestep voltage decrement, so make it explicit
        # in terms of simulation timestep (dt) and membrane constants.
        beta = float(params["time_step"] / params['tau_mem'])
        beta_rec = float(params["time_step"] / params['tau_mem_rec'])
    else:
        # Exponential decay uses a per-timestep retention factor.
        beta = float(np.exp(-params["time_step"]/params['tau_mem']))
        beta_rec = float(np.exp(-params["time_step"]/params['tau_mem_rec'])
                         )

    # Expose decay factors so traditional e-prop can mirror neuron/readout dynamics.
    params['beta_mem'] = beta
    params['beta_mem_rec'] = beta_rec

    if params["eprop"]:
        params['beta_trace'] = float(
            np.exp(-params["time_step"]/params['tau_trace']))
        params['beta_trace_out'] = float(
            np.exp(-params["time_step"]/params['tau_trace_out']))
        vars_eprop = [params['beta_trace'], params['beta_trace_out']]
    else:
        params['beta_trace'] = None
        params['beta_trace_out'] = None
        vars_eprop = [None, None]

    lr_layer = params.get('eprop_lr_layer', (1.0, 1.0, 1.0))
    if len(lr_layer) != 3:
        raise ValueError("eprop_lr_layer must contain exactly 3 values: (in, rec, out).")
    params['eprop_lr_layer'] = tuple(float(v) for v in lr_layer)

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = fwd_weight_scale*params['weight_scale_factor']
    
    logger.debug(f"Building network architecture:")
    logger.debug(f"  Input neurons: {nb_inputs}")
    logger.debug(f"  Hidden neurons: {nb_hidden}")
    logger.debug(f"  Output neurons: {nb_outputs}")
    logger.debug(f"  Forward weight scale: {fwd_weight_scale}")
    logger.debug(f"  Recurrent weight scale: {rec_weight_scale}")
    logger.debug(f"  Alpha (synaptic decay): {alpha}")
    logger.debug(f"  Beta (membrane decay): {beta_rec}")

    # Spiking network
    # recurrent layer
    rec_layer = recurrent_layer(batch_size=params["batch_size"],
                                nb_inputs=nb_inputs,
                                nb_neurons=nb_hidden,
                                fwd_weight_scale=fwd_weight_scale,
                                rec_weight_scale=rec_weight_scale,
                                alpha=alpha,
                                beta=beta_rec,
                                weight_variance = params["weight_distribution"],
                                eprop=params["eprop"],
                                linear_decay=params["linear_decay"],
                                device=params["device"],
                                dtype=params["dtype_torch"],
                                ref_per=params["ref_per_timesteps"],
                                gamma=params["gamma"],
                                spike_threshold=params["spike_threshold"],
                                soft_reset=params.get("soft_reset", False))

    # readout layer
    ff_layer = feedforward_layer(batch_size=params["batch_size"],
                                 nb_inputs=nb_hidden,
                                 nb_neurons=nb_outputs,
                                 fwd_weight_scale=fwd_weight_scale,
                                 alpha=alpha,
                                 beta=beta,
                                 weight_variance = params["weight_distribution"],
                                 eprop=params["eprop"],
                                 linear_decay=params["linear_decay"],
                                 device=params["device"],
                                 dtype=params["dtype_torch"],
                                 ref_per=params["ref_per_timesteps"],
                                 gamma=params["gamma"],
                                 spike_threshold=params["spike_threshold"],
                                 soft_reset=params.get("soft_reset", False))

    layers = [rec_layer, ff_layer]

    if resume_weights is not None:
        apply_resume_weights(layers, resume_weights, params)
        logger.info("Resume weights applied to initialized layers.")

    train_rec_ff = bool(params.get("train_rec_ff", True))
    train_rec_rec = bool(params.get("train_rec_rec", True))
    train_out_ff = bool(params.get("train_out_ff", True))

    rec_layer.ff_weights.requires_grad_(train_rec_ff)
    rec_layer.rec_weights.requires_grad_(train_rec_rec)
    ff_layer.ff_weights.requires_grad_(train_out_ff)

    logger.info(
        "Trainable weights: rec_ff=%s, rec_rec=%s, out_ff=%s",
        train_rec_ff,
        train_rec_rec,
        train_out_ff,
    )

    # Save initial weights before training
    initial_weights = save_weights(layers)
    logger.debug(f"Initial weights saved (rec_ff: {initial_weights['rec_ff_weights'].shape}, "
                f"rec_rec: {initial_weights['rec_rec_weights'].shape}, "
                f"out: {initial_weights['out_weights'].shape})")
    
    # Log initial weight statistics in DEBUG mode
    logger.debug(f"Initial recurrent ff weight stats: mean={initial_weights['rec_ff_weights'].mean():.6f}, "
                f"std={initial_weights['rec_ff_weights'].std():.6f}")
    logger.debug(f"Initial recurrent rec weight stats: mean={initial_weights['rec_rec_weights'].mean():.6f}, "
                f"std={initial_weights['rec_rec_weights'].std():.6f}")
    logger.debug(f"Initial output weight stats: mean={initial_weights['out_weights'].mean():.6f}, "
                f"std={initial_weights['out_weights'].std():.6f}")

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

    return loss_hist_epochs, accs_hist_epochs, best_layers, vars_eprop, initial_weights


def train(params: dict, dataset: TensorDataset, layers: list, vars_eprop: list,
          dataset_test: TensorDataset | None = None) -> tuple:
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
        - 'eprop' : bool
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
        - 'quantize_weights' : bool
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
            If True, automatically sets log_level to DEBUG for detailed diagnostics
        - 'random_tie_breaking' : bool
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
        - Both are None if eprop=False

    dataset_test : TensorDataset, optional
        Test dataset for evaluation after each epoch (default: None)
        If None, only training accuracy is tracked

    Returns
    -------
    tuple
        (loss_hist_epochs, accs_hist_epochs, best_acc_layers) where:

        - loss_hist_epochs : list of float
            Training loss values per epoch (length: epochs)
            Loss includes NLL loss + regularization (for BPTT only, e-prop computes gradients directly)
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
    - Weight quantization applied in forward path via STE (master params remain float)

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
    - Supports random tie-breaking if random_tie_breaking=True

    **Debug Logging (first batch of first epoch only, when debug=True):**
    - Logged via logger.debug() at DEBUG level
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
    if dataset_test is None:
        raise ValueError("dataset_test must be provided.")

    all_weights = [
        ("rec_ff_weights", layers[0].ff_weights),
        ("rec_rec_weights", layers[0].rec_weights),
        ("out_weights", layers[1].ff_weights),
    ]
    weights = [weight for _, weight in all_weights if weight.requires_grad]
    if len(weights) == 0:
        frozen = ", ".join(name for name, _ in all_weights)
        raise ValueError(
            f"No trainable weights found (all frozen): {frozen}. "
            "Enable requires_grad on at least one weight matrix."
        )
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
    optimizer = torch.optim.Adamax(
        weights, lr=params["learning_rate"], betas=(0.9, 0.995))
    count_epoch = 0
    for _ in pbar_training:
        # learning rate decreases over epochs
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

            # average_spike_output[count_epoch] = torch.mean(
            #     torch.sum(spk_rec_readout, 1))
            # average_spike_recurrent[count_epoch] = torch.mean(
            #     torch.sum(recs[1], 1))

            spk_rec_hidden = recs[1]

            # Compute gradients based on selected learning algorithm
            if params["eprop"]:
                eprop_mode = _normalize_eprop_mode(
                    params.get("eprop_mode", "frenkel")
                )
                one_hot_encoded = torch.nn.functional.one_hot(
                    y_local, num_classes=len(params['letters']))
                # E-prop: manual gradient computation via eligibility traces
                mem_rec_hidden, spk_rec_hidden, mem_rec_readout = recs[:3]
                # For classification, use time-resolved class probabilities
                # derived from the readout membrane state.
                yo = torch.softmax(mem_rec_readout, dim=2)

                if params["delayed_output"] is not None and params["delayed_output"] > 0:
                    summed_scores, neuron_idc = compute_winning_neuron(
                        yo[:, -params["delayed_output"]:, :], params=params)
                else:
                    summed_scores, neuron_idc = compute_winning_neuron(
                        spk_rec_readout=yo, params=params)
                debug_scores = summed_scores

                w_in_math: torch.Tensor
                w_rec_math: torch.Tensor
                w_out_math: torch.Tensor
                grad_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None

                if params.get("quantize_weights", False):
                    possible_weights = params.get('possible_weights')
                    if possible_weights is None:
                        raise ValueError("possible_weights must be set when quantize_weights=True")
                    possible_weights_tensor = cast(torch.Tensor, possible_weights)
                    w_in_math = cast(torch.Tensor, ste_fn(layers[0].ff_weights, possible_weights_tensor))
                    w_rec_math = cast(torch.Tensor, ste_fn(layers[0].rec_weights, possible_weights_tensor))
                    w_out_math = cast(torch.Tensor, ste_fn(layers[1].ff_weights, possible_weights_tensor))
                    grad_targets = (
                        layers[0].ff_weights,
                        layers[0].rec_weights,
                        layers[1].ff_weights,
                    )
                else:
                    w_in_math = layers[0].ff_weights
                    w_rec_math = layers[0].rec_weights
                    w_out_math = layers[1].ff_weights
                    grad_targets = None

                grads_batch(x=x_local.permute(1, 0, 2),
                            yo=yo.permute(1, 0, 2),
                            yt=one_hot_encoded,
                            thr=params["spike_threshold"],
                            v=mem_rec_hidden.permute(1, 0, 2),
                            z=spk_rec_hidden.permute(1, 0, 2),
                            w_in=w_in_math,
                            w_rec=w_rec_math,
                            w_out=w_out_math,
                            params=params,
                            grad_targets=grad_targets)

                # Compute loss for tracking purposes (not used for gradients in e-prop)
                log_p_y = log_softmax_fn(summed_scores)
                loss_val = loss_fn(log_p_y, y_local)
            else:
                # Use all timesteps for BPTT predictions
                if params["delayed_output"] is not None and params["delayed_output"] > 0:
                    summed_spikes, neuron_idc = compute_winning_neuron(
                        spk_rec_readout[:, -params["delayed_output"]:, :], params=params)
                else:
                    summed_spikes, neuron_idc = compute_winning_neuron(
                        spk_rec_readout=spk_rec_readout, params=params)
                debug_scores = summed_spikes

                log_p_y = log_softmax_fn(summed_spikes)

                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = params['reg_spikes'] * \
                    torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                logger.debug(f"L1: {reg_loss}")
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                reg_loss += params['reg_neurons'] * \
                    torch.mean(
                        torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                logger.debug(f"L1 + L2: {reg_loss}")
                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                # BPTT: standard backpropagation
                loss_val.backward()

            optimizer.step()

            loss_hist_batches.append(loss_val.item())

            # Debug logging: Log first batch of first epoch to check predictions
            if params['debug'] and count_epoch == 0:
                acc_before_update = np.mean(
                    (y_local == neuron_idc).detach().cpu().numpy())
                logger.debug(f"\nDebug - First batch:")
                logger.debug(
                    f"  True labels (y_local): {y_local[:min(10, len(y_local))].cpu().numpy()}")
                logger.debug(
                    f"  Predictions (neuron_idc): {neuron_idc[:min(10, len(y_local))].cpu().numpy()}")
                logger.debug(
                    f"  Readout scores (summed over time): {debug_scores[:min(10, len(y_local))].detach().cpu().numpy()}")
                logger.debug(
                    f"  Label distribution: 0={torch.sum(y_local==0).item()}, 1={torch.sum(y_local==1).item()}")
                logger.debug(
                    f"  Prediction distribution: 0={torch.sum(neuron_idc==0).item()}, 1={torch.sum(neuron_idc==1).item()}")
                logger.debug(f"  Batch accuracy: {acc_before_update:.4f}")

                # Check input data
                logger.debug(f"\n  Input data statistics:")
                logger.debug(f"    Shape: {x_local.shape}")
                logger.debug(
                    f"    Total input spikes (first 10 samples): {torch.sum(x_local[:min(10, len(y_local))], dim=(0,1)).cpu().numpy()}")
                logger.debug(
                    f"    Input spike rate (mean): {x_local.mean().item():.6f}")

                # Check hidden layer
                logger.debug(f"\n  Hidden layer statistics:")
                # sum over time: [batch, neurons]
                hidden_spike_counts = torch.sum(spk_rec_hidden, dim=0)
                logger.debug(
                    f"    Spike count distribution (mean across batch): {hidden_spike_counts.mean(dim=0).cpu().detach().numpy()}")
                logger.debug(
                    f"    Spike count (min, max): ({hidden_spike_counts.min().item()}, {hidden_spike_counts.max().item()})")
                logger.debug(
                    f"    First 10 samples total spikes: {torch.sum(spk_rec_hidden[:min(10, len(y_local))], dim=(0,1)).cpu().detach().numpy()}")

                # Check output layer
                logger.debug(f"\n  Output layer statistics:")
                logger.debug(f"    Weights shape: {layers[1].ff_weights.shape}")
                logger.debug(
                    f"    Weights mean per neuron: {torch.mean(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
                logger.debug(
                    f"    Weights std per neuron: {torch.std(layers[1].ff_weights, dim=1).detach().cpu().numpy()}")
                h2_sample = torch.einsum(
                    "abc,cd->abd", (spk_rec_hidden[:min(10, len(y_local))], layers[1].ff_weights.t()))
                logger.debug(
                    f"    Input current (h2) mean per output neuron: {h2_sample.mean(dim=(0,1)).detach().cpu().numpy()}")
                logger.debug(
                    f"    Number of positive weights: neuron0={torch.sum(layers[1].ff_weights[0] > 0).item()}, neuron1={torch.sum(layers[1].ff_weights[1] > 0).item()}")
                logger.debug("")

            # Calculate train accuracy in each batch
            train_acc_per_batch, _, _ = compute_classification_accuracy(
                dataset=dataset, layers=layers, params=params)
            accs_hist_batches.append(train_acc_per_batch)

            if params['debug'] and count_epoch == 0:
                logger.debug(
                    f"Train acc after weight update: {train_acc_per_batch*100:.2f}%")

            # Update batch progress bar with current and running average accuracy
            current_acc = train_acc_per_batch * 100
            running_avg_acc = np.mean(accs_hist_batches) * 100
            pbar_batches.set_description(
                f"Batch acc: {current_acc:.1f}%, Running avg: {running_avg_acc:.1f}%")

            # Free up memory by deleting large intermediate tensors
            # del spk_rec_readout, recs, spk_rec_hidden, m, am
            # if not params["eprop"]:
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

        # Early stopping: check if training is not improving in initial epochs
        if params.get('early_stop_epochs', 0) > 0 and count_epoch <= params['early_stop_epochs']:
            current_train_acc = accs_hist_epochs[0][-1]
            
            # Calculate adaptive threshold based on number of classes
            num_classes = len(params['letters'])
            chance_level = 1.0 / num_classes
            percentage_above_chance = params.get('early_stop_threshold', 20.0)  # default 20% above chance
            adaptive_threshold = chance_level + (percentage_above_chance / 100.0)
            
            # Check if we're still below threshold at the end of the early stopping window
            if count_epoch == params['early_stop_epochs'] and current_train_acc < adaptive_threshold:
                logger.warning(f"Early stopping triggered at epoch {count_epoch}:")
                logger.warning(f"  Number of classes: {num_classes}")
                logger.warning(f"  Chance level: {chance_level*100:.2f}%")
                logger.warning(f"  Required threshold: {adaptive_threshold*100:.2f}% ({percentage_above_chance:.1f} percentage points above chance)")
                logger.warning(f"  Training accuracy: {current_train_acc*100:.2f}%")
                logger.warning(f"  Model appears to have poor weight initialization. Stopping training.")
                # If no best model was saved yet, save current state
                if len(best_acc_layers) == 0:
                    best_acc_layers = copy_layers(layers)
                break

    return loss_hist_epochs, accs_hist_epochs, best_acc_layers


def grads_batch(x: torch.Tensor, yo: torch.Tensor, yt: torch.Tensor, thr: int, v: torch.Tensor, z: torch.Tensor, w_in: torch.Tensor, w_rec: torch.Tensor, w_out: torch.Tensor, params: dict, grad_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> None:
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
        - 'eprop' : bool
            Should be True when calling this function
        - 'delayed_output' : int or None
            If set, only uses final delayed_output timesteps for gradient computation
            If None or 0, uses all timesteps
        - 'gamma' : float
            Surrogate gradient scaling factor for the spike function derivative approximation
            Typical value: 15.0 (controls steepness of surrogate gradient)
        - 'beta_trace' : float
            Decay factor for eligibility traces of hidden layer connections
            Computed as exp(-dt/tau_trace) where dt is timestep and tau_trace is trace time constant
            Controls how long past spike events influence current weight updates
        - 'beta_trace_out' : float
            Decay factor for output eligibility traces
            Computed as exp(-dt/tau_trace_out) for output layer
            Typically equal to or larger than beta_trace

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

    if grad_targets is None:
        grad_w_in, grad_w_rec, grad_w_out = w_in, w_rec, w_out
    else:
        grad_w_in, grad_w_rec, grad_w_out = grad_targets

    if grad_w_in.grad is None:
        grad_w_in.grad = torch.zeros_like(grad_w_in)
    if grad_w_rec.grad is None:
        grad_w_rec.grad = torch.zeros_like(grad_w_rec)
    if grad_w_out.grad is None:
        grad_w_out.grad = torch.zeros_like(grad_w_out)

    if _normalize_eprop_mode(params.get("eprop_mode", "frenkel")) == "bellec":
        nb_hidden = int(params['nb_hidden'])
        batch_size = x.shape[1]
        dtype = params['dtype_torch']
        device = params['device']
        beta_trace = float(params.get('beta_mem_rec', params['beta_trace']))
        beta_out = float(params.get('beta_mem', params['beta_trace_out']))
        lr_in, lr_rec, lr_out = params.get('eprop_lr_layer', (1.0, 1.0, 1.0))
        thr_value = float(thr)

        # Bellec et al. e-prop (online recursion):
        # epsilon_t = d h_t / d h_{t-1} * epsilon_{t-1} + d h_t / dW
        # e_t = d z_t / d h_t * epsilon_t
        # gradient = sum_t L_t * \bar{e}_t, with \bar{e}_t low-pass filtered by output leak.
        pre_in = torch.zeros((batch_size, nb_inputs), device=device, dtype=dtype)
        pre_rec = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
        z_prev = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

        bar_e_in = torch.zeros((batch_size, nb_hidden, nb_inputs), device=device, dtype=dtype)
        bar_e_rec = torch.zeros((batch_size, nb_hidden, nb_hidden), device=device, dtype=dtype)
        bar_z = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

        start_idx = 0
        if params["eprop"] and params["delayed_output"] is not None and params["delayed_output"] > 0:
            start_idx = max(0, x.shape[0] - int(params["delayed_output"]))

        yt_float = yt.to(dtype)

        for t in range(x.shape[0]):
            x_t = x[t]
            z_t = z[t]
            v_t = v[t]

            pre_in = beta_trace * pre_in + x_t
            pre_rec = beta_trace * pre_rec + z_prev

            psi = params['gamma'] * torch.clamp(
                1 - torch.abs((v_t - thr_value) / thr_value), min=0.0
            )

            e_in = psi.unsqueeze(-1) * pre_in.unsqueeze(1)
            e_rec = psi.unsqueeze(-1) * pre_rec.unsqueeze(1)

            bar_e_in = beta_out * bar_e_in + e_in
            bar_e_rec = beta_out * bar_e_rec + e_rec
            bar_z = beta_out * bar_z + z_t

            if t >= start_idx:
                err_t = yo[t].to(dtype) - yt_float
                learning_t = torch.einsum('bo,or->br', err_t, w_out)

                grad_w_in.grad += float(lr_in) * torch.einsum('br,bri->ri', learning_t, bar_e_in)
                grad_w_rec.grad += float(lr_rec) * torch.einsum('br,brj->rj', learning_t, bar_e_rec)
                grad_w_out.grad += float(lr_out) * torch.einsum('bo,br->or', err_t, bar_z)

            z_prev = z_t

        return

    # Experimental mode (ReckOn-inspired):
    # decoupled pre-synaptic traces and post-synaptic factors (LS * STE),
    # with neuron-scaled ET storage instead of synapse-sized ET tensors.
    dtype = params['dtype_torch']
    device = params['device']
    nb_hidden = int(params['nb_hidden'])
    batch_size = x.shape[1]
    thr_value = float(thr)
    lr_in, lr_rec, lr_out = params.get('eprop_lr_layer', (1.0, 1.0, 1.0))

    decay_out_raw = params.get('beta_mem', params.get('beta_trace_out'))
    decay_rec_raw = params.get('beta_mem_rec', params.get('beta_trace'))
    if decay_out_raw is None or decay_rec_raw is None:
        raise ValueError(
            "Missing decay constants for experimental e-prop: "
            "expected beta_mem/beta_mem_rec (or beta_trace_out/beta_trace fallback)."
        )
    decay_out = float(decay_out_raw)
    decay_rec = float(decay_rec_raw)

    pre_in_trace = torch.zeros((batch_size, nb_inputs), device=device, dtype=dtype)
    pre_rec_trace = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    output_trace = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    z_prev = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    start_idx = 0
    if params["eprop"] and params["delayed_output"] is not None and params["delayed_output"] > 0:
        start_idx = max(0, x.shape[0] - int(params["delayed_output"]))

    yt_float = yt.to(dtype)

    for t in range(x.shape[0]):
        x_t = x[t].to(dtype)
        z_t = z[t].to(dtype)
        v_t = v[t].to(dtype)

        pre_in_trace = decay_rec * pre_in_trace + x_t
        pre_rec_trace = decay_rec * pre_rec_trace + z_prev
        output_trace = decay_out * output_trace + z_t

        ste_t = params['gamma'] * torch.clamp(
            1 - torch.abs((v_t - thr_value) / thr_value), min=0.0
        )

        err_t = yo[t].to(dtype) - yt_float
        ls_t = torch.einsum('bo,or->br', err_t, w_out)
        post_factor = ste_t * ls_t

        if t >= start_idx:
            grad_w_in.grad += float(lr_in) * torch.einsum('br,bi->ri', post_factor, pre_in_trace)
            grad_w_rec.grad += float(lr_rec) * torch.einsum('br,bj->rj', post_factor, pre_rec_trace)
            grad_w_out.grad += float(lr_out) * torch.einsum('bo,br->or', err_t, output_trace)

        z_prev = z_t

    # Free remaining large tensors
    # del trace_out, L, err


def save_weights(
    layers: list | tuple,
    possible_weights: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """
    Extract weight matrices from network layers as numpy arrays for storage.

    This function extracts all weight matrices from the network layers and converts
    them to numpy arrays (detached from computational graph and moved to CPU).
    Useful for saving weight snapshots at initialization or after training.

    Parameters
    ----------
    layers : list or tuple
        Container with [recurrent_layer, feedforward_layer] objects.
    possible_weights : torch.Tensor or None, optional
        Quantization levels tensor. When provided, exported weights are snapped
        to the nearest level before conversion to numpy.

    Returns
    -------
    dict
        Dictionary containing:
        - 'rec_ff_weights': input-to-hidden weights [hidden_neurons, input_features]
        - 'rec_rec_weights': recurrent weights [hidden_neurons, hidden_neurons]
        - 'out_weights': hidden-to-output weights [output_neurons, hidden_neurons]

    Examples
    --------
    >>> initial_weights = save_weights(layers)
    >>> np.savez('initial_weights.npz', **initial_weights)
    """
    rec_layer, ff_layer = layers

    def _export_tensor(weight_tensor: torch.Tensor) -> np.ndarray:
        tensor = weight_tensor.detach()
        if possible_weights is not None:
            levels = possible_weights.to(device=tensor.device, dtype=tensor.dtype)
            quantized = ste_fn(tensor, levels)
            if quantized is None:
                raise RuntimeError("STE quantization returned None")
            tensor = quantized.detach()
        return tensor.cpu().numpy()

    weights = {
        'rec_ff_weights': _export_tensor(rec_layer.ff_weights.data),
        'rec_rec_weights': _export_tensor(rec_layer.rec_weights.data),
        'out_weights': _export_tensor(ff_layer.ff_weights.data)
    }

    return weights


def apply_resume_weights(layers: list, weights: dict[str, np.ndarray], params: dict) -> None:
    """
    Apply pretrained weights to newly initialized layers.

    Parameters
    ----------
    layers : list
        [recurrent_layer, feedforward_layer] objects to receive weights.
    weights : dict
        Weight dictionary with keys 'rec_ff_weights', 'rec_rec_weights', 'out_weights'.
    params : dict
        Training params containing 'device' and 'dtype_torch'.
    """
    required_keys = ["rec_ff_weights", "rec_rec_weights", "out_weights"]
    missing = [key for key in required_keys if key not in weights]
    if missing:
        raise ValueError(f"Resume weights missing keys: {missing}")

    rec_layer, ff_layer = layers

    rec_ff = np.asarray(weights["rec_ff_weights"])
    rec_rec = np.asarray(weights["rec_rec_weights"])
    out = np.asarray(weights["out_weights"])

    if rec_ff.shape != tuple(rec_layer.ff_weights.shape):
        raise ValueError(
            f"rec_ff_weights shape mismatch: expected {tuple(rec_layer.ff_weights.shape)}, "
            f"got {rec_ff.shape}")
    if rec_rec.shape != tuple(rec_layer.rec_weights.shape):
        raise ValueError(
            f"rec_rec_weights shape mismatch: expected {tuple(rec_layer.rec_weights.shape)}, "
            f"got {rec_rec.shape}")
    if out.shape != tuple(ff_layer.ff_weights.shape):
        raise ValueError(
            f"out_weights shape mismatch: expected {tuple(ff_layer.ff_weights.shape)}, "
            f"got {out.shape}")

    rec_ff_tensor = torch.as_tensor(rec_ff, device=params["device"], dtype=params["dtype_torch"])
    rec_rec_tensor = torch.as_tensor(rec_rec, device=params["device"], dtype=params["dtype_torch"])
    out_tensor = torch.as_tensor(out, device=params["device"], dtype=params["dtype_torch"])

    rec_layer.ff_weights.data.copy_(rec_ff_tensor)
    rec_layer.rec_weights.data.copy_(rec_rec_tensor)
    ff_layer.ff_weights.data.copy_(out_tensor)


def load_weights_from_model(model_path: str, map_location="cpu") -> dict:
    """
    Load a saved model (.pt) and return a weight dictionary.

    Parameters
    ----------
    model_path : str
        Path to a saved model file created with torch.save(best_layers, ...).
    map_location : str or torch.device
        Device mapping for torch.load (default: "cpu").
    """
    # NOTE: these checkpoints store full layer objects; weights_only must be False.
    layers = torch.load(model_path, map_location=map_location, weights_only=False)
    if not isinstance(layers, (list, tuple)) or len(layers) != 2:
        raise ValueError("Expected a list/tuple of [recurrent_layer, feedforward_layer].")
    return save_weights(layers)


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
        eprop=rec_layer.eprop,
        linear_decay=rec_layer.linear_decay,
        device=rec_layer.device,
        dtype=rec_layer.dtype,
        ref_per=rec_layer.ref_per,
        gamma=rec_layer.gamma,
        spike_threshold=rec_layer.spike_threshold,
        soft_reset=getattr(rec_layer, "soft_reset", False)
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
        eprop=ff_layer.eprop,
        linear_decay=ff_layer.linear_decay,
        device=ff_layer.device,
        dtype=ff_layer.dtype,
        ref_per=ff_layer.ref_per,
        gamma=ff_layer.gamma,
        spike_threshold=ff_layer.spike_threshold,
        soft_reset=getattr(ff_layer, "soft_reset", False)
    )
    # Copy weights (detached and cloned)
    new_ff_layer.ff_weights.data = ff_layer.ff_weights.data.detach().clone()

    return [new_rec_layer, new_ff_layer]
