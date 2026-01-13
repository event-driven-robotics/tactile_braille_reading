"""braille_reading_rsnn_mod_eprop_reduce_label.py

Comprehensive training script for spiking recurrent neural networks (SRNNs) on tactile 
braille letter classification tasks with full command-line configurability.

This script provides a complete training pipeline for spiking neural networks with:
- Configurable network architecture (neurons, layers, connections)
- Multiple learning algorithms (BPTT, e-prop)
- Flexible neuron dynamics (exponential/linear decay, synaptic filtering, refractory periods)
- Data preprocessing options (mechanoreceptor encoding, channel selection, letter filtering)
- Regularization and weight quantization support
- Comprehensive evaluation with confusion matrices and network activity visualization
- Reproducible experiments with seed control
- GPU acceleration support

The script uses argparse for all hyperparameters, making it suitable for systematic
hyperparameter searches, ablation studies, and production training runs.

Usage Examples
--------------
Train with default settings (all letters, e-prop disabled, 50 neurons):
    python braille_reading_rsnn_mod_eprop_reduce_label.py

Train with e-prop on specific letters:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --use_eprop --letters A B C

Custom architecture with validation set:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --nb_hidden 100 --use_validation

Select specific tactile sensors:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --selected_channels 0 1 2 5 8

Author: Simon F. Muller-Cleve
Date: January 13, 2026
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_and_extract
from utils.train_snn import build_and_train
from utils.validate_snn import (compute_classification_accuracy,
                                get_network_activity, plot_confusion_matrix,
                                plot_network_activity,
                                plot_training_perfromance)

# Configure PyTorch memory allocator for better GPU memory management
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Clear any cached GPU memory from previous runs
torch.cuda.empty_cache()


def parse_arguments():
    """
    Parse command line arguments for experiment configuration.

    Returns
    -------
    dict
        Dictionary containing all experiment parameters with computed derived values.
        Includes network architecture, training hyperparameters, neuron dynamics,
        regularization settings, data configuration, and model options.
    """
    parser = argparse.ArgumentParser(
        description='Train RSNN for Braille letter classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with verbose output')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA for GPU acceleration')
    parser.add_argument('--use_seed', action='store_true', default=False,
                        help='Use seed for reproducibility')
    parser.add_argument('--use_validation', action='store_true', default=False,
                        help='Create validation set from training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')

    # Network architecture
    parser.add_argument('--selected_channels', type=int, nargs='+', default=list(range(12)),
                        help='List of selected input channels (taxels)')
    parser.add_argument('--nb_hidden', type=int, default=50,
                        help='Number of recurrent hidden neurons')
    parser.add_argument('--nb_input_copies', type=int, default=1,
                        help='Number of copies for each input channel')

    # Neuron dynamics
    parser.add_argument('--tau_mem', type=float, default=0.06,
                        help='Membrane time constant (seconds)')
    parser.add_argument('--tau_mem_rec', type=float, default=0.06,
                        help='Recurrent membrane time constant (seconds)')
    parser.add_argument('--tau_trace', type=float, default=0.14,
                        help='Eligibility trace time constant (seconds)')
    parser.add_argument('--tau_trace_out', type=float, default=0.14,
                        help='Output trace time constant (seconds)')
    parser.add_argument('--tau_ratio', type=float, default=10,
                        help='Ratio for tau_syn calculation')
    parser.add_argument('--ref_per_timesteps', type=int, default=3,
                        help='Refractory period in timesteps')
    parser.add_argument('--lower_bound', type=float, default=-1.0,
                        help='Lower bound for membrane potential')

    # Weight parameters
    parser.add_argument('--fwd_weight_scale', type=float, default=10,
                        help='Forward weight initialization scale')
    parser.add_argument('--weight_scale_factor', type=float, default=0.2,
                        help='Recurrent weight scale factor')

    # Regularization
    parser.add_argument('--reg_spikes', type=float, default=0.0015,
                        help='L1 regularization coefficient for spikes')
    parser.add_argument('--reg_neurons', type=float, default=0.0,
                        help='L2 regularization coefficient for neurons')

    # Learning algorithm
    parser.add_argument('--use_eprop', action='store_true', default=False,
                        help='Use e-prop instead of BPTT')
    parser.add_argument('--gamma', type=float, default=15.0,
                        help='Surrogate gradient scale factor')

    # Data parameters
    parser.add_argument('--letters', type=str, nargs='+',
                        default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                        help='List of letters to use for classification')
    parser.add_argument('--threshold', type=int, default=2, choices=[1, 2, 5, 10],
                        help='Threshold for data encoding')
    parser.add_argument('--time_bin_size', type=int, default=1,
                        help='Time bin size in milliseconds')
    parser.add_argument('--use_mechanoreceptor_encoding', action='store_true', default=True,
                        help='Use mechanoreceptor encoding (vs sigma-delta)')

    # Model options
    parser.add_argument('--no_synapse', action='store_true', default=True,
                        help='Disable synaptic dynamics')
    parser.add_argument('--use_linear_decay', action='store_true', default=False,
                        help='Use linear decay instead of exponential')
    parser.add_argument('--use_weight_quantization', action='store_true', default=False,
                        help='Enable weight quantization')
    parser.add_argument('--use_random_tie_breaking', action='store_true', default=False,
                        help='Use random tie breaking for predictions')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float16', 'float32', 'float64'],
                        help='Torch data type for computations')

    args = parser.parse_args()

    # Compute derived parameters
    if args.use_mechanoreceptor_encoding:
        args.max_time = 3700
    else:
        args.max_time = 3501

    # Convert to dict for backward compatibility
    return vars(args)


##############################################################################
# CONFIGURATION SETUP
##############################################################################

# Parse all command-line arguments into params dictionary
params = parse_arguments()

# Configure letter set for classification
letters = params['letters']
all_letters = (len(letters) == 27)  # True if using complete alphabet + space
print(f"Using letters: {', '.join(letters)}")

# Configure PyTorch data type for all tensors
dtype_map = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64
}
params['dtype_torch'] = dtype_map[params['dtype']]
torch.set_default_dtype(params['dtype_torch'])
print(f"Using torch dtype: {params['dtype']}")

##############################################################################
# VISUALIZATION AND OUTPUT CONFIGURATION
##############################################################################

# Control how many network activity plots to generate
NB_BATCHES_TO_PLOT = 1  # Number of batches to visualize
NB_TRIALS_TO_PLOT = 1   # Number of trials per batch to visualize

# Create output directory for figures
path = './figures'
if not os.path.exists(path):
    os.makedirs(path)

##############################################################################
# DEVICE CONFIGURATION
##############################################################################

# Select computation device (GPU if available and requested, otherwise CPU)
if params["use_cuda"] and torch.cuda.is_available():
    params['device'] = torch.device("cuda:0")
    print("Using CUDA for computation.")
else:
    params['device'] = torch.device("cpu")
    print("Using CPU for computation.")

##############################################################################
# WEIGHT QUANTIZATION SETUP (Optional)
##############################################################################

# Configure weight quantization for neuromorphic hardware deployment
if params['use_weight_quantization']:
    print("Using weight quantization.")
    # Generate discrete weight levels based on capacitor bank values (256 levels)
    neg_capacitance = torch.arange(255, -1, -1)
    pos_capacitance = torch.arange(1, 257)
    diff_cap = pos_capacitance - neg_capacitance
    diff_cap = diff_cap.to(params['device'])
    q = 1/256

    possible_weights = diff_cap * q

    # Round to 3 decimal places for hardware precision
    factor = 10 ** 3
    params['possible_weights'] = torch.floor(
        possible_weights * factor) / factor


##############################################################################
# RANDOM SEED CONFIGURATION (For Reproducibility)
##############################################################################

# Set random seeds for reproducible experiments
if params["use_seed"]:
    seed = 42  # Answer to the Ultimate Question of Life, the Universe, and Everything
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")

##############################################################################
# DATA FILE CONFIGURATION
##############################################################################

# Determine which data file to load based on encoding method
file_dir_data = './data/100Hz/'
if params["use_mechanoreceptor_encoding"]:
    file_name = file_dir_data + 'mechanoreceptor_encoded.pkl'
else:
    # Use sigma-delta encoding with specified threshold
    file_type = 'data_braille_letters_100Hz_th'
    file_thr = str(params["threshold"])
    file_name = file_dir_data + file_type + file_thr + '.pkl'


if __name__ == '__main__':
    ##########################################################################
    # MAIN TRAINING PIPELINE
    ##########################################################################

    # Display learning algorithm configuration
    print(f"\n{'='*60}")
    print(
        f"Training with: {'e-prop' if params['use_eprop'] else 'BPTT (Backpropagation Through Time)'}")
    print(f"{'='*60}\n")

    # Generate descriptive string for output files based on letter set
    if all_letters:
        str_letters = 'all_letters'
    else:
        str_letters = '_'.join(letters)

    nb_hidden = params['nb_hidden']

    # Define paths for saving results and models
    results_file = f"./results/braille_reading_rsnn_{nb_hidden}_neurons_{str_letters}.npz"

    ##########################################################################
    # DATA LOADING AND PREPROCESSING
    ##########################################################################

    # Load and preprocess tactile sensor data
    ds_train, ds_test, ds_validation, labels = load_and_extract(
        params=params, file_name=file_name, letter_written=letters)

    # Clear GPU cache before training
    torch.cuda.empty_cache()

    ##########################################################################
    # DATASET INFORMATION
    ##########################################################################

    # Print comprehensive dataset statistics
    print("Number of training data %i." % len(ds_train))
    print("Number of testing data %i." % len(ds_test))
    if params["use_validation"]:
        print("Number of validation data %i." % len(ds_validation))
    print("Number of outputs %i." % len(np.unique(labels)))
    print("Number of input channels %i." % len(params["selected_channels"]))
    print("Number of hidden neurons %i." % nb_hidden)
    print("Number of timesteps %i." % params["data_steps"])
    print("Delayed output ", params["delayed_output"])
    if params["no_synapse"]:
        print(f"No synapse dynamics.")
    if params["lower_bound"]:
        print(f"Clamp membrane voltage to: {params['lower_bound']}.")
    if params["use_linear_decay"]:
        print(f"Use linear decay.")
    else:
        print(f"Use exponential decay.")
    if params["ref_per_timesteps"]:
        print(
            f"Refractory period set to {params['ref_per_timesteps']} simulation timesteps.")
    print("Input duration %fs" % (params["data_steps"]*params["time_step"]))
    print("---------------------------\n")

    ##########################################################################
    # NETWORK TRAINING
    ##########################################################################

    # Build network architecture and train with specified algorithm
    loss_hist_epochs, acc_hist, best_layers, vars_eprop = build_and_train(
        params=params, ds_train=ds_train, ds_test=ds_test)

    ##########################################################################
    # MODEL EVALUATION
    ##########################################################################

    # Compute final accuracy and predictions on validation or test set
    if params["use_validation"]:
        val_acc, trues, preds = compute_classification_accuracy(
            dataset=ds_validation, layers=best_layers, params=params)
    else:
        val_acc, trues, preds = compute_classification_accuracy(
            dataset=ds_test, layers=best_layers, params=params)

    ##########################################################################
    # SAVE RESULTS
    ##########################################################################

    # Save trained model weights
    torch.save(
        best_layers, f'./model/best_model_{nb_hidden}_neurons_{str_letters}.pt')

    # Save training metrics and hyperparameters
    np.savez(results_file,
             acc_train=acc_hist[0],
             acc_test=acc_hist[1],
             loss_train=loss_hist_epochs,
             nb_hidden=nb_hidden)
    print(f"Results saved to {results_file}")

    ##########################################################################
    # GENERATE VISUALIZATIONS
    ##########################################################################

    # Plot training curves (loss and accuracy over epochs)
    plot_training_perfromance(
        path=f"./figures/best_model_{nb_hidden}_neurons_{str_letters}_training_performance",
        acc_train=np.array([acc_hist[0]]),
        acc_test=np.array([acc_hist[1]]),
        loss_train=np.array([loss_hist_epochs]))

    # Generate confusion matrix showing classification performance per letter
    plot_confusion_matrix(
        path=f"./figures/best_model_{nb_hidden}_neurons_{str_letters}_confusion_matrix",
        trues=trues,
        preds=preds,
        labels=letters)

    ##########################################################################
    # NETWORK ACTIVITY VISUALIZATION (Raster Plots)
    ##########################################################################

    # Extract spike activity from hidden and readout layers
    accs, spk_rec_readout_array, spk_rec_hidden_array = get_network_activity(
        dataset=ds_test, layers=best_layers, params=params)

    # Generate raster plots showing spike timing patterns
    layer_names = ["Hidden layer", "Readout layer"]
    total_nb_batches = len(accs)

    # Randomly select batches to visualize (or use all if fewer than requested)
    if NB_BATCHES_TO_PLOT > total_nb_batches:
        batch_selection = range(total_nb_batches)
    else:
        batch_selection = np.random.choice(
            total_nb_batches, NB_BATCHES_TO_PLOT, replace=False)

    # Generate raster plots for selected batches and trials
    for batch_idx in batch_selection:
        spk_rec_readout_batch = spk_rec_readout_array[batch_idx]
        spk_rec_hidden_batch = spk_rec_hidden_array[batch_idx]

        total_nb_trials = len(spk_rec_readout_batch)

        # Randomly select trials to visualize within each batch
        if NB_TRIALS_TO_PLOT > total_nb_trials:
            trial_selection = range(total_nb_trials)
        else:
            trial_selection = np.random.choice(
                total_nb_trials, NB_TRIALS_TO_PLOT, replace=False)

        # Create raster plot for each selected trial
        for trial_idx in trial_selection:
            spr_recs = [spk_rec_hidden_batch[trial_idx],
                        spk_rec_readout_batch[trial_idx]]
            plot_network_activity(spr_recs=spr_recs, layer_names=layer_names, params=params,
                                  figname=f"./figures/best_model_{nb_hidden}_neurons_{str_letters}_network_activity_batch_{batch_idx}_trial_{trial_idx}")

    ##########################################################################
    # CLEANUP AND COMPLETION
    ##########################################################################

    # Free GPU memory
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Training complete! Results saved to {results_file}")
    print(f"{'='*60}")
