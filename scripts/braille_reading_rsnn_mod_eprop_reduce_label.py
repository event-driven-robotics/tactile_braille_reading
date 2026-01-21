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
Train with default settings (all letters, e-prop disabled, 450 neurons):
    python braille_reading_rsnn_mod_eprop_reduce_label.py

Train with e-prop on specific letters:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --eprop --letters A B C

Custom architecture with validation set:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --nb_hidden 100 --validation

Select specific tactile sensors:
    python braille_reading_rsnn_mod_eprop_reduce_label.py --selected_channels 0 1 2 5 8

Author: Simon F. Muller-Cleve
Date: January 15, 2026
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from utils.data_loader import load_and_extract
from utils.train_snn import build_and_train
from utils.validate_snn import (compute_classification_accuracy,
                                get_network_activity, plot_confusion_matrix,
                                plot_network_activity,
                                plot_training_performance,
                                plot_training_performance_repetitive_runs)

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
    # Path configuration
    parser.add_argument('--fig_path', type=str, default='./figures',
                        help='Path to save figures')
    parser.add_argument('--model_path', type=str, default='./model',
                        help='Path to save models')
    parser.add_argument('--results_path', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--input_data_path', type=str, default='./data/100Hz/',
                        help='Path to input data files')

    # Training parameters
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with verbose output')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA for GPU acceleration')
    parser.add_argument('--seed', action='store_true', default=False,
                        help='Use seed for reproducibility')
    parser.add_argument('--validation', action='store_true', default=False,
                        help='Create validation set from training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions of training runs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('--early_stop_epochs', type=int, default=0,
                        help='Number of initial epochs to check for improvement. '
                             'If training accuracy does not improve beyond random chance '
                             'within this many epochs, training stops early. '
                             'Set to 0 to disable early stopping. (default: 0)')
    parser.add_argument('--early_stop_threshold', type=float, default=5.0,
                        help='Percentage points above chance level required to continue training. '
                             'Chance level = 1/num_classes. For example, with 2 classes (50%% chance), '
                             'a threshold of 5.0 means training must reach 55%% accuracy. '
                             'With 26 classes (3.85%% chance), 5.0 means 8.85%% accuracy. '
                             'If training accuracy stays below this adaptive threshold for the first '
                             'early_stop_epochs, training is stopped. (default: 5.0)')

    # Network architecture
    parser.add_argument('--selected_channels', type=int, nargs='+', default=list(range(12)),
                        help='List of selected input channels (taxels)')
    parser.add_argument('--nb_hidden', type=int, default=450,
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
    parser.add_argument('--fwd_weight_scale', type=float, default=1.0,
                        help='Forward weight initialization scale')
    parser.add_argument('--weight_scale_factor', type=float, default=0.02,
                        help='Recurrent weight scale factor')

    # Regularization
    parser.add_argument('--reg_spikes', type=float, default=0.0015,
                        help='L1 regularization coefficient for spikes')
    parser.add_argument('--reg_neurons', type=float, default=0.001,
                        help='L2 regularization coefficient for neurons')

    # Learning algorithm
    parser.add_argument('--eprop', action='store_true', default=False,
                        help='Use e-prop instead of BPTT')
    parser.add_argument('--gamma', type=float, default=15.0,
                        help='Surrogate gradient scale factor')
    parser.add_argument('--spike_threshold', type=float, default=1.0,
                        help='Spike threshold for neurons (membrane potential - spike_threshold)')

    # Data parameters
    parser.add_argument('--letters', type=str, nargs='+',
                        default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                        help='List of letters to use for classification')
    parser.add_argument('--threshold', type=int, default=2, choices=[1, 2, 5, 10],
                        help='Threshold for data encoding')
    parser.add_argument('--time_bin_size', type=int, default=1,
                        help='Time bin size in milliseconds')
    parser.add_argument('--mechanoreceptor_encoding', action='store_true', default=True,
                        help='Use mechanoreceptor encoding for input (default: True, alternatives: sigma-delta encoding)')

    # Model options
    parser.add_argument('--synapse', action='store_true', default=False,
                        help='Enable synaptic dynamics (default: disabled)')
    parser.add_argument('--linear_decay', action='store_true', default=False,
                        help='Use linear decay instead of exponential')
    parser.add_argument('--quantize_weights', action='store_true', default=False,
                        help='Enable weight quantization')
    parser.add_argument('--random_tie_breaking', action='store_true', default=False,
                        help='Use random tie breaking for predictions')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float16', 'float32', 'float64'],
                        help='Torch data type for computations')

    args = parser.parse_args()

    # Compute derived parameters
    if args.mechanoreceptor_encoding:
        args.max_time = 3700
    else:
        args.max_time = 3501

    # Handle synapse flag inversion (command line uses --synapse to ENABLE, params dict stores as synapse)
    # synapse default is False (disabled), which matches the old no_synapse default of True (disabled)
    # Convert to the new naming convention

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

# Create all necessary output directories
for dir in [params['fig_path'], params['model_path'], params['results_path']]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create timestamped subfolders for this experiment run
run_id = datetime.now().strftime('%Y%m%d_%H%M')
figures_dir = os.path.join(params['fig_path'], run_id)
models_dir = os.path.join(params['model_path'], run_id)
results_dir = os.path.join(params['results_path'], run_id)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


##############################################################################
# DEVICE CONFIGURATION
##############################################################################

# Select computation device (GPU if available and requested, otherwise CPU)
if params["cuda"] and torch.cuda.is_available():
    params['device'] = torch.device("cuda:0")
    print("Using CUDA for computation.")
else:
    params['device'] = torch.device("cpu")
    print("Using CPU for computation.")

##############################################################################
# WEIGHT QUANTIZATION SETUP (Optional)
##############################################################################

# Configure weight quantization for neuromorphic hardware deployment
if params['quantize_weights']:
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
if params["seed"]:
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

# Save experiment parameters to JSON for reproducibility
params_to_save = params.copy()
# Convert non-serializable objects to string representations
params_to_save['device'] = str(params['device'])
params_to_save['dtype_torch'] = str(params['dtype_torch'])
if 'possible_weights' in params_to_save:
    params_to_save['possible_weights'] = 'Quantized weight array (256 levels)'
params_to_save['run_id'] = run_id
params_to_save['timestamp'] = datetime.now().isoformat()

params_file = os.path.join(results_dir, 'experiment_parameters.json')
with open(params_file, 'w') as f:
    json.dump(params_to_save, f, indent=4, sort_keys=True)
print(f"Parameters saved to {params_file}")

##############################################################################
# DATA FILE CONFIGURATION
##############################################################################

# Determine which data file to load based on encoding method
file_dir_data = params['input_data_path']
if params["mechanoreceptor_encoding"]:
    file_name = file_dir_data + 'mechanoreceptor_encoded.pkl'
else:
    # Use sigma-delta encoding with specified threshold
    file_type = 'data_braille_letters_100Hz_th'
    file_thr = str(params["threshold"])
    file_name = file_dir_data + file_type + file_thr + '.pkl'


if __name__ == '__main__':
    """
    Main training pipeline execution.

    This block orchestrates the complete training workflow:
    1. For each repetition:
       a. Load and preprocess tactile sensor data
       b. Build and train the SRNN (using either e-prop or BPTT)
       c. Evaluate on validation/test set and compute confusion matrix
       d. Save model weights and training metrics
       e. Generate visualizations (learning curves, confusion matrix, network activity)
    2. After all repetitions complete, generate aggregate performance plots

    Data flow:
    - Raw braille letter data (mechanoreceptor or sigma-delta encoded)
    - Pre-processed into train/test splits (with optional validation set)
    - Fed through SRNN in batches
    - Predictions compared to labels
    - Metrics and plots saved to timestamped results directory

    Output artifacts:
    - Trained model weights: model/<timestamp>/best_model_*.pt
    - Metrics and loss curves: results/<timestamp>/*.npz
    - Visualizations: figures/<timestamp>/*.png
    - Experiment configuration: results/<timestamp>/experiment_parameters.json
    """
    ##########################################################################
    # MAIN TRAINING PIPELINE
    ##########################################################################

    # Display learning algorithm configuration
    print(f"\n{'='*60}")
    print(
        f"Training with: {'e-prop' if params['eprop'] else 'BPTT (Backpropagation Through Time)'}")
    print(f"{'='*60}\n")

    # Generate descriptive string for output files based on letter set
    if all_letters:
        str_letters = 'all_letters'
    else:
        str_letters = '_'.join(letters)

    nb_hidden = params['nb_hidden']

    loss_hist_repetition = []
    accs_hist_repetition = []

    # Define paths for saving results and models
    results_file = os.path.join(
        results_dir, f"braille_reading_rsnn_{nb_hidden}_neurons_{str_letters}.npz")

    for repetition in range(params['repetitions']):
        print(
            f"\n{'#'*20} Starting repetition {repetition + 1} of {params['repetitions']} {'#'*20}\n")

        ##########################################################################
        # DATA LOADING AND PREPROCESSING
        ##########################################################################

        # Load and preprocess tactile sensor data
        # This also computes and stores in params:
        #   - 'data_steps': Number of simulation timesteps
        #   - 'time_step': Duration of each simulation timestep (seconds)
        #   - 'delayed_output': Optional parameter for e-prop (final timesteps used for gradient computation)
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
        if params["validation"]:
            print("Number of validation data %i." % len(ds_validation))
        print("Number of outputs %i." % len(np.unique(labels)))
        print("Number of input channels %i." %
              len(params["selected_channels"]))
        print("Number of hidden neurons %i." % nb_hidden)
        print("Number of timesteps %i." % params["data_steps"])
        print("Delayed output ", params["delayed_output"])
        if not params["synapse"]:
            print(f"No synaptic dynamics.")
        if params["lower_bound"]:
            print(f"Clamp membrane voltage to: {params['lower_bound']}.")
        if params["linear_decay"]:
            print(f"Use linear decay.")
        else:
            print(f"Use exponential decay.")
        if params["ref_per_timesteps"]:
            print(
                f"Refractory period set to {params['ref_per_timesteps']} simulation timesteps.")
        print("Input duration %fs" %
              (params["data_steps"]*params["time_step"]))
        print("---------------------------\n")

        ##########################################################################
        # NETWORK TRAINING
        ##########################################################################

        # Build network architecture and train with specified algorithm
        # Note: params dict is modified in-place by build_and_train() to include:
        #   - 'beta_trace': Eligibility trace decay for hidden layer (e-prop only)
        #   - 'beta_trace_out': Eligibility trace decay for output layer (e-prop only)
        loss_hist_epochs, accs_hist, best_layers, vars_eprop, initial_weights = build_and_train(
            params=params, ds_train=ds_train, ds_test=ds_test)

        loss_hist_repetition.append(loss_hist_epochs)
        accs_hist_repetition.append(accs_hist)

        ##########################################################################
        # MODEL EVALUATION
        ##########################################################################

        # Compute final accuracy and predictions on validation or test set
        # Use validation set if available (created from training set)
        # Otherwise use test set for evaluation
        if params["validation"]:
            val_acc, trues, preds = compute_classification_accuracy(
                dataset=ds_validation, layers=best_layers, params=params)
        else:
            val_acc, trues, preds = compute_classification_accuracy(
                dataset=ds_test, layers=best_layers, params=params)

        ##########################################################################
        # SAVE RESULTS
        ##########################################################################

        # Save initial weights (at initialization, before training)
        np.savez(os.path.join(models_dir, f'initial_weights_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.npz'),
                 **initial_weights)

        # Save trained model weights (full layer objects for evaluation)
        torch.save(
            best_layers, os.path.join(models_dir, f'best_model_{nb_hidden}_neurons_{str_letters}.pt'))
        
        # Save final weights as numpy arrays for easy analysis
        from utils.train_snn import save_weights
        final_weights = save_weights(best_layers)
        np.savez(os.path.join(models_dir, f'final_weights_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.npz'),
                 **final_weights)

        # Save training metrics and hyperparameters
        np.savez(results_file,
                 acc_train=accs_hist[0],
                 acc_test=accs_hist[1],
                 loss_train=loss_hist_epochs,
                 val_acc=val_acc,
                 repetition=repetition + 1,
                 nb_hidden=nb_hidden,
                 nb_epochs=params['epochs'],
                 learning_rate=params['learning_rate'],
                 batch_size=params['batch_size'],
                 letters=str_letters,
                 eprop=params['eprop'],
                 run_id=run_id)
        print(f"Results saved to {results_file}")

        ##########################################################################
        # GENERATE VISUALIZATIONS
        ##########################################################################

        # Plot training curves (loss and accuracy over epochs)
        plot_training_performance(
            path=os.path.join(
                figures_dir, f"{nb_hidden}_neurons_{str_letters}_training_performance_rep_{repetition+1}"),
            acc_train=np.array(accs_hist[0]),
            acc_test=np.array(accs_hist[1]),
            loss_train=np.array(loss_hist_epochs))

        # Generate confusion matrix showing classification performance per letter
        plot_confusion_matrix(
            path=os.path.join(
                figures_dir, f"best_model_{nb_hidden}_neurons_{str_letters}_confusion_matrix_rep_{repetition+1}"),
            trues=trues,
            preds=preds,
            labels=letters)

        ##########################################################################
        # NETWORK ACTIVITY VISUALIZATION (Raster Plots)
        ##########################################################################

        # Extract spike activity from hidden and readout layers
        spk_rec_readout_array, spk_rec_hidden_array = get_network_activity(
            dataset=ds_test, layers=best_layers, params=params)

        # Generate raster plots showing spike timing patterns
        layer_names = ["Hidden layer", "Readout layer"]
        total_nb_batches = len(spk_rec_readout_array)

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
                                      figname=os.path.join(figures_dir, f"best_model_{nb_hidden}_neurons_{str_letters}_network_activity_batch_{batch_idx}_trial_{trial_idx}_rep_{repetition+1}"))

        ##########################################################################
        # CLEANUP AND COMPLETION
        ##########################################################################

        # Free GPU memory
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"Training complete! Results saved to {results_file}")
        print(f"{'='*60}")

    # Plot training curves (loss and accuracy over epochs)
    plot_training_performance_repetitive_runs(
        path=os.path.join(
            figures_dir, f"{nb_hidden}_neurons_{str_letters}_training_performance_{params['repetitions']}_rep"),
        acc_train=np.array(accs_hist_repetition)[:, 0],
        acc_test=np.array(accs_hist_repetition)[:, 1],
        loss_train=np.array(loss_hist_repetition))
