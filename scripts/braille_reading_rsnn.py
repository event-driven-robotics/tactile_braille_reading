"""braille_reading_rsnn.py

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
    python braille_reading_rsnn.py

Train with e-prop on specific letters:
    python braille_reading_rsnn.py --eprop --letters A B C

Custom architecture with validation set:
    python braille_reading_rsnn.py --nb_hidden 100 --validation

Select specific tactile sensors:
    python braille_reading_rsnn.py --selected_channels 0 1 2 5 8

Run inference-only evaluation from a resumed model (no training):
    python braille_reading_rsnn.py --resume-run-id 20260130_1151_exploration --inference-only

Equivalent forms also supported:
    python braille_reading_rsnn.py --resume_training 20260130_1151_exploration --inference_only=true
    python braille_reading_rsnn.py --resume_training 20260130_1151_exploration --inference_only=false

Resume existing training from a previous run folder (auto-picks newest best_model_*.pt):
    python braille_reading_rsnn.py --resume-run-id 20260130_1151_exploration

Resume from a specific checkpoint file in that run folder:
    python braille_reading_rsnn.py --resume-run-id 20260130_1151_exploration --resume-model-name best_model_50_neurons_A_B_rep_1.pt

Resume behavior and CLI overrides:
    - When --resume-run-id is used, parameters are loaded from:
      results/<run_id>/experiment_parameters.json
        - Loaded parameters are merged with CLI arguments.
        - Any parameter explicitly passed on the CLI overrides the loaded value.
    - --resume-model-name only selects which checkpoint to load; it does not alter
      the loaded experiment hyperparameters.

Author: Simon F. Muller-Cleve
Date: January 15, 2026
"""
import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_and_extract
from utils.train_snn import build_and_train, load_weights_from_model
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
    def _str2bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {'true', 't', '1', 'yes', 'y'}:
            return True
        if value in {'false', 'f', '0', 'no', 'n'}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

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
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Path to save log files')
    parser.add_argument('--input_data_path', type=str, default='./data/100Hz/',
                        help='Path to input data files')

    # Training parameters
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode: automatically sets log_level to DEBUG for detailed diagnostics')
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
    parser.add_argument('--ref_per_ms', type=float, default=3.0,
                        help='Refractory period in milliseconds (converted to timesteps using time_bin_size)')
    parser.add_argument('--ref_per_timesteps', type=int, default=None,
                        help='[Deprecated] Refractory period in timesteps. If set, it overrides --ref_per_ms.')
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
    parser.add_argument('--soft_reset', action='store_true', default=False,
                        help='Use soft reset (subtract threshold after spike) instead of hard reset to zero')

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
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (overridden by --debug flag): DEBUG for diagnostics, INFO normal, WARNING/ERROR for minimal output')
    parser.add_argument('--resume-run-id', '--resume-training', '--resume_training',
                        dest='resume_run_id', type=str, default='',
                        help='Run ID to resume from (loads model and params from model/results). '
                        'When resuming, explicitly provided CLI arguments automatically override '
                        'loaded experiment parameters; aliases: --resume-training, --resume_training')
    parser.add_argument('--resume-model-name', type=str, default='',
                        help='Optional best_model filename to load within the run_id model folder')
    parser.add_argument('--inference-only', '--inference_only',
                        dest='inference_only', nargs='?', const=True, default=False, type=_str2bool,
                        help='Skip training and only run evaluation/plots using a resumed model (requires --resume-run-id). Supports forms like --inference-only, --inference_only=true, --inference_only=false')

    args = parser.parse_args()

    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_cli_dests = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest is not None:
            explicit_cli_dests.add(dest)

    # Compute derived parameters
    if args.mechanoreceptor_encoding:
        args.max_time = 3700
    else:
        args.max_time = 3501

    # Handle synapse flag inversion (command line uses --synapse to ENABLE, params dict stores as synapse)
    # synapse default is False (disabled), which matches the old no_synapse default of True (disabled)
    # Convert to the new naming convention

    # Convert to dict for backward compatibility
    args_dict = vars(args)
    args_dict["_explicit_cli_dests"] = sorted(explicit_cli_dests)
    return args_dict


def _resolve_resume_paths(resume_run_id, params):
    model_dir = Path(params["model_path"]) / resume_run_id
    results_dir = Path(params["results_path"]) / resume_run_id

    if params.get("resume_model_name"):
        resume_from = model_dir / params["resume_model_name"]
        if not resume_from.is_file():
            raise FileNotFoundError(f"Resume model not found: {resume_from}")
    else:
        candidates = sorted(model_dir.glob("best_model_*.pt"))
        if len(candidates) == 0:
            raise FileNotFoundError(f"No best_model_*.pt found in {model_dir}")
        if len(candidates) > 1:
            candidates = sorted(
                candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            print(
                f"Multiple best_model_*.pt files found in {model_dir}; "
                f"using newest: {candidates[0].name}")

        resume_from = candidates[0]

    resume_from = str(resume_from)
    resume_params_path = results_dir / "experiment_parameters.json"
    if not resume_params_path.is_file():
        raise FileNotFoundError(
            f"Resume parameters file not found: {resume_params_path}")
    return resume_from, str(resume_params_path)


def _resolve_refractory_params(params):
    """Resolve refractory period consistently from ms and timestep settings.

    Priority:
    1) If ref_per_timesteps is explicitly set, use it and derive ref_per_ms.
    2) Otherwise, derive ref_per_timesteps from ref_per_ms and time_bin_size.
    """
    time_bin_size_ms = float(params.get("time_bin_size", 1))
    if time_bin_size_ms <= 0.0:
        raise ValueError(
            f"time_bin_size must be > 0 ms, got {time_bin_size_ms}")

    explicit_cli_dests = set(params.get("_explicit_cli_dests", []))
    ref_steps_explicit = "ref_per_timesteps" in explicit_cli_dests
    ref_ms_explicit = "ref_per_ms" in explicit_cli_dests

    use_ref_steps = params.get("ref_per_timesteps") is not None
    if ref_ms_explicit and not ref_steps_explicit:
        # If ref_per_ms was explicitly provided on CLI (resume case),
        # derive timesteps from ms even if loaded params contain ref_per_timesteps.
        use_ref_steps = False

    if use_ref_steps:
        ref_steps = int(params["ref_per_timesteps"])
        if ref_steps <= 0:
            params["ref_per_timesteps"] = None
            params["ref_per_ms"] = 0.0
        else:
            params["ref_per_timesteps"] = ref_steps
            params["ref_per_ms"] = ref_steps * time_bin_size_ms
        return

    ref_ms = float(params.get("ref_per_ms", 3.0))
    if ref_ms <= 0.0:
        params["ref_per_timesteps"] = None
        params["ref_per_ms"] = 0.0
    else:
        ref_steps = max(1, int(math.ceil(ref_ms / time_bin_size_ms)))
        params["ref_per_timesteps"] = ref_steps
        params["ref_per_ms"] = ref_ms


def _format_timestep_count(n_steps):
    """Return human-readable timestep count with correct singular/plural wording."""
    step_count = int(n_steps)
    unit = "timestep" if step_count == 1 else "timesteps"
    return f"{step_count} {unit}"


def _prepare_layers_for_inference(resume_path, params):
    """Load saved layers and align tensor/device settings for inference."""
    layers = torch.load(resume_path, map_location=params['device'])

    for layer in layers:
        layer.device = params['device']
        layer.dtype = params['dtype_torch']

        # Backward compatibility for legacy serialized layers
        if not hasattr(layer, 'spike_threshold'):
            layer.spike_threshold = params.get('spike_threshold', 1.0)
        if not hasattr(layer, 'gamma'):
            layer.gamma = params.get('gamma', 15.0)
        if not hasattr(layer, 'eprop'):
            layer.eprop = params.get('eprop', False)
        if not hasattr(layer, 'linear_decay'):
            layer.linear_decay = params.get('linear_decay', False)
        if not hasattr(layer, 'ref_per'):
            layer.ref_per = params.get('ref_per_timesteps')
        if not hasattr(layer, 'soft_reset'):
            layer.soft_reset = params.get('soft_reset', False)

        if hasattr(layer, 'ff_weights'):
            layer.ff_weights = layer.ff_weights.to(
                device=params['device'], dtype=params['dtype_torch'])
        if hasattr(layer, 'rec_weights'):
            layer.rec_weights = layer.rec_weights.to(
                device=params['device'], dtype=params['dtype_torch'])
        if layer.ref_per is not None and layer.ref_per > 0 and not hasattr(layer, 'ref_per_tensor'):
            layer.ref_per_tensor = torch.zeros(
                (params['batch_size'], layer.nb_neurons), device=params['device'], dtype=torch.int)
        if hasattr(layer, 'ref_per_tensor') and layer.ref_per_tensor is not None:
            layer.ref_per_tensor = layer.ref_per_tensor.to(
                device=params['device'])

    return layers


def setup_logger(log_dir, run_id, log_level='INFO'):
    """
    Configure logging to write to both console and file.

    Parameters
    ----------
    log_dir : str
        Directory where log file will be saved
    run_id : str
        Timestamp identifier for this run
    log_level : str, optional
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'INFO')

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('braille_training')
    log_level_enum = getattr(logging, log_level.upper())
    logger.setLevel(log_level_enum)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # File handler - save to logs directory
    log_file = os.path.join(log_dir, f'training_log_{run_id}.txt')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level_enum)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - print to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_enum)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


##############################################################################
# CONFIGURATION SETUP
##############################################################################

# Parse all command-line arguments into params dictionary
cli_params = parse_arguments()
params = cli_params

if params.get("resume_run_id"):
    try:
        resume_from, resume_params_path = _resolve_resume_paths(
            params["resume_run_id"], params)
    except Exception as exc:
        print(f"Failed to resolve resume paths: {exc}")
        sys.exit(1)

    print(f"Resume run_id: {params['resume_run_id']}")
    print(f"Resume model path: {resume_from}")
    print(f"Resume params path: {resume_params_path}")

    try:
        with open(resume_params_path, "r") as handle:
            loaded_params = json.load(handle)
    except Exception as exc:
        print(f"Failed to load resume params: {exc}")
        sys.exit(1)

    print("Overriding CLI parameters with loaded experiment parameters.")
    print(json.dumps(loaded_params, indent=2, sort_keys=True))

    merged_params = dict(cli_params)
    merged_params.update(loaded_params)

    # Always preserve current CLI control flags for this invocation
    merged_params["resume_run_id"] = cli_params.get("resume_run_id", "")
    merged_params["resume_model_name"] = cli_params.get(
        "resume_model_name", "")
    merged_params["inference_only"] = cli_params.get("inference_only", False)
    merged_params["_explicit_cli_dests"] = cli_params.get(
        "_explicit_cli_dests", [])

    legacy_map = {
        "use_cuda": "cuda",
        "use_seed": "seed",
        "use_validation": "validation",
        "use_eprop": "eprop",
        "use_linear_decay": "linear_decay",
        "use_soft_reset": "soft_reset",
        "use_mechanoreceptor_encoding": "mechanoreceptor_encoding",
        "use_random_tie_breaking": "random_tie_breaking",
        "use_weight_quantization": "quantize_weights",
        "no_synapse": "synapse",
    }
    for legacy_key, current_key in legacy_map.items():
        if legacy_key in merged_params and current_key not in merged_params:
            if legacy_key == "no_synapse":
                merged_params[current_key] = not bool(
                    merged_params[legacy_key])
            else:
                merged_params[current_key] = merged_params[legacy_key]

    params = merged_params
    params["resume_from"] = resume_from
    params["resume_params"] = resume_params_path

    auto_override_exclusions = {
        "resume_run_id", "resume_model_name", "inference_only", "_explicit_cli_dests"
    }
    explicit_override_keys = [
        key for key in cli_params.get("_explicit_cli_dests", [])
        if key not in auto_override_exclusions
    ]

    if explicit_override_keys:
        print("Applying explicit CLI overrides while resuming:")
        for key in explicit_override_keys:
            if key not in cli_params:
                continue
            params[key] = cli_params[key]
            print(f"  - {key}: {params[key]}")

if params.get('inference_only', False) and not params.get("resume_run_id"):
    print("--inference-only requires --resume-run-id to load an existing model.")
    sys.exit(1)

# Resolve refractory settings after all parameter merging/overrides
_resolve_refractory_params(params)

# If --debug flag is set, automatically use DEBUG log level
if params.get('debug', False):
    params['log_level'] = 'DEBUG'

# Configure letter set for classification
letters = params['letters']
all_letters = (len(letters) == 27)  # True if using complete alphabet + space

# Configure PyTorch data type for all tensors
dtype_map = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64
}
params['dtype_torch'] = dtype_map[params['dtype']]
torch.set_default_dtype(params['dtype_torch'])

##############################################################################
# VISUALIZATION AND OUTPUT CONFIGURATION
##############################################################################

# Control how many network activity plots to generate
NB_BATCHES_TO_PLOT = 1  # Number of batches to visualize
NB_TRIALS_TO_PLOT = 1   # Number of trials per batch to visualize

# Create all necessary output directories
for dir in [params['fig_path'], params['model_path'], params['results_path'], params['log_path']]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create timestamped subfolders for this experiment run
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
figures_dir = os.path.join(params['fig_path'], run_id)
models_dir = os.path.join(params['model_path'], run_id)
results_dir = os.path.join(params['results_path'], run_id)
logs_dir = os.path.join(params['log_path'], run_id)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Initialize logger (must be done after logs_dir is created)
logger = setup_logger(logs_dir, run_id, params['log_level'])
logger.info(f"="*80)
logger.info(f"Braille Reading RSNN Training - Run ID: {run_id}")
logger.info(f"="*80)
logger.info(
    f"Log file: {os.path.join(logs_dir, f'training_log_{run_id}.txt')}")
logger.info(f"Log level: {params['log_level']}")
logger.info(f"Using letters: {', '.join(letters)}")
logger.info(f"Using torch dtype: {params['dtype']}")
time_bin_size_ms = float(params['time_bin_size'])
logger.info(
    f"Time bin size: {time_bin_size_ms:g} ms ({time_bin_size_ms * 0.001:g} s)")
if params['ref_per_timesteps'] is None:
    logger.info("Refractory period: disabled (0.0 ms, 0 timesteps)")
else:
    effective_ref_ms = params['ref_per_timesteps'] * time_bin_size_ms
    ref_steps_text = _format_timestep_count(params['ref_per_timesteps'])
    logger.info(
        f"Refractory period configured: {params['ref_per_ms']:g} ms")
    logger.info(
        f"Refractory period resolved: {ref_steps_text} ({effective_ref_ms:g} ms effective)")
logger.info(f"Logs directory: {logs_dir}")
logger.info(f"Results directory: {results_dir}")
logger.info(f"Figures directory: {figures_dir}")
logger.info(f"Models directory: {models_dir}")

if params.get('resume_run_id') and params.get('inference_only', False):
    run_mode = 'Resume + Inference Only'
elif params.get('resume_run_id'):
    run_mode = 'Resume + Training'
else:
    run_mode = 'Fresh Training'

logger.info(f"Run mode: {run_mode}")
logger.info(f"Inference only: {params.get('inference_only', False)}")
if params.get('resume_run_id'):
    logger.info(f"Resume run ID: {params['resume_run_id']}")
if params.get("resume_params"):
    logger.info(f"Resume parameters: {params['resume_params']}")
if params.get("resume_from"):
    logger.info(f"Resume model: {params['resume_from']}")


##############################################################################
# DEVICE CONFIGURATION
##############################################################################

# Select computation device (GPU if available and requested, otherwise CPU)
if params["cuda"] and torch.cuda.is_available():
    params['device'] = torch.device("cuda:0")
    logger.info("Using CUDA for computation.")
elif params["cuda"] and not torch.cuda.is_available():
    params['device'] = torch.device("cpu")
    logger.warning("CUDA requested but not available. Falling back to CPU computation. "
                   "Training will be significantly slower.")
else:
    params['device'] = torch.device("cpu")
    logger.info("Using CPU for computation.")

##############################################################################
# WEIGHT QUANTIZATION SETUP (Optional)
##############################################################################

# Configure weight quantization for neuromorphic hardware deployment
if params['quantize_weights']:
    logger.info("Using weight quantization.")
    # Generate a symmetric signed quantization grid with exact zero.
    # We intentionally use -127..127 (255 levels) so the range is symmetric
    # and includes 0 exactly.
    signed_levels = torch.arange(-127, 128, device=params['device'])
    q = 1 / 127
    possible_weights = signed_levels * q

    # Round to 3 decimal places for hardware precision using proper rounding
    # (instead of floor truncation, which introduces a one-sided bias).
    factor = 10 ** 3
    params['possible_weights'] = torch.round(
        possible_weights * factor) / factor
    quant_step = float((params['possible_weights'][1] - params['possible_weights'][0]).item())
    quant_min = float(params['possible_weights'].min().item())
    quant_max = float(params['possible_weights'].max().item())
    logger.info(
        "Quantization grid: symmetric signed with exact zero | "
        f"levels={params['possible_weights'].numel()} | "
        f"range=[{quant_min:.3f}, {quant_max:.3f}] | "
        f"step={quant_step:.3f}"
    )

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
    logger.info("Seed set to {}".format(seed))
else:
    logger.info("Shuffle data randomly")

# Save experiment parameters to JSON for reproducibility
params_to_save = params.copy()
# Convert non-serializable objects to string representations
params_to_save['device'] = str(params['device'])
params_to_save['dtype_torch'] = str(params['dtype_torch'])
params_to_save.pop('_explicit_cli_dests', None)
if 'possible_weights' in params_to_save:
    params_to_save['possible_weights'] = 'Quantized weight array (symmetric -127..127, 255 levels incl. 0)'
params_to_save['run_id'] = run_id
params_to_save['timestamp'] = datetime.now().isoformat()

params_file = os.path.join(results_dir, 'experiment_parameters.json')
try:
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=4, sort_keys=True)
    logger.info(f"Parameters saved to {params_file}")
except Exception as e:
    logger.error(f"Failed to save parameters to {params_file}: {str(e)}")

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
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Training with: {'e-prop' if params['eprop'] else 'BPTT (Backpropagation Through Time)'}")
    logger.info(f"{'='*60}\n")

    # Generate descriptive string for output files based on letter set
    if all_letters:
        str_letters = 'all_letters'
    else:
        str_letters = '_'.join(letters)

    nb_hidden = params['nb_hidden']

    loss_hist_repetition = []
    accs_hist_repetition = []
    eval_acc_repetition = []

    resume_weights = None
    inference_layers = None
    if params.get("resume_from"):
        resume_path = params["resume_from"]
        if not os.path.isfile(resume_path):
            logger.error(f"Resume file not found: {resume_path}")
            sys.exit(1)
        try:
            resume_weights = load_weights_from_model(
                resume_path, map_location="cpu")
            logger.info(f"Resuming from model: {resume_path}")
        except Exception as e:
            logger.error(f"Failed to load resume model: {resume_path}")
            logger.error(str(e))
            sys.exit(1)

        if params.get('inference_only', False):
            try:
                inference_layers = _prepare_layers_for_inference(
                    resume_path, params)
                logger.info(
                    "Inference-only mode enabled: loaded resumed model for evaluation.")
            except Exception as e:
                logger.error(
                    f"Failed to load model layers for inference from: {resume_path}")
                logger.error(str(e))
                sys.exit(1)

    for repetition in range(params['repetitions']):
        logger.info(
            f"\n{'#'*20} Starting repetition {repetition + 1} of {params['repetitions']} {'#'*20}\n")

        ##########################################################################
        # DATA LOADING AND PREPROCESSING
        ##########################################################################

        # Load and preprocess tactile sensor data
        # This also computes and stores in params:
        #   - 'data_steps': Number of simulation timesteps
        #   - 'time_step': Duration of each simulation timestep (seconds)
        #   - 'delayed_output': Optional parameter for e-prop (final timesteps used for gradient computation)
        try:
            ds_train, ds_test, ds_validation, labels = load_and_extract(
                params=params, file_name=file_name, letter_written=letters)
        except Exception as e:
            logger.error(f"Failed to load data from {file_name}: {str(e)}")
            logger.error(f"Skipping repetition {repetition + 1}")
            continue

        dataset_steps = None
        if len(ds_train) > 0:
            dataset_steps = int(ds_train.tensors[0].shape[1])
        elif len(ds_test) > 0:
            dataset_steps = int(ds_test.tensors[0].shape[1])
        elif params["validation"] and ds_validation is not None and len(ds_validation) > 0:
            dataset_steps = int(ds_validation.tensors[0].shape[1])

        if dataset_steps is None:
            logger.error(
                "No samples available to derive simulation timesteps from prepared dataset.")
            logger.error(f"Skipping repetition {repetition + 1}")
            continue

        if params.get("data_steps") != dataset_steps:
            logger.warning(
                f"data_steps mismatch: params has {params.get('data_steps')}, "
                f"prepared dataset has {dataset_steps}. Using dataset-derived value.")
            params["data_steps"] = dataset_steps

        # Clear GPU cache before training
        torch.cuda.empty_cache()

        ##########################################################################
        # DATASET INFORMATION
        ##########################################################################

        # Print comprehensive dataset statistics
        logger.info("Number of training samples %i." % len(ds_train))
        logger.info("Number of testing samples %i." % len(ds_test))
        if params["validation"]:
            logger.info("Number of validation samples %i." %
                        len(ds_validation))
        logger.info("Number of output classes %i." % len(np.unique(labels)))
        logger.info("Number of input channels %i." %
                    len(params["selected_channels"]))
        logger.info("Number of hidden neurons %i." % nb_hidden)
        logger.info("Simulation time step %.3f ms (%.6f s)." %
                    (params["time_step"] * 1000.0, params["time_step"]))
        logger.info(
            "Simulation timesteps (derived from prepared dataset): %i." % params["data_steps"])
        logger.info("Delayed output %s" % params["delayed_output"])

        expected_time_step_s = float(params["time_bin_size"]) * 0.001
        if not np.isclose(params["time_step"], expected_time_step_s):
            logger.warning(
                f"Inconsistent timestep conversion: time_step={params['time_step']:.6f}s "
                f"but expected time_bin_size*0.001={expected_time_step_s:.6f}s")

        # Warn if dataset is very small
        if len(ds_train) < params['batch_size']:
            logger.warning(f"Training set size ({len(ds_train)}) is smaller than batch size ({params['batch_size']}). "
                           "This may cause training issues.")
        if len(ds_test) < 10:
            logger.warning(f"Test set size ({len(ds_test)}) is very small. "
                           "Results may not be statistically significant.")
        if not params["synapse"]:
            logger.info(f"No synaptic dynamics.")
        if params["lower_bound"]:
            logger.info(f"Clamp membrane voltage to: {params['lower_bound']}.")
        if params["linear_decay"]:
            logger.info(f"Use linear decay.")
        else:
            logger.info(f"Use exponential decay.")
        if params["ref_per_timesteps"]:
            ref_steps_text = _format_timestep_count(
                params['ref_per_timesteps'])
            logger.info(
                f"Refractory period set to {ref_steps_text} "
                f"({params['ref_per_ms']:g} ms target, "
                f"{params['ref_per_timesteps'] * float(params['time_bin_size']):g} ms effective).")
        else:
            logger.info("Refractory period disabled (0.0 ms, 0 timesteps).")
        total_duration_s = params["data_steps"] * params["time_step"]
        total_duration_ms = total_duration_s * 1000.0
        logger.info(
            "Simulation duration %.3f ms (%.6f s)." %
            (total_duration_ms, total_duration_s))
        logger.info("---------------------------\n")

        ##########################################################################
        # NETWORK TRAINING
        ##########################################################################

        # Build network architecture and train with specified algorithm
        # Note: params dict is modified in-place by build_and_train() to include:
        #   - 'beta_trace': Eligibility trace decay for hidden layer (e-prop only)
        #   - 'beta_trace_out': Eligibility trace decay for output layer (e-prop only)
        loss_hist_epochs = []
        accs_hist = [[], []]
        initial_weights: dict[str, np.ndarray] | None = None

        if params.get('inference_only', False):
            if inference_layers is None:
                logger.error(
                    "Inference-only mode requires a loaded resume model.")
                logger.error(f"Skipping repetition {repetition + 1}")
                continue
            best_layers = inference_layers
            logger.info(
                "Inference-only mode: skipping training and running evaluation only.")
        else:
            try:
                loss_hist_epochs, accs_hist, best_layers, vars_eprop, initial_weights = build_and_train(
                    params=params, ds_train=ds_train, ds_test=ds_test, resume_weights=resume_weights)
            except Exception as e:
                logger.error(
                    f"Training failed for repetition {repetition + 1}: {str(e)}")
                logger.error(f"Skipping this repetition and continuing...")
                continue

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
        eval_acc_repetition.append(val_acc)

        # Warn if final accuracy is low
        num_classes = len(params['letters'])
        chance_level = 1.0 / num_classes
        if val_acc < (chance_level + 0.10):  # Less than 10% above chance
            if params.get('inference_only', False):
                logger.warning(
                    f"Final accuracy ({val_acc*100:.2f}%) is near chance level ({chance_level*100:.2f}%).")
            else:
                logger.warning(f"Final accuracy ({val_acc*100:.2f}%) is near chance level ({chance_level*100:.2f}%). "
                               f"Poor weight initialization or insufficient training.")

        ##########################################################################
        # SAVE RESULTS
        ##########################################################################

        if not params.get('inference_only', False):
            # Save initial weights (at initialization, before training)
            try:
                if initial_weights is None:
                    raise ValueError(
                        "initial_weights are missing after training.")
                np.savez(
                    os.path.join(
                        models_dir, f'initial_weights_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.npz'),
                    **initial_weights,
                )
            except Exception as e:
                logger.error(f"Failed to save initial weights: {str(e)}")

            # Save trained model weights (full layer objects for evaluation)
            try:
                torch.save(
                    best_layers,
                    os.path.join(
                        models_dir,
                        f'best_model_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.pt'))
            except Exception as e:
                logger.error(f"Failed to save trained model: {str(e)}")

            # Save best-model weights as numpy arrays for easy analysis
            try:
                from utils.train_snn import save_weights
                best_model_weights = save_weights(best_layers)
                np.savez(os.path.join(models_dir, f'best_model_weights_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.npz'),
                         **best_model_weights)
            except Exception as e:
                logger.error(f"Failed to save best-model weights: {str(e)}")

        # Results path for this repetition
        results_file = os.path.join(
            results_dir,
            f"braille_reading_rsnn_{nb_hidden}_neurons_{str_letters}_rep_{repetition+1}.npz")

        ##########################################################################
        # GENERATE VISUALIZATIONS
        ##########################################################################

        if not params.get('inference_only', False):
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

        # Save training metrics, hyperparameters, and full spike recordings
        # Concatenate per-batch arrays to avoid object arrays and keep load path simple.
        try:
            spk_rec_readout = np.concatenate(spk_rec_readout_array, axis=0)
            spk_rec_hidden = np.concatenate(spk_rec_hidden_array, axis=0)
            trues_array = np.concatenate(trues, axis=0)
            preds_array = np.concatenate(preds, axis=0)
            network_input = ds_test.tensors[0].detach().cpu().numpy()
            network_input_labels = ds_test.tensors[1].detach().cpu().numpy()

            np.savez_compressed(results_file,
                                acc_train=accs_hist[0],
                                acc_test=accs_hist[1],
                                loss_train=loss_hist_epochs,
                                val_acc=val_acc,
                                trues=trues_array,
                                preds=preds_array,
                                repetition=repetition + 1,
                                nb_hidden=nb_hidden,
                                nb_epochs=params['epochs'],
                                learning_rate=params['learning_rate'],
                                batch_size=params['batch_size'],
                                letters=str_letters,
                                eprop=params['eprop'],
                                run_id=run_id,
                                network_input=network_input,
                                network_input_labels=network_input_labels,
                                spk_rec_hidden=spk_rec_hidden,
                                spk_rec_readout=spk_rec_readout)

            logger.info(
                f"Results (including spike recordings) saved to {results_file} "
                f"[input={network_input.shape}, hidden={spk_rec_hidden.shape}, readout={spk_rec_readout.shape}]")
        except Exception as e:
            logger.error(f"Failed to save results to {results_file}: {str(e)}")

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

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Repetition {repetition + 1}/{params['repetitions']} complete! Results saved to {results_file}")
        logger.info(f"{'='*60}")

    # Plot training curves (loss and accuracy over epochs)
    if not params.get('inference_only', False):
        if len(accs_hist_repetition) == 0 or len(loss_hist_repetition) == 0:
            logger.warning(
                "No successful repetitions completed; skipping summary performance plot.")
        else:
            plot_training_performance_repetitive_runs(
                path=os.path.join(
                    figures_dir, f"{nb_hidden}_neurons_{str_letters}_training_performance_{params['repetitions']}_rep"),
                acc_train=np.array(accs_hist_repetition)[:, 0],
                acc_test=np.array(accs_hist_repetition)[:, 1],
                loss_train=np.array(loss_hist_repetition))

    ##########################################################################
    # FINAL PERFORMANCE REPORT
    ##########################################################################

    logger.info(f"\n\n{'='*80}")
    logger.info(f"FINAL PERFORMANCE REPORT")
    logger.info(f"{'='*80}\n")

    # Experiment configuration summary
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Run ID: {run_id}")
    logger.info(
        f"  Learning Algorithm: {'e-prop' if params['eprop'] else 'BPTT'}")
    logger.info(f"  Letters: {str_letters} ({len(params['letters'])} classes)")
    logger.info(f"  Hidden Neurons: {nb_hidden}")
    logger.info(f"  Training Epochs: {params['epochs']}")
    logger.info(f"  Batch Size: {params['batch_size']}")
    logger.info(f"  Learning Rate: {params['learning_rate']}")
    logger.info(f"  Device: {params['device']}")
    logger.info("")

    # Performance across repetitions
    if params.get('inference_only', False):
        if len(eval_acc_repetition) > 0:
            eval_accs = np.array(eval_acc_repetition)
            num_classes = len(params['letters'])
            chance_level = 1.0 / num_classes

            logger.info(
                f"Inference-only Summary ({len(eval_acc_repetition)} successful repetitions):")
            logger.info("")
            logger.info(f"  Evaluation Accuracy:")
            logger.info(
                f"    Mean: {np.mean(eval_accs)*100:.2f}% ± {np.std(eval_accs)*100:.2f}%")
            logger.info(f"    Min:  {np.min(eval_accs)*100:.2f}%")
            logger.info(f"    Max:  {np.max(eval_accs)*100:.2f}%")
            logger.info("")
            logger.info(
                f"  Chance Level: {chance_level*100:.2f}% (1/{num_classes} classes)")
            logger.info(
                f"  Improvement over chance (mean eval): {(np.mean(eval_accs) - chance_level)*100:.2f} percentage points")
            logger.info("")
        else:
            logger.warning(f"  No successful repetitions to report.")
            logger.info(f"")
    elif len(accs_hist_repetition) > 0:
        accs_array = np.array(accs_hist_repetition)
        loss_array = np.array(loss_hist_repetition)

        # Extract final and best accuracies for each repetition
        final_train_accs = accs_array[:, 0, -1]  # Last epoch train accuracy
        final_test_accs = accs_array[:, 1, -1]   # Last epoch test accuracy
        best_train_accs = np.max(
            accs_array[:, 0, :], axis=1)  # Best train accuracy
        best_test_accs = np.max(
            accs_array[:, 1, :], axis=1)   # Best test accuracy
        final_losses = loss_array[:, -1]  # Last epoch loss

        logger.info(
            f"Performance Summary ({len(accs_hist_repetition)} successful repetitions):")
        logger.info(f"")
        logger.info(f"  Training Accuracy (final epoch):")
        logger.info(
            f"    Mean: {np.mean(final_train_accs)*100:.2f}% ± {np.std(final_train_accs)*100:.2f}%")
        logger.info(f"    Min:  {np.min(final_train_accs)*100:.2f}%")
        logger.info(f"    Max:  {np.max(final_train_accs)*100:.2f}%")
        logger.info(f"")
        logger.info(f"  Test Accuracy (final epoch):")
        logger.info(
            f"    Mean: {np.mean(final_test_accs)*100:.2f}% ± {np.std(final_test_accs)*100:.2f}%")
        logger.info(f"    Min:  {np.min(final_test_accs)*100:.2f}%")
        logger.info(f"    Max:  {np.max(final_test_accs)*100:.2f}%")
        logger.info(f"")
        logger.info(f"  Best Training Accuracy (across all epochs):")
        logger.info(
            f"    Mean: {np.mean(best_train_accs)*100:.2f}% ± {np.std(best_train_accs)*100:.2f}%")
        logger.info(f"    Min:  {np.min(best_train_accs)*100:.2f}%")
        logger.info(f"    Max:  {np.max(best_train_accs)*100:.2f}%")
        logger.info(f"")
        logger.info(f"  Best Test Accuracy (across all epochs):")
        logger.info(
            f"    Mean: {np.mean(best_test_accs)*100:.2f}% ± {np.std(best_test_accs)*100:.2f}%")
        logger.info(f"    Min:  {np.min(best_test_accs)*100:.2f}%")
        logger.info(f"    Max:  {np.max(best_test_accs)*100:.2f}%")
        logger.info(f"")
        logger.info(f"  Final Loss:")
        logger.info(
            f"    Mean: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
        logger.info(f"    Min:  {np.min(final_losses):.4f}")
        logger.info(f"    Max:  {np.max(final_losses):.4f}")
        logger.info(f"")

        # Chance level comparison
        num_classes = len(params['letters'])
        chance_level = 1.0 / num_classes
        logger.info(
            f"  Chance Level: {chance_level*100:.2f}% (1/{num_classes} classes)")
        logger.info(
            f"  Improvement over chance (mean test): {(np.mean(final_test_accs) - chance_level)*100:.2f} percentage points")
        logger.info(f"")
    else:
        logger.warning(f"  No successful repetitions to report.")
        logger.info(f"")

    # Output directories
    logger.info(f"Output Locations:")
    logger.info(f"  Results:  {results_dir}")
    logger.info(f"  Figures:  {figures_dir}")
    logger.info(f"  Models:   {models_dir}")
    logger.info(f"  Logs:     {logs_dir}")

    logger.info(f"\n{'='*80}")
    successful_repetitions = len(eval_acc_repetition) if params.get(
        'inference_only', False) else len(accs_hist_repetition)
    logger.info(
        f"ALL TRAINING COMPLETE - {params['repetitions']} repetitions requested, {successful_repetitions} successful")
    logger.info(f"{'='*80}\n")
