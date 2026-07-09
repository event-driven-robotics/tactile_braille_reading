"""mechanoreceptor_encoding.py

Encoder script that processes raw tactile data through mechanoreceptor models.

Reads braille letter tactile data and generates FA-I and SA-II mechanoreceptor
spike responses using event-based neuron models. Outputs encoded spike data
for downstream neural network training.

Author: Simon F. Muller-Cleve
Date: January 12, 2026
"""

import pickle as pkl
import sys
import argparse
import inspect
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import numpy as np

# Ensure local package imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.neuron_models import RA_I_mechanoreceptor, SA_II_mechanoreceptor, AdExLIF_neuron, CuBaLIF_neuron, IZ_neuron, LIF_neuron, MN_neuron

ENCODING_CHOICES = ["mechanoreceptor", "sigma_delta", "neuron_model"]
NEURON_MODEL_CLASSES = {
    "AdExLIF_neuron": AdExLIF_neuron,
    "CuBaLIF_neuron": CuBaLIF_neuron,
    "IZ_neuron": IZ_neuron,
    "LIF_neuron": LIF_neuron,
    "MN_neuron": MN_neuron,
}
NEURON_MODEL_CHOICES = list(NEURON_MODEL_CLASSES)

# Options: "mechanoreceptor", "sigma_delta", "neuron_model"
ENCODING_TYPE = "neuron_model"
NEURON_MODEL = "MN_neuron"  # Options: "AdExLIF_neuron", "CuBaLIF_neuron", "IZ_neuron", "LIF_neuron", "MN_neuron"
UPSAMPLE_STRATEGY = "linear"  # Options: "linear", "hold"
UPSAMPLE_DT_S = 0.001  # Fixed target delta t in seconds (1 ms)
MECHANORECEPTOR_THRESHOLD = 2.0
MECHANORECEPTOR_MAX_FREQUENCY = 150.0
MECHANORECEPTOR_REFRACTORY_PERIOD = 0.003
SIGMA_DELTA_THRESHOLD = 2.0
SIGMA_DELTA_REFRACTORY_PERIOD = 0.003
NEURON_MODEL_PARAMS = {}
OUTPUT_PATH_OVERRIDE = None
SELECTED_LETTERS = None
NB_TRIALS = None

data_path = "data/100Hz"
data_files = ["data_braille_letters_0.0.pkl",
              "data_braille_letters_0.000125.pkl"]


def get_default_neuron_params(model_name: str) -> dict:
    """Return constructor defaults that users can tune for a neuron model."""
    model_class = NEURON_MODEL_CLASSES[model_name]
    signature = inspect.signature(model_class.__init__)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "nb_inputs", "device", "dt"}:
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        defaults[name] = parameter.default
    return defaults


def parse_typed_value(raw_value: str, default_value):
    """Convert a prompted/CLI string using the type of a default value."""
    if isinstance(default_value, bool):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got '{raw_value}'.")
    if isinstance(default_value, int):
        if any(marker in raw_value for marker in [".", "e", "E"]):
            return float(raw_value)
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    return raw_value


def parse_neuron_param_assignments(assignments: list[str], model_name: str) -> dict:
    """Parse repeated NAME=VALUE neuron parameter overrides."""
    defaults = get_default_neuron_params(model_name)
    parsed_params = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"Invalid --neuron-param '{assignment}'. Use NAME=VALUE.")
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if name == "dt":
            raise ValueError(
                "Neuron-model dt is controlled by --upsample-dt-s so data "
                "sampling and neuron dynamics stay coupled."
            )
        if name not in defaults:
            valid = ", ".join(defaults)
            raise ValueError(
                f"Unknown parameter '{name}' for {model_name}. Valid parameters: {valid}")
        parsed_params[name] = parse_typed_value(raw_value, defaults[name])
    return parsed_params


def prompt_text(prompt: str, default=None, prefill_default: bool = False) -> str:
    """Prompt for text, accepting Enter as the default."""
    if default is None:
        return input(f"{prompt}: ").strip()

    default_text = str(default)
    if default_text == "":
        return input(f"{prompt}: ").strip()

    if prefill_default and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            import readline

            def prefill():
                readline.insert_text(default_text)
                readline.redisplay()

            readline.set_startup_hook(prefill)
            try:
                return input(f"{prompt}: ").strip() or default_text
            finally:
                readline.set_startup_hook(None)
        except ImportError:
            pass

    value = input(f"{prompt} [{default_text}]: ").strip()
    return value or default_text


def prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt for a value from a fixed list."""
    while True:
        print(f"\n{prompt}")
        for index, choice in enumerate(choices, start=1):
            marker = " (default)" if choice == default else ""
            print(f"  {index}. {choice}{marker}")

        raw_value = input(f"Select 1-{len(choices)} or type a value [{default}]: ").strip()
        if not raw_value:
            return default
        if raw_value.isdigit():
            selected_index = int(raw_value)
            if 1 <= selected_index <= len(choices):
                return choices[selected_index - 1]
        if raw_value in choices:
            return raw_value
        print(f"Please choose one of: {', '.join(choices)}")


def prompt_float(prompt: str, default: float) -> float:
    """Prompt until a valid float is entered."""
    while True:
        raw_value = prompt_text(prompt, default, prefill_default=True)
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a number.")


def prompt_neuron_params(model_name: str, cli_assignments: list[str]) -> dict:
    """Prompt for only the constructor parameters of the selected model."""
    defaults = get_default_neuron_params(model_name)
    try:
        defaults.update(parse_neuron_param_assignments(cli_assignments, model_name))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\n{model_name} parameters")
    print("Press Enter to keep the shown default, or edit the value before pressing Enter.")

    params = {}
    for name, default in defaults.items():
        while True:
            raw_value = prompt_text(name, default, prefill_default=True)
            try:
                params[name] = parse_typed_value(raw_value, default)
                break
            except ValueError as exc:
                print(exc)
    return params


def prompt_common_args(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for file/output settings shared by all encoding modes."""
    print("\nInput/output")
    args.data_path = prompt_text("Data path", args.data_path, prefill_default=True)
    data_files_default = " ".join(args.data_files)
    data_files_text = prompt_text(
        "Data files, separated by spaces", data_files_default, prefill_default=True)
    args.data_files = data_files_text.split()

    output_default = args.output_path or ""
    output_path = prompt_text(
        "Output path override (blank uses default naming)", output_default)
    args.output_path = output_path or None

    letters_default = " ".join(args.letters) if args.letters else ""
    letters_text = prompt_text(
        "Letters subset, separated by spaces (blank uses all)",
        letters_default,
    )
    args.letters = normalize_letters(letters_text.split()) if letters_text else None

    nb_trials_default = "" if args.nb_trials is None else args.nb_trials
    nb_trials_text = prompt_text(
        "Trials per letter to encode (blank uses all)",
        nb_trials_default,
    )
    args.nb_trials = int(nb_trials_text) if str(nb_trials_text).strip() else None
    if args.nb_trials is not None and args.nb_trials <= 0:
        raise SystemExit("--nb-trials must be a positive integer.")
    return args


def prompt_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    """Collect branch-specific event transform options interactively."""
    print("Interactive event transform")
    print("Use Ctrl+C to cancel.\n")

    args.encoding_type = prompt_choice(
        "Encoding type", ENCODING_CHOICES, args.encoding_type)

    if args.encoding_type == "mechanoreceptor":
        print("\nMechanoreceptor parameters")
        args.mechanoreceptor_threshold = prompt_float(
            "Threshold", args.mechanoreceptor_threshold)
        args.mechanoreceptor_max_frequency = prompt_float(
            "Max frequency", args.mechanoreceptor_max_frequency)
        args.mechanoreceptor_refractory_period = prompt_float(
            "Refractory period (seconds)", args.mechanoreceptor_refractory_period)
        args.neuron_params = {}
    elif args.encoding_type == "sigma_delta":
        print("\nSigma-delta parameters")
        args.sigma_delta_threshold = prompt_float(
            "Threshold", args.sigma_delta_threshold)
        args.sigma_delta_refractory_period = prompt_float(
            "Refractory period (seconds)", args.sigma_delta_refractory_period)
        args.neuron_params = {}
    elif args.encoding_type == "neuron_model":
        args.neuron_model = prompt_choice(
            "Neuron model", NEURON_MODEL_CHOICES, args.neuron_model)
        print("\nUpsampling")
        args.upsample_strategy = prompt_choice(
            "Upsample strategy", ["linear", "hold"], args.upsample_strategy)
        args.upsample_dt_s = prompt_float(
            "Upsample dt (seconds)", args.upsample_dt_s)
        args.neuron_params = prompt_neuron_params(
            args.neuron_model, args.neuron_param)

    return prompt_common_args(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode tactile data with mechanoreceptor, sigma-delta, or neuron-model pipelines."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for encoding options in a guided terminal wizard.",
    )
    parser.add_argument(
        "--encoding-type",
        choices=ENCODING_CHOICES,
        default=ENCODING_TYPE,
        help="Encoding pipeline to run.",
    )
    parser.add_argument(
        "--neuron-model",
        choices=NEURON_MODEL_CHOICES,
        default=NEURON_MODEL,
        help="Neuron model used when --encoding-type neuron_model.",
    )
    parser.add_argument(
        "--neuron-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a selected neuron-model constructor parameter. Can be repeated.",
    )
    parser.add_argument(
        "--upsample-strategy",
        choices=["linear", "hold"],
        default=UPSAMPLE_STRATEGY,
        help="Upsampling strategy for neuron-model encoding.",
    )
    parser.add_argument(
        "--upsample-dt-s",
        type=float,
        default=UPSAMPLE_DT_S,
        help="Target timestep in seconds for upsampling (e.g., 0.001 for 1 ms).",
    )
    parser.add_argument(
        "--mechanoreceptor-threshold",
        type=float,
        default=MECHANORECEPTOR_THRESHOLD,
        help="FA-I threshold used by mechanoreceptor encoding.",
    )
    parser.add_argument(
        "--mechanoreceptor-max-frequency",
        type=float,
        default=MECHANORECEPTOR_MAX_FREQUENCY,
        help="SA-II maximum firing frequency in Hz used by mechanoreceptor encoding.",
    )
    parser.add_argument(
        "--mechanoreceptor-refractory-period",
        type=float,
        default=MECHANORECEPTOR_REFRACTORY_PERIOD,
        help="Refractory period in seconds used by mechanoreceptor encoding.",
    )
    parser.add_argument(
        "--sigma-delta-threshold",
        type=float,
        default=SIGMA_DELTA_THRESHOLD,
        help="Threshold used by sigma-delta encoding.",
    )
    parser.add_argument(
        "--sigma-delta-refractory-period",
        type=float,
        default=SIGMA_DELTA_REFRACTORY_PERIOD,
        help="Refractory period in seconds used by sigma-delta encoding.",
    )
    parser.add_argument(
        "--data-path",
        default=data_path,
        help="Input directory containing tactile data files.",
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=data_files,
        help="One or more input pickle filenames relative to --data-path.",
    )
    parser.add_argument(
        "--output-path",
        default=OUTPUT_PATH_OVERRIDE,
        help="Optional explicit output file path. If unset, default naming is used.",
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        default=SELECTED_LETTERS,
        help="Optional subset of letters to encode (e.g., --letters A B).",
    )
    parser.add_argument(
        "--nb-trials",
        type=int,
        default=NB_TRIALS,
        help="Optional number of trials to encode per selected letter, after sorting by letter.",
    )
    args = parser.parse_args()
    if args.interactive:
        return prompt_interactive_args(args)

    args.letters = normalize_letters(args.letters)
    if args.nb_trials is not None and args.nb_trials <= 0:
        parser.error("--nb-trials must be a positive integer.")

    try:
        args.neuron_params = parse_neuron_param_assignments(
            args.neuron_param, args.neuron_model)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def apply_args(args: argparse.Namespace) -> None:
    global ENCODING_TYPE, NEURON_MODEL, UPSAMPLE_STRATEGY, UPSAMPLE_DT_S
    global MECHANORECEPTOR_THRESHOLD, MECHANORECEPTOR_MAX_FREQUENCY, MECHANORECEPTOR_REFRACTORY_PERIOD
    global SIGMA_DELTA_THRESHOLD, SIGMA_DELTA_REFRACTORY_PERIOD, NEURON_MODEL_PARAMS
    global data_path, data_files, OUTPUT_PATH_OVERRIDE, SELECTED_LETTERS, NB_TRIALS

    ENCODING_TYPE = args.encoding_type
    NEURON_MODEL = args.neuron_model
    UPSAMPLE_STRATEGY = args.upsample_strategy
    UPSAMPLE_DT_S = args.upsample_dt_s
    MECHANORECEPTOR_THRESHOLD = args.mechanoreceptor_threshold
    MECHANORECEPTOR_MAX_FREQUENCY = args.mechanoreceptor_max_frequency
    MECHANORECEPTOR_REFRACTORY_PERIOD = args.mechanoreceptor_refractory_period
    SIGMA_DELTA_THRESHOLD = args.sigma_delta_threshold
    SIGMA_DELTA_REFRACTORY_PERIOD = args.sigma_delta_refractory_period
    NEURON_MODEL_PARAMS = args.neuron_params
    data_path = args.data_path
    data_files = args.data_files
    OUTPUT_PATH_OVERRIDE = args.output_path
    SELECTED_LETTERS = args.letters
    NB_TRIALS = args.nb_trials


def normalize_letters(letters: list[str] | None) -> list[str] | None:
    """Normalize optional letter args and accept comma-separated entries."""
    if not letters:
        return None

    normalized = []
    for letter in letters:
        for token in str(letter).split(","):
            token = token.strip()
            if token:
                normalized.append(token.upper())
    return normalized or None


def sort_output_dict_by_letter(out_dict: dict, letter_key: str = "letter") -> dict:
    """Sort all per-trial list entries in out_dict using the order of `letter_key`."""
    if letter_key not in out_dict or len(out_dict[letter_key]) == 0:
        return out_dict

    sorted_indices = np.argsort(out_dict[letter_key])
    expected_len = len(out_dict[letter_key])

    for key, value in out_dict.items():
        if isinstance(value, list) and len(value) == expected_len:
            out_dict[key] = [value[i] for i in sorted_indices]

    return out_dict


def save_encoded_output(out_dict: dict, output_path: str) -> None:
    """Save encoded output dictionary to pickle with a consistent log message."""
    print("Saving encoded data...")
    with open(output_path, "wb") as f:
        pkl.dump(out_dict, f)
    print("Done.")


def load_trials(input_data_path: str, input_files: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load all trials across files and return them as a flat list."""
    trials = []
    for file in tqdm(input_files, desc="Loading files"):
        with open(f"{input_data_path}/{file}", "rb") as f:
            data = pkl.load(f)

        letter_list = data["letter"].values
        taxels_list = data["taxel_data"].values
        timestamps_list = data["timestamp"].values

        for letter, taxels, timestamps in zip(letter_list, taxels_list, timestamps_list):
            trials.append((str(letter), taxels, timestamps))
    return trials


def filter_sorted_trials(
    trials: list[tuple[str, np.ndarray, np.ndarray]],
    letters: list[str] | None = None,
    nb_trials: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Filter sorted trials by letter subset and optional trials-per-letter limit."""
    if not trials:
        return []

    selected_letters = set(normalize_letters(letters) or [])
    filtered = []
    per_letter_counts = {}

    for letter, taxels, timestamps in trials:
        normalized_trial_letter = str(letter).upper()
        if selected_letters and normalized_trial_letter not in selected_letters:
            continue

        if nb_trials is not None:
            count = per_letter_counts.get(normalized_trial_letter, 0)
            if count >= nb_trials:
                continue
            per_letter_counts[normalized_trial_letter] = count + 1

        filtered.append((letter, taxels, timestamps))

    return filtered


def iterate_trials(
    input_data_path: str,
    input_files: list[str],
    letters: list[str] | None = None,
    nb_trials: int | None = None,
):
    """Yield sorted and optionally filtered (letter, taxels, timestamps) trials."""
    trials = load_trials(input_data_path, input_files)
    trials.sort(key=lambda x: x[0])

    filtered_trials = filter_sorted_trials(
        trials,
        letters=letters,
        nb_trials=nb_trials,
    )

    if letters or nb_trials is not None:
        print(
            "Selected "
            f"{len(filtered_trials)}/{len(trials)} trials "
            f"(letters={letters or 'all'}, nb_trials={nb_trials if nb_trials is not None else 'all'})."
        )

    for letter, taxels, timestamps in tqdm(filtered_trials, desc="Encoding letters"):
        yield letter, taxels, timestamps


def add_subset_suffix(output_path: str, letters: list[str] | None, nb_trials: int | None) -> str:
    """Append subset metadata to the default output filename."""
    if not letters and nb_trials is None:
        return output_path

    path = Path(output_path)
    suffix_parts = []
    normalized_letters = normalize_letters(letters)
    if normalized_letters:
        suffix_parts.append("letters-" + "-".join(normalized_letters))
    if nb_trials is not None:
        suffix_parts.append(f"trials-{nb_trials}")

    return str(path.with_name(f"{path.stem}_{'_'.join(suffix_parts)}{path.suffix}"))


def update_trial_duration_bounds(
    timestamps: np.ndarray,
    longest_trial: float,
    shortest_trial: float,
) -> tuple[float, float]:
    """Update min/max trial duration bounds from one timestamp array."""
    trial_end = timestamps[-1]
    longest_trial = max(longest_trial, trial_end)
    shortest_trial = min(shortest_trial, trial_end)
    return longest_trial, shortest_trial


def build_neuron_model(
    model_name: str,
    nb_inputs: int,
    neuron_params: dict | None = None,
    dt_s: float | None = None,
):
    """Create a neuron model instance for the configured model name."""
    if model_name not in NEURON_MODEL_CLASSES:
        raise ValueError(f"Unknown neuron model: {model_name}")
    model_class = NEURON_MODEL_CLASSES[model_name]
    params = dict(neuron_params or {})
    if dt_s is not None and "dt" in inspect.signature(model_class.__init__).parameters:
        params["dt"] = dt_s
    return model_class(
        nb_inputs=nb_inputs,
        **params,
    )


def upsample_taxel_trial(
    timestamps: np.ndarray,
    taxels: np.ndarray,
    dt_s: float = UPSAMPLE_DT_S,
    strategy: str = UPSAMPLE_STRATEGY,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one trial to a fixed timestep using linear or zero-order hold."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    taxels = np.asarray(taxels, dtype=float)

    if taxels.ndim != 2:
        raise ValueError("taxels must be shaped (time, channels)")
    if timestamps.shape[0] != taxels.shape[0]:
        raise ValueError(
            "timestamps and taxels must have matching time length")
    if timestamps.shape[0] < 2:
        return timestamps, taxels
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    # Ensure monotonic timestamps and collapse duplicates to the most recent sample.
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    taxels = taxels[order]
    keep_last_duplicate = np.concatenate(
        (timestamps[1:] != timestamps[:-1], [True]))
    timestamps = timestamps[keep_last_duplicate]
    taxels = taxels[keep_last_duplicate]

    if timestamps.shape[0] < 2:
        return timestamps, taxels

    t_start = timestamps[0]
    t_end = timestamps[-1]
    n_steps = int(np.floor((t_end - t_start) / dt_s)) + 1
    upsampled_timestamps = t_start + np.arange(n_steps, dtype=float) * dt_s

    if strategy == "linear":
        upsampled_taxels = np.empty(
            (upsampled_timestamps.shape[0], taxels.shape[1]), dtype=float)
        for i in range(taxels.shape[1]):
            upsampled_taxels[:, i] = np.interp(
                upsampled_timestamps, timestamps, taxels[:, i])
    elif strategy == "hold":
        indices = np.searchsorted(
            timestamps, upsampled_timestamps, side="right") - 1
        indices = np.clip(indices, 0, timestamps.shape[0] - 1)
        upsampled_taxels = taxels[indices]
    else:
        raise ValueError(
            f"Unsupported UPSAMPLE_STRATEGY '{strategy}'. Use 'linear' or 'hold'.")

    # cleaning up floating point precision issues
    upsampled_timestamps = np.round(upsampled_timestamps, 8)
    return upsampled_timestamps, upsampled_taxels


def main() -> None:
    longest_trial = -np.inf
    shortest_trial = np.inf

    if ENCODING_TYPE == "mechanoreceptor":
        print("Encoding tactile data using mechanoreceptor models...")
        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "fa_spikes": [],
                    "sa_spikes": []}

        for letter, taxels, timestamps in iterate_trials(
            data_path,
            data_files,
            letters=SELECTED_LETTERS,
            nb_trials=NB_TRIALS,
        ):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            fa_spikes = []
            sa_spikes = []
            fa_encoding = RA_I_mechanoreceptor(
                taxel_values=taxels[0],
                fa_threshold=MECHANORECEPTOR_THRESHOLD,
                ref_period=MECHANORECEPTOR_REFRACTORY_PERIOD,
            )
            sa_encoding = SA_II_mechanoreceptor(
                channels=len(taxels[0]),
                max_frequ=MECHANORECEPTOR_MAX_FREQUENCY,
                ref_period=MECHANORECEPTOR_REFRACTORY_PERIOD,
            )
            last_time = timestamps[0]
            for t_idx in range(1, taxels.shape[0]):
                current_time = timestamps[t_idx]
                current_taxels = taxels[t_idx]
                fa_result = fa_encoding.step(
                    taxel_values=current_taxels, current_time=current_time, last_time=last_time)
                sa_result = sa_encoding.step(
                    taxel_values=current_taxels, current_time=current_time, last_time=last_time)

                # Only append if there are events (not empty)
                if fa_result.shape[0] > 0:
                    fa_spikes.extend(fa_result)
                if sa_result.shape[0] > 0:
                    sa_spikes.extend(sa_result)

                last_time = current_time
            # print("Encoding complete. Adding spikes to data...")
            out_dict["letter"].append(letter)
            out_dict["taxel_data"].append(taxels)
            out_dict["timestamps"].append(timestamps)
            out_dict["fa_spikes"].append(np.array(fa_spikes, dtype=float))
            out_dict["sa_spikes"].append(np.array(sa_spikes, dtype=float))

        out_path = f"{data_path}/mechanoreceptor_encoded.pkl"

    elif ENCODING_TYPE == "sigma_delta":
        print("Sigma-delta encoding is used.")

        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "ON_spikes": [],
                    "OFF_spikes": []}

        for letter, taxels, timestamps in iterate_trials(
            data_path,
            data_files,
            letters=SELECTED_LETTERS,
            nb_trials=NB_TRIALS,
        ):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            ON_spikes = []
            OFF_spikes = []
            sigma_delta_encoding = RA_I_mechanoreceptor(
                taxel_values=taxels[0],
                fa_threshold=SIGMA_DELTA_THRESHOLD,
                ref_period=SIGMA_DELTA_REFRACTORY_PERIOD,
            )
            last_time = timestamps[0]
            diff_taxels = np.diff(taxels, axis=0)
            for t_idx in range(1, taxels.shape[0]):
                current_time = timestamps[t_idx]
                current_taxels = taxels[t_idx]
                sigma_delta_results = sigma_delta_encoding.step(
                    taxel_values=current_taxels, current_time=current_time, last_time=last_time)

                # Only append if there are events (not empty)
                if sigma_delta_results.shape[0] > 0:
                    ON_mask = np.where(diff_taxels[t_idx - 1] > 0)[0]
                    OFF_mask = np.where(diff_taxels[t_idx - 1] < 0)[0]

                    # sigma_delta_results is expected as (N, 2): [time, taxel_index]
                    event_rows = np.asarray(sigma_delta_results)
                    if event_rows.ndim == 2 and event_rows.shape[1] >= 2:
                        event_taxels = event_rows[:, 1].astype(int)

                        if ON_mask.size > 0:
                            on_events = event_rows[np.isin(
                                event_taxels, ON_mask)]
                            if on_events.shape[0] > 0:
                                ON_spikes.extend(on_events)

                        if OFF_mask.size > 0:
                            off_events = event_rows[np.isin(
                                event_taxels, OFF_mask)]
                            if off_events.shape[0] > 0:
                                OFF_spikes.extend(off_events)

                last_time = current_time
            # print("Encoding complete. Adding spikes to data...")
            out_dict["letter"].append(letter)
            out_dict["taxel_data"].append(taxels)
            out_dict["timestamps"].append(timestamps)
            out_dict["ON_spikes"].append(np.array(ON_spikes, dtype=float))
            out_dict["OFF_spikes"].append(np.array(OFF_spikes, dtype=float))

        out_path = f"{data_path}/sigma_delta_encoded.pkl"

    elif ENCODING_TYPE == "neuron_model":

        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "spikes": []}

        for letter, taxels, timestamps in iterate_trials(
            data_path,
            data_files,
            letters=SELECTED_LETTERS,
            nb_trials=NB_TRIALS,
        ):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            timestamps, taxels = upsample_taxel_trial(
                timestamps=timestamps,
                taxels=taxels,
                dt_s=UPSAMPLE_DT_S,
                strategy=UPSAMPLE_STRATEGY,
            )
            spikes = []
            neuron_model = build_neuron_model(
                NEURON_MODEL,
                nb_inputs=len(taxels[0]),
                neuron_params=NEURON_MODEL_PARAMS,
                dt_s=UPSAMPLE_DT_S,
            )

            # let us now normalize the taxel values to be in the range [0, 1] for the neuron model
            taxels = taxels/255.0

            last_time = timestamps[0]
            for t_idx in range(1, taxels.shape[0]):
                current_taxels = torch.as_tensor(taxels[t_idx])
                neuron_response = neuron_model.forward(input=current_taxels)
                if neuron_response is None:
                    continue
                neuron_response = neuron_response.detach().cpu().numpy()
                neuron_response = neuron_response[0, :]
                if np.sum(neuron_response) > 0:
                    aer_spikes = np.array([[timestamps[t_idx], i] for i in range(
                        len(neuron_response)) if neuron_response[i] > 0])
                    spikes.extend(aer_spikes)

            # print("Encoding complete. Adding spikes to data...")
            out_dict["letter"].append(letter)
            out_dict["taxel_data"].append(taxels)
            out_dict["timestamps"].append(timestamps)
            out_dict["spikes"].append(np.array(spikes, dtype=float))
        out_path = f"{data_path}/{NEURON_MODEL}_encoded.pkl"

    else:
        raise ValueError(f"Unsupported ENCODING_TYPE: {ENCODING_TYPE}")

    if not out_dict["letter"]:
        print("No trials were processed. Nothing to save.")
        raise SystemExit(0)

    print(f"Longest trial duration: {longest_trial} seconds")
    print(f"Shortest trial duration: {shortest_trial} seconds")

    out_dict = sort_output_dict_by_letter(out_dict)
    output_path = OUTPUT_PATH_OVERRIDE or add_subset_suffix(
        out_path,
        letters=SELECTED_LETTERS,
        nb_trials=NB_TRIALS,
    )
    save_encoded_output(out_dict, output_path)


if __name__ == "__main__":
    apply_args(parse_args())
    main()
