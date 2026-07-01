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

# Options: "mechanoreceptor", "sigma_delta", "neuron_model"
ENCODING_TYPE = "neuron_model"
NEURON_MODEL = "MN_neuron"  # Options: "AdExLIF_neuron", "CuBaLIF_neuron", "IZ_neuron", "LIF_neuron", "MN_neuron"
UPSAMPLE_STRATEGY = "linear"  # Options: "linear", "hold"
UPSAMPLE_DT_S = 0.001  # Fixed target delta t in seconds (1 ms)
OUTPUT_PATH_OVERRIDE = None

data_path = "data/100Hz"
data_files = ["data_braille_letters_0.0.pkl",
              "data_braille_letters_0.000125.pkl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode tactile data with mechanoreceptor, sigma-delta, or neuron-model pipelines."
    )
    parser.add_argument(
        "--encoding-type",
        choices=["mechanoreceptor", "sigma_delta", "neuron_model"],
        default=ENCODING_TYPE,
        help="Encoding pipeline to run.",
    )
    parser.add_argument(
        "--neuron-model",
        choices=["AdExLIF_neuron", "CuBaLIF_neuron", "IZ_neuron", "LIF_neuron", "MN_neuron"],
        default=NEURON_MODEL,
        help="Neuron model used when --encoding-type neuron_model.",
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
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global ENCODING_TYPE, NEURON_MODEL, UPSAMPLE_STRATEGY, UPSAMPLE_DT_S
    global data_path, data_files, OUTPUT_PATH_OVERRIDE

    ENCODING_TYPE = args.encoding_type
    NEURON_MODEL = args.neuron_model
    UPSAMPLE_STRATEGY = args.upsample_strategy
    UPSAMPLE_DT_S = args.upsample_dt_s
    data_path = args.data_path
    data_files = args.data_files
    OUTPUT_PATH_OVERRIDE = args.output_path


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


def iterate_trials(input_data_path: str, input_files: list[str]):
    """Yield (letter, taxels, timestamps) for each trial across all input files."""
    for file in tqdm(input_files, desc="Processing files"):
        with open(f"{input_data_path}/{file}", "rb") as f:
            data = pkl.load(f)

        letter_list = data["letter"].values
        taxels_list = data["taxel_data"].values
        timestamps_list = data["timestamp"].values

        for letter, taxels, timestamps in tqdm(
            zip(letter_list, taxels_list, timestamps_list),
            total=len(letter_list),
            desc="Encoding letters",
            leave=False,
        ):
            yield letter, taxels, timestamps


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


def build_neuron_model(model_name: str, nb_inputs: int):
    """Create a neuron model instance for the configured model name."""
    if model_name == "AdExLIF_neuron":
        return AdExLIF_neuron(nb_inputs=nb_inputs)
    if model_name == "CuBaLIF_neuron":
        return CuBaLIF_neuron(nb_inputs=nb_inputs)
    if model_name == "IZ_neuron":
        return IZ_neuron(nb_inputs=nb_inputs)
    if model_name == "LIF_neuron":
        return LIF_neuron(nb_inputs=nb_inputs)
    if model_name == "MN_neuron":
        return MN_neuron(nb_inputs=nb_inputs)
    raise ValueError(f"Unknown neuron model: {model_name}")


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

        for letter, taxels, timestamps in iterate_trials(data_path, data_files):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            fa_spikes = []
            sa_spikes = []
            fa_encoding = RA_I_mechanoreceptor(
                taxel_values=taxels[0], fa_threshold=2, ref_period=0.003)
            sa_encoding = SA_II_mechanoreceptor(
                channels=len(taxels[0]), max_frequ=150, ref_period=0.003)
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

        for letter, taxels, timestamps in iterate_trials(data_path, data_files):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            ON_spikes = []
            OFF_spikes = []
            sigma_delta_encoding = RA_I_mechanoreceptor(
                taxel_values=taxels[0], fa_threshold=2, ref_period=0.003)
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

        for letter, taxels, timestamps in iterate_trials(data_path, data_files):
            longest_trial, shortest_trial = update_trial_duration_bounds(
                timestamps, longest_trial, shortest_trial
            )
            spikes = []
            neuron_model = build_neuron_model(
                NEURON_MODEL, nb_inputs=len(taxels[0]))

            timestamps, taxels = upsample_taxel_trial(
                timestamps=timestamps,
                taxels=taxels,
                dt_s=UPSAMPLE_DT_S,
                strategy=UPSAMPLE_STRATEGY,
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
    save_encoded_output(out_dict, OUTPUT_PATH_OVERRIDE or out_path)


if __name__ == "__main__":
    apply_args(parse_args())
    main()
