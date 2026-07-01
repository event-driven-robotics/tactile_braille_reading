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
from pathlib import Path

import numpy as np
from tqdm import tqdm
import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

# Ensure local package imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.neuron_models import RA_I_mechanoreceptor, SA_II_mechanoreceptor, AdExLIF_neuron, CuBaLIF_neuron, IZ_neuron, LIF_neuron, MN_neuron

ENCODING_TYPE = "neuron_model"  # Options: "mechanoreceptor", "sigma_delta", "neuron_model"
NEURON_MODEL = "MN_neuron"
UPSAMPLE_STRATEGY = "linear"  # Options: "linear", "hold"
UPSAMPLE_DT_S = 0.001  # Fixed target delta t in seconds (1 ms)

data_path = "data/100Hz"
data_files = ["data_braille_letters_0.0.pkl",
              "data_braille_letters_0.000125.pkl"]


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
        raise ValueError("timestamps and taxels must have matching time length")
    if timestamps.shape[0] < 2:
        return timestamps, taxels
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    # Ensure monotonic timestamps and collapse duplicates to the most recent sample.
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    taxels = taxels[order]
    keep_last_duplicate = np.concatenate((timestamps[1:] != timestamps[:-1], [True]))
    timestamps = timestamps[keep_last_duplicate]
    taxels = taxels[keep_last_duplicate]

    if timestamps.shape[0] < 2:
        return timestamps, taxels

    t_start = timestamps[0]
    t_end = timestamps[-1]
    n_steps = int(np.floor((t_end - t_start) / dt_s)) + 1
    upsampled_timestamps = t_start + np.arange(n_steps, dtype=float) * dt_s

    if strategy == "linear":
        upsampled_taxels = np.empty((upsampled_timestamps.shape[0], taxels.shape[1]), dtype=float)
        for i in range(taxels.shape[1]):
            upsampled_taxels[:, i] = np.interp(upsampled_timestamps, timestamps, taxels[:, i])
    elif strategy == "hold":
        indices = np.searchsorted(timestamps, upsampled_timestamps, side="right") - 1
        indices = np.clip(indices, 0, timestamps.shape[0] - 1)
        upsampled_taxels = taxels[indices]
    else:
        raise ValueError(f"Unsupported UPSAMPLE_STRATEGY '{strategy}'. Use 'linear' or 'hold'.")

    upsampled_timestamps = np.round(upsampled_timestamps, 8)  # cleaning up floating point precision issues
    return upsampled_timestamps, upsampled_taxels


def main() -> None:
    if ENCODING_TYPE == "mechanoreceptor":
        print("Encoding tactile data using mechanoreceptor models...")
        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "fa_spikes": [],
                    "sa_spikes": []}

        longest_trial = -np.inf
        shortest_trial = np.inf

        for file in tqdm(data_files, desc="Processing files"):
            with open(f"{data_path}/{file}", "rb") as f:
                data = pkl.load(f)
            letter_list = data["letter"].values
            taxels_list = data["taxel_data"].values
            timestamps_list = data["timestamp"].values

            for (letter, taxels, timestamps) in tqdm(zip(letter_list, taxels_list, timestamps_list),
                                                                    total=len(
                                                                        letter_list),
                                                                    desc=f"Encoding letters",
                                                                    leave=False):
                if timestamps[-1] > longest_trial:
                    longest_trial = timestamps[-1]            
                if timestamps[-1] < shortest_trial:
                    shortest_trial = timestamps[-1]
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
            pass

        if not out_dict["letter"]:
            print("No trials were processed. Nothing to save.")
            raise SystemExit(0)

        print(f"Longest trial duration: {longest_trial} seconds")
        print(f"Shortest trial duration: {shortest_trial} seconds")

        out_path = f"{data_path}/mechanoreceptor_encoded.pkl"
        
    elif ENCODING_TYPE == "sigma_delta":
        print("Sigma-delta encoding is used.")
        
        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "ON_spikes": [],
                    "OFF_spikes": []}

        longest_trial = -np.inf
        shortest_trial = np.inf

        for file in tqdm(data_files, desc="Processing files"):
            with open(f"{data_path}/{file}", "rb") as f:
                data = pkl.load(f)
            letter_list = data["letter"].values
            taxels_list = data["taxel_data"].values
            timestamps_list = data["timestamp"].values

            for (letter, taxels, timestamps) in tqdm(zip(letter_list, taxels_list, timestamps_list),
                                                                    total=len(
                                                                        letter_list),
                                                                    desc=f"Encoding letters",
                                                                    leave=False):
                if timestamps[-1] > longest_trial:
                    longest_trial = timestamps[-1]            
                if timestamps[-1] < shortest_trial:
                    shortest_trial = timestamps[-1]
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
                                on_events = event_rows[np.isin(event_taxels, ON_mask)]
                                if on_events.shape[0] > 0:
                                    ON_spikes.extend(on_events)

                            if OFF_mask.size > 0:
                                off_events = event_rows[np.isin(event_taxels, OFF_mask)]
                                if off_events.shape[0] > 0:
                                    OFF_spikes.extend(off_events)

                    last_time = current_time
                # print("Encoding complete. Adding spikes to data...")
                out_dict["letter"].append(letter)
                out_dict["taxel_data"].append(taxels)
                out_dict["timestamps"].append(timestamps)
                out_dict["ON_spikes"].append(np.array(ON_spikes, dtype=float))
                out_dict["OFF_spikes"].append(np.array(OFF_spikes, dtype=float))
            pass

        if not out_dict["letter"]:
            print("No trials were processed. Nothing to save.")
            raise SystemExit(0)

        print(f"Longest trial duration: {longest_trial} seconds")
        print(f"Shortest trial duration: {shortest_trial} seconds")
        out_path = f"{data_path}/sigma_delta_encoded.pkl"
            
    elif ENCODING_TYPE == "neuron_model":
       
        out_dict = {"letter": [],
                    "taxel_data": [],
                    "timestamps": [],
                    "spikes": []}

        longest_trial = -np.inf
        shortest_trial = np.inf

        for file in tqdm(data_files, desc="Processing files"):
            with open(f"{data_path}/{file}", "rb") as f:
                data = pkl.load(f)
            letter_list = data["letter"].values
            taxels_list = data["taxel_data"].values
            timestamps_list = data["timestamp"].values

            for (letter, taxels, timestamps) in tqdm(zip(letter_list, taxels_list, timestamps_list),
                                                                    total=len(
                                                                        letter_list),
                                                                    desc=f"Encoding letters",
                                                                    leave=False):
                if timestamps[-1] > longest_trial:
                    longest_trial = timestamps[-1]            
                if timestamps[-1] < shortest_trial:
                    shortest_trial = timestamps[-1]
                spikes = []
                
                if NEURON_MODEL == "AdExLIF_neuron":
                    neuron_model = AdExLIF_neuron(nb_inputs=len(taxels[0]))  # TODO here we should add (and change) the neuron model parameters to see the impact on the encoding. For now, we use the default parameters.
                elif NEURON_MODEL == "CuBaLIF_neuron":
                    neuron_model = CuBaLIF_neuron(nb_inputs=len(taxels[0]))
                elif NEURON_MODEL == "IZ_neuron":
                    neuron_model = IZ_neuron(nb_inputs=len(taxels[0]))
                elif NEURON_MODEL == "LIF_neuron":
                    neuron_model = LIF_neuron(nb_inputs=len(taxels[0]))
                elif NEURON_MODEL == "MN_neuron":
                    neuron_model = MN_neuron(nb_inputs=len(taxels[0]))
                else:
                    print("Unknown neuron model. Exiting.")
                    raise SystemExit(0)
                
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
                        aer_spikes = np.array([[timestamps[t_idx], i] for i in range(len(neuron_response)) if neuron_response[i] > 0])
                        spikes.extend(aer_spikes)

                # print("Encoding complete. Adding spikes to data...")
                out_dict["letter"].append(letter)
                out_dict["taxel_data"].append(taxels)
                out_dict["timestamps"].append(timestamps)
                out_dict["spikes"].append(np.array(spikes, dtype=float))

        if not out_dict["letter"]:
            print("No trials were processed. Nothing to save.")
            raise SystemExit(0)

        print(f"Longest trial duration: {longest_trial} seconds")
        print(f"Shortest trial duration: {shortest_trial} seconds")
        out_path = f"{data_path}/{NEURON_MODEL}_encoded.pkl"
        
    out_dict = sort_output_dict_by_letter(out_dict)
    save_encoded_output(out_dict, out_path)
             
    
if __name__ == "__main__":
    main()