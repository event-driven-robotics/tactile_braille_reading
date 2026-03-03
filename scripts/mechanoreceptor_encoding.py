"""mechanoreceptor_encoding.py

Encoder script that processes raw tactile data through mechanoreceptor models.

Reads braille letter tactile data and generates FA-I and SA-II mechanoreceptor
spike responses using event-based neuron models. Outputs encoded spike data
for downstream neural network training.

Author: Simon F. Muller-Cleve
Date: January 12, 2026
"""

import pickle as pkl

import numpy as np
from tqdm import tqdm

from utils.neuron_models import FA_I_mechanoreceptor, SA_II_mechanoreceptor

print("Loading mechanoreceptor models...")

data_path = "data/100Hz"
data_files = ["data_braille_letters_0.0.pkl",
              "data_braille_letters_0.000125.pkl"]

if __name__ == "__main__":
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

        for trial, (letter, taxels, timestamps) in enumerate(tqdm(zip(letter_list, taxels_list, timestamps_list),
                                                                total=len(
                                                                    letter_list),
                                                                desc=f"Encoding letters",
                                                                leave=False)):
            if timestamps[-1] > longest_trial:
                longest_trial = timestamps[-1]            
            if timestamps[-1] < shortest_trial:
                shortest_trial = timestamps[-1]
            fa_spikes = []
            sa_spikes = []
            fa_encoding = FA_I_mechanoreceptor(
                taxel_values=taxels[0], fa_threshold=1, ref_period=0.003)
            sa_encoding = SA_II_mechanoreceptor(
                channels=len(taxels[0]), max_frequ=100)
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

    # let us sort the dict by letters to have a consistent order
    sorted_indices = np.argsort(out_dict["letter"])
    out_dict["letter"] = [out_dict["letter"][i] for i in sorted_indices]
    out_dict["taxel_data"] = [out_dict["taxel_data"][i] for i in sorted_indices]
    out_dict["timestamps"] = [out_dict["timestamps"][i] for i in sorted_indices]
    out_dict["fa_spikes"] = [out_dict["fa_spikes"][i] for i in sorted_indices]
    out_dict["sa_spikes"] = [out_dict["sa_spikes"][i] for i in sorted_indices]

    print("Saving encoded data...")
    with open(f"{data_path}/mechanoreceptor_encoded.pkl", "wb") as f:
        pkl.dump(out_dict, f)
    print("Done.")
