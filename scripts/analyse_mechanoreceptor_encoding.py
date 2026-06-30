"""analyse_mechanoreceptor_encoding.py

Analysis script for visualizing mechanoreceptor-encoded braille letter data.

Loads pre-encoded mechanoreceptor spike data and generates visualizations for
multiple samples of each braille letter, showing FA-I and SA-II spike patterns.

Author: Simon F. Muller-Cleve
Date: January 12, 2026
"""

import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

data_path = "data/100Hz"
file_name = "mechanoreceptor_encoded.pkl"

if __name__ == "__main__":
    with open(f"{data_path}/{file_name}", "rb") as f:
        out_dict = pkl.load(f)

    letters_list = out_dict["letter"]
    taxel_data_list = out_dict["taxel_data"]
    timestamps_list = out_dict["timestamps"]
    fa_spikes_list = out_dict["fa_spikes"]
    sa_spikes_list = out_dict["sa_spikes"]

    nb_of_leters_to_visualize = 5

    # let's find all the unique letters position and select nb_of_leters_to_visualize of each to visualize
    unique_letters = sorted(set(letters_list))
    selected_indices = []
    np.random.seed(42)  # for reproducibility
    for letter in unique_letters:
        indices = [i for i, l in enumerate(letters_list) if l == letter]
        # Randomly sample nb_of_leters_to_visualize indices (or all if fewer available)
        n_samples = min(nb_of_leters_to_visualize, len(indices))
        selected = np.random.choice(indices, size=n_samples, replace=False)
        selected_indices.extend(selected)

    # now we can visualize the selected letters
    for idx in tqdm(selected_indices):
        letter = letters_list[idx]
        taxel_data = taxel_data_list[idx]
        timestamps = timestamps_list[idx]
        fa_spikes = fa_spikes_list[idx]
        sa_spikes = sa_spikes_list[idx]

        fig, (ax_sensor, ax_fa, ax_sa) = plt.subplots(
            3,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )

        ax_sensor.plot(timestamps, taxel_data)
        ax_sensor.set_title(f"Taxel Data for Letter '{letter}'")
        ax_sensor.set_xlabel("Time (s)")
        ax_sensor.set_ylabel("Taxel Value")
        ax_sensor.set_xlim(timestamps[0], timestamps[-1])

        if fa_spikes.shape[0] > 0:
            ax_fa.scatter(fa_spikes[:, 0], fa_spikes[:, 1], s=1)
        ax_fa.set_title(f"FA-I Mechanoreceptor Spikes for Letter '{letter}'")
        ax_fa.set_xlabel("Time (s)")
        ax_fa.set_ylabel("Channel")
        ax_fa.set_xlim(timestamps[0], timestamps[-1])
        ax_fa.set_ylim(-1, 13)

        if sa_spikes.shape[0] > 0:
            ax_sa.scatter(sa_spikes[:, 0], sa_spikes[:, 1], s=1, color="orange")
        ax_sa.set_title(f"SA-II Mechanoreceptor Spikes for Letter '{letter}'")
        ax_sa.set_xlabel("Time (s)")
        ax_sa.set_ylabel("Channel")
        ax_sa.set_xlim(timestamps[0], timestamps[-1])
        ax_sa.set_ylim(-1, 13)

        fig.tight_layout()
        fig.savefig(f"figures/letter_{letter}_idx_{idx}.pdf")
        plt.close(fig)
        # plt.show()
    pass
