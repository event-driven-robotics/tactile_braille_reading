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
    for idx in selected_indices:
        letter = letters_list[idx]
        taxel_data = taxel_data_list[idx]
        timestamps = timestamps_list[idx]
        fa_spikes = fa_spikes_list[idx]
        sa_spikes = sa_spikes_list[idx]

        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(timestamps, taxel_data)
        plt.title(f"Taxel Data for Letter '{letter}'")
        plt.xlabel("Time (s)")
        plt.ylabel("Taxel Value")
        plt.xlim(timestamps[0], timestamps[-1])

        plt.subplot(3, 1, 2)
        if fa_spikes.shape[0] > 0:
            plt.scatter(fa_spikes[:, 0], fa_spikes[:, 1], s=1)
        plt.title(f"FA-I Mechanoreceptor Spikes for Letter '{letter}'")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel")
        plt.xlim(timestamps[0], timestamps[-1])
        plt.ylim(-1, 13)

        plt.subplot(3, 1, 3)
        if sa_spikes.shape[0] > 0:
            plt.scatter(sa_spikes[:, 0], sa_spikes[:, 1], s=1, color='orange')
        plt.title(f"SA-II Mechanoreceptor Spikes for Letter '{letter}'")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel")
        plt.xlim(timestamps[0], timestamps[-1])
        plt.ylim(-1, 13)

        plt.tight_layout()
        plt.savefig(f"figures/letter_{letter}_idx_{idx}.pdf")
        # plt.show()
    pass