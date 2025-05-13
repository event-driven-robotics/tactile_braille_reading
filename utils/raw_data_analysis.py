import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # Import Line2D for custom legend entries
from tqdm import tqdm


def extract_data_icub_raw_integers_100Hz(file_name: str) -> tuple:
    ''' Read the files and convert taxel data and labels
        file_name: filename of the dataset in format dict{'taxel_data':, 'letter':}
    '''
    # print("file name {}".format(file_name))
    with open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)
    data = data_dict["taxel_data"].values
    letter = data_dict["letter"].values
    timestamps = data_dict["timestamp"].values
    return data, letter, timestamps


if __name__ == "__main__":
    # 300 trials per letter, 100 per recording, 3 recordings, 8100 trials in total
    f = 100  # Hz
    data_raw = []
    labels_raw = []
    displacement = ["0.0", "0.000125"]  # erronous P in: "6.25e-05"
    tqdm_displacement = tqdm(displacement, desc="Displacement",
                             position=0, leave=False, total=len(displacement))
    for dis in tqdm_displacement:
        tqdm_displacement.set_description(f"Displacement: {dis}")
        data_tmp, labels_tmp, _ = extract_data_icub_raw_integers_100Hz(
            f'./data/100Hz/data_braille_letters_{dis}.pkl')
        data_raw.extend(data_tmp)
        labels_raw.extend(labels_tmp)
    order = np.argsort(labels_raw)
    data_raw = np.array(data_raw)[order]
    labels_raw = np.array(labels_raw)[order]

    # let's create plot for each letter
    letters = np.unique(labels_raw)
    all_idc = []
    for letter in letters:
        all_idc.append(np.where(labels_raw == letter)[0])
    all_idc = np.array(all_idc)

    max_val = np.max(data_raw)
    tqdm_plot = tqdm(zip(letters, all_idc), desc="Plotting",
                     position=0, leave=False, total=len(letters))
    for letter, idc in tqdm_plot:
        tqdm_plot.set_description(f"Plotting: {letter}")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        # let's calculate the mean and std for the trial at this idc
        mean = np.mean(data_raw[idc], axis=0)
        std = np.std(data_raw[idc], axis=0)
        # let's plot the mean and std
        time = np.arange(len(mean)) / f
        for taxel in range(mean.shape[1]):
            ax.plot(time, mean[:, taxel], label=f"Taxel {taxel + 1}")
            ax.fill_between(
                time, mean[:, taxel] - std[:, taxel], mean[:, taxel] + std[:, taxel], alpha=0.2)
        ax.set_ylim(0, max_val)
        ax.set_title(f"Letter: {letter}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Taxel Value")
        ax.legend()
        fig.savefig(f"./figures/raw_data_analysis/letter_{letter}.pdf")
        plt.close(fig)

        # let's check for outliers
        # Use a colormap for consistent colors
        colors = plt.cm.tab10(np.linspace(0, 1, mean.shape[1]))
        factor = 5
        threshold = 10
        for trial in idc:
            # I want to exclude taxel which mean value is below threshold
            total_mean = np.mean(mean, axis=0)
            idc_to_keep = np.where(total_mean > threshold)[0]
            if np.any(data_raw[trial, :, idc_to_keep] > mean[:, idc_to_keep].transpose() + factor * std[:, idc_to_keep].transpose()) or np.any(data_raw[trial, :, idc_to_keep] < mean[:, idc_to_keep].transpose() - factor * std[:, idc_to_keep].transpose()):
                # print(f"Trial {trial} is an outlier")
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                # let's plot the mean and std for this trial
                time = np.arange(len(data_raw[trial])) / f
                for taxel in idc_to_keep:
                    ax.plot(
                        time, mean[:, taxel], label=f"Mean Taxel {taxel + 1}", linestyle='--', color=colors[taxel % len(colors)])
                    ax.fill_between(time, mean[:, taxel] - factor*std[:, taxel], mean[:, taxel] +
                                    factor*std[:, taxel], alpha=0.2, color=colors[taxel % len(colors)])
                for taxel in idc_to_keep:
                    ax.plot(time, data_raw[trial, :, taxel],
                            label=f"Taxel {taxel + 1}", color=colors[taxel % len(colors)])
                # let's highlight the outlier using a scatter plot with red dots
                outlier_indices = np.where(
                    (data_raw[trial] > mean + factor * std) | (data_raw[trial] < mean - factor * std))
                ax.scatter(time[outlier_indices[0]], data_raw[trial]
                           [outlier_indices], color='red', label='Outliers', s=10)
                ax.set_ylim(0, max_val)
                ax.set_title(f"Letter: {letter}, Trial: {trial}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Taxel Value")
                ax.legend()
                fig.savefig(
                    f"./figures/raw_data_analysis//outlier_analysis/outlier_letter_{letter}_trial_{trial}.pdf")
                plt.close(fig)
        pass
