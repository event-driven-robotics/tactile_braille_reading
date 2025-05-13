'''
Copyright (C) 2021
Authors: Alejandro Pequeno-Zurro

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
'''
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from tqdm import tqdm

DEBUG = False
DirOut = "./figures/event_encoding"
save_fig = False
EXPERIMENT_TYPE = "100Hz"  # 40Hz, 100Hz

def sample_to_changes(sample: list, f: float, threshold: int, save: str = False, DEBUG: str = False) -> list:
    ''' 
    Convert one sample time-based to event-based
    sample: time-based sample
    f: frequency of the time-based sequence
    threshold: create an event at certain threshold value
    Find the local max and min values of the sequence and applied interpolation in time based on threshold to find
    the correspondent event time.
    '''
    Precision = 8   # Fix numerical errors due to float values in arange method 29.000000000000014
    n = sample.shape[0]
    dt = 1/f
    taxel_samples = np.transpose(sample, (1, 0)).tolist()
    sample_list = list()
    for nt, taxel in enumerate(taxel_samples):
        # Find indexes in the sequence with local maximum and minimum to apply interpolation.
        txl = np.array(taxel, dtype=int)
        #   max
        ind_max = np.squeeze(np.array(argrelextrema(txl, np.greater_equal)))
        # Match dimensions of the index
        d_ixtr = np.insert(np.diff(ind_max), 0, -1)
        max_p = ind_max[d_ixtr != 1]
        #   min
        ind_min = np.squeeze(np.array(argrelextrema(txl, np.less_equal)))
        # Match dimensions of the index
        d_ixtr = np.insert(np.diff(ind_min), 0, -1)
        min_p = ind_min[d_ixtr != 1]
        #   add index with same values
        all_indx = np.append(max_p, min_p)
        i = 0
        while i < len(all_indx):
            try:
                ival = all_indx[i]
                if txl[ival + 1] - txl[ival] == 0:
                    all_indx = np.append(all_indx, np.array(ival + 1))
            except IndexError:
                None
            i += 1
        # Corresponding values in the sequence
        all_t = np.unique(np.sort(all_indx))
        all_values = txl[all_t]
        # Find the events [ON, OFF]
        taxel_list = list()
        on_events = np.array([])
        off_events = np.array([])
        # Compare each pair of points and generate event times based on threshold
        # Last value storage controls when threshold is not reached
        last_value = all_values[0]
        for i in range(len(all_values) - 1):
            d_pair = all_values[i+1] - last_value
            if d_pair > 0:
                start = last_value + threshold
                stop = all_values[i+1] + 0.0001
                spk_values = np.round(
                    np.arange(start, stop, threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp.astype(float), 'linear')
                on_events = np.append(
                    on_events, np.apply_along_axis(f, 0, spk_values))
                # Change value of sensor when spike
                last_value = spk_values[-1] if spk_values.size > 0 else last_value
            elif d_pair < 0:
                start = last_value - threshold
                stop = all_values[i+1] - 0.0001  # No Threshold
                spk_values = np.round(
                    np.arange(start, stop, -1*threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp, 'linear')
                off_events = np.append(
                    off_events, np.apply_along_axis(f, 0, spk_values))
                # Change value of sensor when spike
                last_value = spk_values[-1] if spk_values.size > 0 else last_value
        # Assign events
        taxel_list.append((on_events * dt).tolist())
        taxel_list.append((off_events * dt).tolist())
        sample_list.append(taxel_list)
        # Plot conversions. Run in debug mode
        if DEBUG:
            n = len(txl)
            scale = 1#/5

            # SAMPLE PLOT
            fig = plt.figure(figsize=(14, 8)) 
            ax = fig.add_subplot(2, 1, 1)
            # ax.set_xlim([0, ((scale * n) - 0.5) * dt])
            ax.set_xlim([1.8, 2.2])
            ax.set_ylim([-66, -24])
            ax.plot(np.arange(start=0, stop=(n - 0.5)
                     * dt, step=dt), txl - txl[0], '-*')
            ax.set_ylabel("Sensor value", fontdict={'size': 15})
            ax.set_xticklabels([])            
            ax.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
            # Enable and style the minor grid
            ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
            ax.minorticks_on()  # Turn on minor ticks

            # EVENT PLOT
            ax = fig.add_subplot(2, 1, 2)
            # ax.set_xlim([0, ((scale * n) - 0.5) * dt])
            ax.set_yticks([-0.15, 0.15])
            ax.set_yticklabels(['OFF', 'ON'], fontdict={'size': 12})
            ax.set_xlim([1.8, 2.2])
            ax.set_ylim([-0.4, 0.4])
            if taxel_list[0]:
                ax.eventplot(taxel_list[0], lineoffsets=0.15,
                              colors='green', linelength=0.25)
            if taxel_list[1]:
                ax.eventplot(taxel_list[1], lineoffsets=-0.15,
                              colors='red', linelength=0.25)

            # ax.set_ylabel(r'$\vartheta = ${}'.format(str(threshold)), fontdict={'size': 15})
            ax.set_ylabel("Events", fontdict={'size': 15})
            ax.set_xlabel('t(s)', fontdict={'size': 12})
            ax.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
            # Enable and style the minor grid
            ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
            ax.minorticks_on()  # Turn on minor ticks
            fig.align_ylabels()

            if save:
                plt.savefig(f'{DirOut}/encoding_TH{threshold}_taxel_{nt}.pdf', bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

    return sample_list


def extract_data_icub_raw_integers_40Hz(file_name: str) -> tuple:
    ''' Read the files and convert taxel data and labels
        file_name: filename of the dataset in format dict{'taxel_data':, 'letter':}
    '''
    data = []
    labels = []
    print("file name {}".format(file_name))
    with open(file_name, 'rb') as infile:
        data_dict = pickle.load(infile)
    for item in data_dict:
        dat = np.abs(255 - item['taxel_data'][:])
        data.append(dat)
        labels.append(item['letter'])
    return data, labels


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


def main():
    ''' Convert time-based data into event-based data '''
    Spk_thresholds = [1, 2, 5, 10]  # (2 default)
    tqdm_threshold = tqdm(Spk_thresholds, desc="Thresholds", position=0, leave=False, total=len(Spk_thresholds))
    for Spk_threshold in tqdm_threshold:
        tqdm_threshold.set_description(f"Threshold {Spk_threshold}")
        samples = list()
        if EXPERIMENT_TYPE == '40Hz':
            f = 40  # Hz
            data_raw, labels_raw = extract_data_icub_raw_integers_40Hz(
                './data/40Hz/data_braille_letters_digits.pkl')
            Events_filename_out = f'./data/40Hz/data_braille_letters_40Hz_th{Spk_threshold}.pkl'

            # Each sequence sample is parsed to events
            pbar_samples = tqdm(zip(data_raw, labels_raw), total=len(data_raw), desc="Encoding samples", position=1, leave=False)
            for sample_raw, label in pbar_samples:
                pbar_samples.set_description(f"Encoding {label}")
                data_dict_events = {}
                events_per_samples = sample_to_changes(
                    sample_raw, f, Spk_threshold, save=save_fig)
                # Dict of the sample
                data_dict_events['letter'] = label
                data_dict_events['events'] = events_per_samples
                samples.append(data_dict_events)


        elif EXPERIMENT_TYPE == '100Hz':
            # 300 trials per letter, 100 per recording, 3 recordings, 8100 trials in total
            f = 100  # Hz
            data_raw = []
            labels_raw = []
            displacement = ["0.0", "0.000125"]
            for dis in displacement:
                data_tmp, labels_tmp, _ = extract_data_icub_raw_integers_100Hz(
                    f'./data/100Hz/data_braille_letters_{dis}.pkl')
                data_raw.extend(data_tmp)
                labels_raw.extend(labels_tmp)
            order = np.argsort(labels_raw)
            data_raw = np.array(data_raw)[order]
            labels_raw = np.array(labels_raw)[order]
            Events_filename_out = f'./data/100Hz/data_braille_letters_100Hz_th{Spk_threshold}.pkl'

            # Each sequence sample is parsed to events
            pbar_samples = tqdm(zip(data_raw, labels_raw), total=len(data_raw), desc="Encoding samples", position=1, leave=False)
            for sample_raw, label in pbar_samples:
                pbar_samples.set_description(f"Encoding {label}")
                data_dict_events = {}
                events_per_samples = sample_to_changes(
                    sample_raw, f, Spk_threshold, save=save_fig, DEBUG=DEBUG)
                # Dict of the sample
                data_dict_events['letter'] = label
                data_dict_events['events'] = events_per_samples
                data_dict_events['samples'] = sample_raw
                samples.append(data_dict_events)
                
        else:
            raise ValueError('Experiment type not supported')

        if save_fig:
            isExist = os.path.exists(DirOut)
            if not isExist:
                os.makedirs(DirOut)


        with open(Events_filename_out, 'wb') as outf:
            pickle.dump(samples, outf)
    
    print('Finished conversion')


if __name__ == "__main__":
    main()
