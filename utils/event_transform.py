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
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

DEBUG = False
DirOut = "../plots/"
save_fig = False

def sample_to_changes(sample, f, threshold, save):
    ''' Convert one sample time-based to event-based
        sample: time-based sample
        f: frequency of the time-based sequence
        threshold: create an event at certain threshold value
        Find the local max and min values of the sequence and applied interpolation in time based on threshold to find
        the correspondent event time.
    '''
    Precision = 4   # Fix numerical errors due to float values in arange method 29.000000000000014
    n = sample.shape[0]
    dt = 1/f
    taxel_samples = np.transpose(sample, (1, 0)).tolist()
    sample_list = list()
    for nt, taxel in enumerate(taxel_samples):
        # Find indexes in the sequence with local maximum and minimum to apply interpolation.
        txl = np.array(taxel, dtype=int)
        #   max
        ind_max = np.squeeze(np.array(argrelextrema(txl, np.greater_equal)))
        d_ixtr = np.insert(np.diff(ind_max), 0, -1)    # Match dimensions of the index
        max_p = ind_max[d_ixtr != 1]
        #   min
        ind_min = np.squeeze(np.array(argrelextrema(txl, np.less_equal)))
        d_ixtr = np.insert(np.diff(ind_min), 0, -1)  # Match dimensions of the index
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
        on_events = np.array([]); off_events = np.array([])
        # Compare each pair of points and generate event times based on threshold
        last_value = all_values[0]  # Last value storage controls when threshold is not reached
        for i in range(len(all_values) - 1):
            d_pair = all_values[i+1] - last_value
            if d_pair > 0:
                start = last_value + threshold
                stop = all_values[i+1] + 0.0001
                spk_values = np.round(np.arange(start, stop, threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp.astype(float), 'linear')
                on_events = np.append(on_events, np.apply_along_axis(f, 0, spk_values))
                last_value = spk_values[-1] if spk_values.size > 0 else last_value   # Change value of sensor when spike
            elif d_pair < 0:
                start = last_value - threshold
                stop = all_values[i+1] - 0.0001 # No Threshold
                spk_values = np.round(np.arange(start, stop, -1*threshold), Precision)
                # Interpolation with all the values of the pair
                pts = all_t[i+1] - all_t[i] + 1
                t_interp = np.linspace(all_t[i], all_t[i+1], pts, dtype=int)
                vals_interp = txl[t_interp]
                f = interp1d(vals_interp, t_interp, 'linear')
                off_events = np.append(off_events, np.apply_along_axis(f, 0, spk_values))
                last_value = spk_values[-1] if spk_values.size > 0 else last_value    # Change value of sensor when spike
        # Assign events
        taxel_list.append((on_events * dt).tolist())
        taxel_list.append((off_events * dt).tolist())
        sample_list.append(taxel_list)
        # Plot conversions. Run in debug mode
        if DEBUG:
            plt.rcParams['text.usetex'] = True
            f1 = plt.figure()
            axes = plt.axes()
            n = len(txl)
            scale = 1/5
            axes.set_xlim([0, ((scale * n) - 0.5) * dt])
            axes.set_ylim([-0.5, 0.5])
            if taxel_list[0]:
                plt.eventplot(taxel_list[0], lineoffsets=0.15,
                              colors='green', linelength=0.25)
            if  taxel_list[1]:
                plt.eventplot(taxel_list[1], lineoffsets=-0.15,
                              colors='red', linelength=0.25)

            axes.set_ylabel(r'$\vartheta = ${}'.format(str(threshold)))
            if save:
                plt.savefig('{}encoding_TH{}_taxel_{}_events.png'.format(DirOut, str(threshold), str(nt)), dpi=200)
            f2 = plt.figure()
            axes = plt.axes()
            axes.set_xlim([0, ((scale * n) - 0.5) * dt])
            plt.plot(np.arange(start=0, stop=(n - 0.5) * dt, step=dt), txl - txl[0], '-o')
            axes.set_ylabel("Sensor value")
            axes.set_xlabel('t(s)')
            if save:
                plt.savefig('{}encoding_TH{}_taxel_{}_sample.png'.format(DirOut, str(threshold), str(nt)), dpi=200)

    return sample_list

def extract_data_icub_raw_integers(file_name):
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

def main():
    ''' Convert time-based data into event-based data '''
    Spk_threshold = 1 # 1, 2, 5, 10 (default)
    Events_filename_out = '../data/data_braille_letters_th{}'.format(str(Spk_threshold))
    f = 40  # Hz
    data_raw, labels_raw = extract_data_icub_raw_integers('../data/data_braille_letters_raw')
    samples = list()
    if save_fig:
        isExist = os.path.exists(DirOut)
        if not isExist:
            os.makedirs(DirOut)
    # Each sequence sample is parsed to events
    for sample_raw, label in zip(data_raw, labels_raw):
        data_dict_events = {}
        events_per_samples = sample_to_changes(sample_raw, f, Spk_threshold, save=save_fig)
        # Dict of the sample
        data_dict_events['letter'] = label
        data_dict_events['events'] = events_per_samples
        samples.append(data_dict_events)
    print('Finished conversion')
    with open(Events_filename_out, 'wb') as outf:
        pickle.dump(samples, outf)

if __name__ == "__main__":
    main()