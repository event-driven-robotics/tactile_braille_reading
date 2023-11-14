'''
Here we prepare the data.
'''

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_and_extract(params, file_name, taxels=None, letter_written=None, dtype=torch.float):

    max_time = int(51*25)  # ms old dataset
    # max_time = int(350*10)  # ms new dataset
    time_bin_size = int(params['time_bin_size'])  # ms
    global time
    time = range(0, max_time, time_bin_size)

    global time_step
    time_step = time_bin_size*0.001
    data_steps = len(time)

    data_dict = pd.read_pickle(file_name)

    # Extract data
    data = []
    labels = []
    bins = 1000  # ms conversion
    nchan = len(data_dict['events'][1])  # number of channels per sensor
    # loop over all trials
    for i, sample in enumerate(data_dict['events']):
        events_array = np.zeros(
            [nchan, round((max_time/time_bin_size)+0.5), 2])
        # loop over sensors (taxel)
        for taxel in range(len(sample)):
            # loop over On and Off channels
            for event_type in range(len(sample[taxel])):
                if sample[taxel][event_type]:
                    indx = bins*(np.array(sample[taxel][event_type]))
                    indx = np.array((indx/time_bin_size).round(), dtype=int)
                    events_array[taxel, indx, event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[
                                      :, taxels, :], (events_array.shape[1], -1))
            selected_chans = 2*len(taxels)
        else:
            events_array = np.reshape(np.transpose(
                events_array, (1, 0, 2)), (events_array.shape[1], -1))
            selected_chans = 2*nchan
        data.append(events_array)
        labels.append(letter_written.index(data_dict['letter'][i]))

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)
    # labels)
    data = torch.tensor(data, dtype=dtype)
    labels = torch.tensor(labels, dtype=torch.long)

    # create 70/20/10 train/test/validation split
    # first create 70/30 train/(test + validation)
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(
        data, labels, test_size=0.30, shuffle=True, stratify=labels)
    # split test and validation 2/1
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_validation_test, y_validation_test, test_size=0.33, shuffle=True, stratify=y_validation_test)

    ds_train = TensorDataset(x_train, y_train)
    ds_validation = TensorDataset(x_validation, y_validation)
    ds_test = TensorDataset(x_test, y_test)

    return ds_train, ds_validation, ds_test, labels, selected_chans, data_steps, time_step
