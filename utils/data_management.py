import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def load_and_extract(params, file_name, taxels=None, letter_written=letters, prepare_validation=False):
    """
    Load Braille event data, extract spike arrays per taxel, and prepare PyTorch datasets.

    Args:
        params (dict):
            'max_time' (int): Maximum recording time in milliseconds.
            'time_bin_size' (int): Time bin size in milliseconds.
        file_name (str): Path to the pickle file containing event data and labels.
        taxels (list[int] or None): Indices of taxels to select; if None, all taxels are used.
        letter_written (list[str]): List of letters to map labels; defaults to the global `letters`.
        prepare_validation (bool): 
            If False, returns train/test split (70/30).
            If True, returns train/validation/test split (70/20/10).

    Returns:
        If prepare_validation is False:
            ds_train (TensorDataset): Training dataset of shape (n_train, time_steps, n_channels).
            ds_test (TensorDataset): Test dataset of shape (n_test, time_steps, n_channels).
            labels (Tensor): All labels before splitting.
            selected_chans (int): Number of channels per sample (2 x number of taxels).
            data_steps (int): Number of time bins per sample.
        If prepare_validation is True:
            ds_train (TensorDataset): Training dataset.
            ds_test (TensorDataset): Test dataset.
            ds_validation (TensorDataset): Validation dataset.
            labels (Tensor): All labels before splitting.
            selected_chans (int): Number of channels per sample.
            data_steps (int): Number of time bins per sample.
    """

    max_time = int(params['max_time'])  # ms
    # so far from laoded file, but can be set manually here
    time_bin_size = int(params['time_bin_size'])

    data_steps = len(range(0, max_time, time_bin_size))

    global delayed_output
    delayed_output = data_steps

    data_dict = pd.read_pickle(file_name)
    data_dict = pd.DataFrame(data_dict)
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

    # Print average number of spikes for each letter
    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    if prepare_validation:
        # create 70/20/10 train/validation/test split
        # first create 70/30 train/(validation + test)
        x_train, x_temp, y_train, y_temp = train_test_split(
            data, labels, test_size=0.30, shuffle=True, stratify=labels)
        # split temp into validation and test 2/1
        x_validation, x_test, y_validation, y_test = train_test_split(
            x_temp, y_temp, test_size=0.33, shuffle=True, stratify=y_temp)

        ds_train = TensorDataset(x_train, y_train)
        ds_validation = TensorDataset(x_validation, y_validation)
        ds_test = TensorDataset(x_test, y_test)

        return ds_train, ds_test, ds_validation, labels, selected_chans, data_steps

    else:
        # create 70/30 train/test
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.30, shuffle=True, stratify=labels)

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)

        return ds_train, ds_test, labels, selected_chans, data_steps
