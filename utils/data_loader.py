"""data_loader.py

Data loading and preprocessing utilities for braille letter classification.

Provides functions to load event-based tactile sensor data from pickle files, 
preprocess and bin the data into time windows, filter by specific tactile sensors 
and letters, and create train/test/validation datasets for spiking neural network 
training. Supports both mechanoreceptor encoding (FA/SA channels) and sigma-delta 
encoding (ON/OFF channels) schemes.

Key Features:
- Event-based data binning into discrete time windows
- Taxel (tactile sensor) selection for targeted channel filtering
- Stratified train/test/validation splits with configurable ratios
- Automatic parameter updates for network input dimensions
- Support for both encoding schemes with consistent API

Author: Simon F. Muller-Cleve
Date: January 15, 2026
"""

import logging
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# Get logger instance
logger = logging.getLogger('braille_training')


def load_and_extract(params: dict, file_name: str, letter_written: list) -> tuple:
    """
    Load and preprocess braille letter tactile sensor data from a pickle file.

    This function loads event-based tactile sensor data, bins it into time windows,
    optionally filters by specific taxels (tactile sensors) and letters, and creates
    train/test (and optionally validation) datasets. The function also updates the
    params dictionary with the actual number of input channels based on the selected
    taxels and encoding scheme.

    Parameters
    ----------
    params : dict
        Dictionary containing experimental parameters (modified in-place):
        - 'max_time' : int
            Maximum time duration in milliseconds
        - 'time_bin_size' : int
            Size of time bins in milliseconds
        - 'encoding_type' : str
            If 'mechanoreceptor', uses mechanoreceptor encoding (FA/SA channels);
            If 'sigma-delta', uses sigma-delta encoding (ON/OFF channels)
        - 'selected_channels' : list of int or None
            List of taxel indices to include (0-indexed). If None, all taxels are used.
        - 'dtype_torch' : torch.dtype
            PyTorch data type for tensors
        - 'validation' : bool
            If True, creates 70/20/10 train/test/validation split;
            If False, creates 80/20 train/test split
        - 'debug' : bool
            If True, prints detailed label distribution information

        The following parameters are SET by this function:
        - 'time_step' : float
            Time step in seconds (time_bin_size * 0.001)
        - 'data_steps' : int
            Number of time steps in the binned data
        - 'delayed_output' : None
            Placeholder for delayed output (currently set to None)
        - 'nb_inputs' : int
            Actual number of input channels after taxel selection
            (2 * number of selected taxels for both encoding schemes)

    file_name : str
        Path to the pickle file containing braille letter data.
        For mechanoreceptor encoding: expects dict with keys
            ['letter', 'fa_spikes', 'sa_spikes', 'taxel_data']
        For sigma-delta encoding: expects dict/DataFrame with keys
            ['letter', 'events']

    letter_written : list of str
        List of letter labels to include in the dataset (e.g., ['a', 'b', 'c']).
        Only samples with these labels are included. Letters are mapped to integer
        indices based on their position in this list.

    Returns
    -------
    tuple
        (ds_train, ds_test, ds_validation, labels) where:

        - ds_train : TensorDataset
            Training dataset containing (data, labels) pairs
            Data shape: [n_train_samples, time_steps, n_channels]
        - ds_test : TensorDataset
            Test dataset containing (data, labels) pairs
            Data shape: [n_test_samples, time_steps, n_channels]
        - ds_validation : TensorDataset or None
            Validation dataset (None if params['validation']=False)
            Data shape: [n_val_samples, time_steps, n_channels]
        - labels : torch.Tensor
            All label values in the full dataset (before split), shape: [n_total_samples]

    Notes
    -----
    **Encoding Schemes:**

    - Mechanoreceptor encoding: Each taxel produces 2 channels (FA and SA)
      - FA (Fast Adapting): Responds to changes in tactile input
      - SA (Slow Adapting): Responds to sustained pressure
      - Channel ordering: [FA_taxel0, FA_taxel1, ..., SA_taxel0, SA_taxel1, ...]

    - Sigma-delta encoding: Each taxel produces 2 channels (ON and OFF)
      - ON channel: Positive changes in sensor activation
      - OFF channel: Negative changes in sensor activation
      - Channel ordering: [ON_taxel0, OFF_taxel0, ON_taxel1, OFF_taxel1, ...]

    **Data Processing:**

    - Events are binned into discrete time windows of size time_bin_size
    - Binary spike representation: 1 if event occurred in time bin, 0 otherwise
    - Data shape: [batch, time_steps, channels]
    - Labels are converted to zero-indexed integers based on letter_written order

    **Dataset Splits:**

    - All splits use stratified sampling to maintain class balance
    - If validation=True: 70% train, 20% test, 10% validation
    - If validation=False: 80% train, 20% test, validation is None

    **Side Effects:**

    - Modifies params dict in-place with 'time_step', 'data_steps', 'delayed_output', 'nb_inputs'
    - These updates occur before any layer initialization should happen

    **Debug Output:**

    - If params['debug']=True, prints label distributions before and after splits
    - Shows counts and percentages for each class in train/test/validation sets

    Examples
    --------
    >>> params = {
    ...     'max_time': 1000,
    ...     'time_bin_size': 10,
    ...     'encoding_type': 'mechanoreceptor',
    ...     'selected_channels': [0, 1, 2],  # Use first 3 taxels
    ...     'dtype_torch': torch.float32,
    ...     'validation': True,
    ...     'debug': False
    ... }
    >>> letters = ['a', 'b', 'c']
    >>> train, test, val, labels = load_and_extract(params, 'data.pkl', letters)
    >>> print(params['nb_inputs'])  # Will be 6 (3 taxels * 2 channels)
    6
    """

    max_time = params["max_time"]  # ms
    # int(params['time_bin_size'])  # so far from laoded file, but can be set manually here
    time_bin_size = params["time_bin_size"]
    time = range(0, max_time, time_bin_size)

    params["time_step"] = time_bin_size*0.001  # seconds
    params["data_steps"] = len(time)
    # params["delayed_output"] = data_steps
    params["delayed_output"] = None  # 0  # data_steps
    
    logger.debug(f"Loading data from: {file_name}")
    logger.debug(f"Max time: {max_time}ms, Time bin size: {time_bin_size}ms, Data steps: {params['data_steps']}")
    logger.debug(f"Encoding: {'Mechanoreceptor' if params['encoding_type'] == 'mechanoreceptor' else 'Sigma-delta'}")
    logger.debug(f"Selected channels/taxels: {params['selected_channels']}")

    # Extract data
    import os
    data = []
    labels = []
    # Check file existence
    if not os.path.isfile(file_name):
        logger.error(f"Data file not found: {file_name}")
        raise FileNotFoundError(f"Data file not found: {file_name}")
    try:
        if params['encoding_type'] == 'mechanoreceptor':
            with open(file_name, "rb") as f:
                data_dict = pkl.load(f)
            logger.debug(f"Loaded mechanoreceptor data with {len(data_dict['letter'])} samples")
            nchan = len(data_dict['taxel_data'][0][0])

            # Determine which channels to use
            if params["selected_channels"] is not None:
                selected_taxels = params["selected_channels"]
                selected_chans = 2 * len(params["selected_channels"])
            else:
                selected_taxels = list(range(nchan))
                selected_chans = 2 * nchan

            params["nb_inputs"] = selected_chans
            logger.debug(f"Total input channels: {selected_chans} (taxels: {len(selected_taxels)}, 2 channels per taxel)")

            filtered_out = 0
            for letter, fa_spikes, sa_spikes in zip(data_dict['letter'], data_dict['fa_spikes'], data_dict['sa_spikes']):
                events_array = np.zeros(
                    [selected_chans, round((max_time/time_bin_size)+0.5)])

                fa_spike_times, fa_spike_idc = (
                    fa_spikes[:, 0]*1000).astype(int), fa_spikes[:, 1].astype(int)-1
                sa_spike_times, sa_spike_idc = (
                    sa_spikes[:, 0]*1000).astype(int), (sa_spikes[:, 1].astype(int)-1)

                for local_idx, taxel in enumerate(selected_taxels):
                    spike_times = fa_spike_times[fa_spike_idc == taxel]
                    if spike_times.size > 0:
                        idc = np.array(
                            (spike_times/time_bin_size).round(), dtype=int)
                        idc = idc[idc < max_time//time_bin_size]
                        events_array[local_idx, idc] = 1

                    spike_times = sa_spike_times[sa_spike_idc == taxel]
                    if spike_times.size > 0:
                        idc = np.array(
                            (spike_times/time_bin_size).round(), dtype=int)
                        idc = idc[idc < max_time//time_bin_size]
                        events_array[local_idx + len(selected_taxels), idc] = 1

                if letter in letter_written:
                    data.append(events_array.T)
                    try:
                        labels.append(letter_written.index(letter))
                    except ValueError:
                        logger.debug(f"Letter '{letter}' not found in letter_written list. Skipping.")
                        filtered_out += 1
                else:
                    filtered_out += 1
            logger.debug(f"Filtered out {filtered_out} samples not in letter_written.")
        else:
            data_dict = pd.read_pickle(file_name)
            data_dict = pd.DataFrame(data_dict)
            bins = 1000  # ms conversion
            nchan = len(data_dict['events'][1])
            if params["selected_channels"] is not None:
                selected_chans = 2 * len(params["selected_channels"])
            else:
                selected_chans = 2 * nchan
            params["nb_inputs"] = selected_chans
            filtered_out = 0
            for i, sample in enumerate(data_dict['events']):
                events_array = np.zeros(
                    [nchan, round((max_time/time_bin_size)+0.5), 2])
                for taxel in range(len(sample)):
                    for event_type in range(len(sample[taxel])):
                        if sample[taxel][event_type]:
                            indx = bins*(np.array(sample[taxel][event_type]))
                            indx = np.array(
                                (indx/time_bin_size).round(), dtype=int)
                            events_array[taxel, indx, event_type] = 1
                if params["selected_channels"] is not None:
                    events_array = np.reshape(np.transpose(events_array, (1, 0, 2))[
                        :, params["selected_channels"], :], (events_array.shape[1], -1))
                else:
                    events_array = np.reshape(np.transpose(
                        events_array, (1, 0, 2)), (events_array.shape[1], -1))

                letter_val = data_dict['letter'][i]
                if letter_val in letter_written:
                    data.append(events_array)
                    try:
                        labels.append(letter_written.index(letter_val))
                    except ValueError:
                        logger.debug(f"Letter '{letter_val}' not found in letter_written list. Skipping.")
                        filtered_out += 1
                else:
                    filtered_out += 1
            logger.debug(f"Filtered out {filtered_out} samples not in letter_written.")
    except Exception as e:
        logger.error(f"Failed to load or process data from {file_name}: {e}")
        logger.debug("Exception details:", exc_info=True)
        raise

    # return data,labels
    data = np.array(data)
    labels = np.array(labels)
    # print(labels)
    data = torch.tensor(data, dtype=params['dtype_torch'])
    labels = torch.tensor(labels, dtype=torch.long)

    if params['debug']:
        logger.debug(f"\n=== Label Distribution Before Split ===")
        unique_labels, counts = np.unique(
            labels.cpu().numpy(), return_counts=True)
        total_samples = len(labels)
        for label_idx, count in zip(unique_labels, counts):
            label_name = letter_written[label_idx] if label_idx < len(letter_written) else f"Unknown({label_idx})"
            percentage = (count / total_samples) * 100
            logger.debug(
                f"  Label {label_idx} ({label_name}): {count} samples ({percentage:.1f}%)")
        logger.debug(f"  Total: {total_samples} samples")

    if params['validation']:
        # create 70/20/10 train/test/validation split
        # first create 70/30 train/(test + validation)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.30, shuffle=True, stratify=labels)
        # split test and validation 2/1
        x_test, x_validation, y_test, y_validation = train_test_split(
            x_test, y_test, test_size=0.33, shuffle=True, stratify=y_test)

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)
        ds_validation = TensorDataset(x_validation, y_validation)

    else:
        # create 80/20 train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.20, shuffle=True, stratify=labels)

        ds_train = TensorDataset(x_train, y_train)
        ds_test = TensorDataset(x_test, y_test)
        ds_validation = None

    if params['debug']:
        print(f"\n=== Label Distribution After Split ===")
        print(f"\nTraining set:")
        unique_train, counts_train = np.unique(
            y_train.cpu().numpy(), return_counts=True)
        for label_idx, count in zip(unique_train, counts_train):
            label_name = letter_written[label_idx]
            percentage = (count / len(y_train)) * 100
            print(
                f"  Label {label_idx} ({label_name}): {count} samples ({percentage:.1f}%)")

        print(f"\nTest set:")
        unique_test, counts_test = np.unique(
            y_test.cpu().numpy(), return_counts=True)
        for label_idx, count in zip(unique_test, counts_test):
            label_name = letter_written[label_idx]
            percentage = (count / len(y_test)) * 100
            print(
                f"  Label {label_idx} ({label_name}): {count} samples ({percentage:.1f}%)")

        if params['validation']:
            print(f"\nValidation set:")
            unique_val, counts_val = np.unique(
                y_validation.cpu().numpy(), return_counts=True)
            for label_idx, count in zip(unique_val, counts_val):
                label_name = letter_written[label_idx]
                percentage = (count / len(y_validation)) * 100
                print(
                    f"  Label {label_idx} ({label_name}): {count} samples ({percentage:.1f}%)")

        print()  # blank line for readability

    return ds_train, ds_test, ds_validation, labels
