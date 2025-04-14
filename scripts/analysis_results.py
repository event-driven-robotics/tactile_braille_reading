import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# let's create the final figure
fig_final = plt.figure(figsize=(16, 10))
gs_final = GridSpec(2, 2, figure=fig_final)

#########################
# INPUT DATA AND TRACES #
#########################
# first epoch
file_path = "./results/traces/first_epoch"
files = os.listdir(file_path)
data_dict_first_epoch = {}
for file in files:
    with open(f"{file_path}/{file}", "rb") as f:
        data = pickle.load(f)
    data = data.cpu().numpy()
    # let's remove the dimension of size 1 and make sur time is the first dimension
    # data_shape = np.array(data.shape)
    # rel_dims = np.where(np.array(data_shape) != 1)[0]
    # # sort by time in first dimension
    # ordererd_dims = data_shape[np.argsort(data_shape)]
    # data = data.reshape(ordererd_dims[-1], ordererd_dims[1])
    data_dict_first_epoch[file.split(".")[0]] = data

# input is the time binned data from the two channels per sensor
# trace_in is the eligibility trace linked to the input neurons (two per channel) and is used to update the weights connecting the input neurons to the hidden layer
# trace_rec is the eligibility trace linked to the recurrent neurons and is used to update the weights connecting the recurrent neurons to the output layer
# recurrent_activity is the activity of the recurrent neurons (hidden layer)

# let's find the most and least active neurons in the input layer adn compare there traces
# sum the spikes per neuron/channel over time
input_activity_first_epoch = np.sum(
    data_dict_first_epoch["input"][:, 0, :], axis=0)
# find max and min active neurons
# most active, second most active, third most active, least active
neurons_of_interest = [-1, -2, -3, 0]
idc_input_activity_first_epoch_ordered = np.argsort(input_activity_first_epoch)
input_activity_first_epoch_ordered = data_dict_first_epoch["input"][:, 0,
                                                                    idc_input_activity_first_epoch_ordered[neurons_of_interest]]  # most active neurons first
input_traces_ordered_first_epoch = data_dict_first_epoch["trace_in"][0,
                                                                     idc_input_activity_first_epoch_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(input_traces_ordered_first_epoch.transpose(),
         label=idc_input_activity_first_epoch_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    input_activity_first_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(
    idc_input_activity_first_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, input_traces_ordered_first_epoch.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_first_epoch_subfig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(input_traces_ordered_first_epoch.transpose(),
        label=idc_input_activity_first_epoch_ordered[neurons_of_interest])
ax.set_title("First epoch")
ax.set_ylabel("Trace")
ax.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax.legend(sorted_handles, sorted_labels)
# ax = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    input_activity_first_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax.scatter(sorted_time_indices, sorted_neuron_indices -
           4, color="black", s=10, label="Spikes")
# ax.set_ylabel("Neuron")
ax.set_xlabel("Time (ms)")
ax.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
neuron_ticks = np.arange(-4, 0, 1)
traces_ticks = np.arange(
    0, int(np.max(input_traces_ordered_first_epoch) + 1.5), 2)
places_ticks = np.append(neuron_ticks, traces_ticks)
tick_labels = np.append(
    idc_input_activity_first_epoch_ordered[neurons_of_interest][sorted_indices], traces_ticks)
ax.set_yticks(places_ticks)
ax.set_yticklabels(tick_labels)
ax.set_ylim(-4.5, int(np.max(input_traces_ordered_first_epoch) + 1.5))
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_first_epoch_single_fig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
# Primary axis for traces
ax1 = fig.add_subplot(111)
ax1.plot(input_traces_ordered_first_epoch.transpose(),
         label=idc_input_activity_first_epoch_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(input_traces_ordered_first_epoch) + 1.5), 2))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(input_traces_ordered_first_epoch) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(input_traces_ordered_first_epoch) + 1.5), 2))
ax1.set_ylim(-5, int(np.max(input_traces_ordered_first_epoch) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    input_activity_first_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron")
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax1.set_xlabel("Time (ms)")
ax2.set_xlim(0, input_traces_ordered_first_epoch.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_input_activity_first_epoch_ordered[neurons_of_interest][sorted_indices])
# Adjust y-axis limits to align with the shifted raster plot
ax2.set_ylim(-5, int(np.max(input_traces_ordered_first_epoch) + 1.5))
# Align and finalize the figure
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_first_epoch_shared_x.pdf",
            bbox_inches="tight")
plt.close(fig)

# RECURRENT NEURONS
# let's find the most and least active neurons in the input layer adn compare there traces
# sum the spikes per neuron/channel over time
recurrent_activity_first_epoch = np.sum(
    data_dict_first_epoch["recurrent_activity"], axis=0)
# find max and min active neurons
# most active, second most active, third most active, least active
neurons_of_interest = [-1, -2, -3, 0]
idc_recurrent_activity_first_epoch_ordered = np.argsort(
    recurrent_activity_first_epoch[0, :])
recurrent_activity_ordered = data_dict_first_epoch["recurrent_activity"][:, 0,
                                                                         idc_recurrent_activity_first_epoch_ordered[neurons_of_interest]]  # most active neurons first
recurrent_traces_first_epoch_ordered = data_dict_first_epoch["trace_rec"][0,
                                                                          idc_recurrent_activity_first_epoch_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(recurrent_traces_first_epoch_ordered.transpose(),
         label=idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, recurrent_traces_first_epoch_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_first_epoch_subfig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
# Primary axis for traces
ax1 = fig.add_subplot(111)
ax1.plot(recurrent_traces_first_epoch_ordered.transpose(),
         label=idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 2))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 2))
ax1.set_ylim(-5, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron")
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest][sorted_indices])
# Adjust y-axis limits to align with the shifted raster plot
ax2.set_ylim(-5, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5))
# Align and finalize the figure
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_first_epoch_shared_x.pdf",
            bbox_inches="tight")
plt.close(fig)

# HERE WE ADD TO THE MAIN FIGURE
# First plot (ax1)
ax1 = fig_final.add_subplot(gs_final[0, 0])
ax1.plot(recurrent_traces_first_epoch_ordered.transpose(),
         label=idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
ax1.set_title("First epoch", fontsize=14)
ax1.set_ylabel("Trace", fontsize=12)
ax1.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 2))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 2))
ax1.set_ylim(-5, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels, loc="upper left", fontsize=12)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron", fontsize=12)
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax1.set_xlabel("Time (ms)", fontsize=12)
ax2.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-5, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5))
# END MAIN FIGURE

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(recurrent_traces_first_epoch_ordered.transpose(),
        label=idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
ax.set_title("First epoch")
ax.set_ylabel("Trace")
ax.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax.legend(sorted_handles, sorted_labels)
# ax = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax.scatter(sorted_time_indices, sorted_neuron_indices -
           4, color="black", s=10, label="Spikes")
# ax.set_ylabel("Neuron")
ax.set_xlabel("Time (ms)")
ax.set_xlim(0, recurrent_traces_first_epoch_ordered.shape[1])
neuron_ticks = np.arange(-4, 0, 1)
traces_ticks = np.arange(
    0, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5), 2)
places_ticks = np.append(neuron_ticks, traces_ticks)
tick_labels = np.append(
    idc_recurrent_activity_first_epoch_ordered[neurons_of_interest][sorted_indices], traces_ticks)
ax.set_yticks(places_ticks)
ax.set_yticklabels(tick_labels)
ax.set_ylim(-4.5, int(np.max(recurrent_traces_first_epoch_ordered) + 1.5))
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_first_epoch_single_fig.pdf",
            bbox_inches="tight")
plt.close(fig)
pass


# last epoch
file_path = "./results/traces/last_epoch"
files = os.listdir(file_path)
data_dict_last_epoch = {}
for file in files:
    with open(f"{file_path}/{file}", "rb") as f:
        data = pickle.load(f)
    data = data.cpu().numpy()
    # let's remove the dimension of size 1 and make sur time is the first dimension
    # data_shape = np.array(data.shape)
    # rel_dims = np.where(np.array(data_shape) != 1)[0]
    # # sort by time in first dimension
    # ordererd_dims = data_shape[np.argsort(data_shape)]
    # data = data.reshape(ordererd_dims[-1], ordererd_dims[1])
    data_dict_last_epoch[file.split(".")[0]] = data

# input is the time binned data from the two channels per sensor
# trace_in is the eligibility trace linked to the input neurons (two per channel) and is used to update the weights connecting the input neurons to the hidden layer
# trace_rec is the eligibility trace linked to the recurrent neurons and is used to update the weights connecting the recurrent neurons to the output layer
# recurrent_activity is the activity of the recurrent neurons (hidden layer)

# let's find the most and least active neurons in the input layer adn compare there traces
# sum the spikes per neuron/channel over time
input_activity_last_epoch = np.sum(
    data_dict_last_epoch["input"][:, 0, :], axis=0)
# find max and min active neurons
# most active, second most active, third most active, least active
neurons_of_interest = [-1, -2, -3, 0]
idc_input_activity_last_epoch_ordered = idc_input_activity_first_epoch_ordered
input_activity_last_epoch_ordered = data_dict_last_epoch["input"][:, 0,
                                                                  idc_input_activity_last_epoch_ordered[neurons_of_interest]]  # most active neurons first
input_traces_ordered_last_epoch = data_dict_last_epoch["trace_in"][0,
                                                                   idc_input_activity_last_epoch_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(input_traces_ordered_last_epoch.transpose(),
         label=idc_input_activity_last_epoch_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    input_activity_last_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(
    idc_input_activity_last_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, input_traces_ordered_last_epoch.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_last_epoch_subfig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(input_traces_ordered_last_epoch.transpose(),
        label=idc_input_activity_last_epoch_ordered[neurons_of_interest])
ax.set_title("Last epoch")
ax.set_ylabel("Trace")
ax.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax.legend(sorted_handles, sorted_labels)
# ax = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    input_activity_last_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax.scatter(sorted_time_indices, sorted_neuron_indices -
           4, color="black", s=10, label="Spikes")
# ax.set_ylabel("Neuron")
ax.set_xlabel("Time (ms)")
ax.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
neuron_ticks = np.arange(-4, 0, 1)
traces_ticks = np.arange(
    0, int(np.max(input_traces_ordered_last_epoch) + 1.5), 2)
places_ticks = np.append(neuron_ticks, traces_ticks)
tick_labels = np.append(
    idc_input_activity_last_epoch_ordered[neurons_of_interest][sorted_indices], traces_ticks)
ax.set_yticks(places_ticks)
ax.set_yticklabels(tick_labels)
ax.set_ylim(-4.5, int(np.max(input_traces_ordered_last_epoch) + 1.5))
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_last_epoch_single_fig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
# Primary axis for traces
ax1 = fig.add_subplot(111)
ax1.plot(input_traces_ordered_last_epoch.transpose(),
         label=idc_input_activity_last_epoch_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(input_traces_ordered_last_epoch) + 1.5), 2))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(input_traces_ordered_last_epoch) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(input_traces_ordered_last_epoch) + 1.5), 2))
ax1.set_ylim(-5, int(np.max(input_traces_ordered_last_epoch) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    input_activity_last_epoch_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_input_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron")
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax1.set_xlabel("Time (ms)")
ax2.set_xlim(0, input_traces_ordered_last_epoch.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_input_activity_last_epoch_ordered[neurons_of_interest][sorted_indices])
# Adjust y-axis limits to align with the shifted raster plot
ax2.set_ylim(-5, int(np.max(input_traces_ordered_last_epoch) + 1.5))
# Align and finalize the figure
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/input_analysis_last_epoch_shared_x.pdf",
            bbox_inches="tight")
plt.close(fig)

# RECURRENT NEURONS
# let's find the most and least active neurons in the input layer adn compare there traces
# sum the spikes per neuron/channel over time
recurrent_activity_last_epoch = np.sum(
    data_dict_last_epoch["recurrent_activity"], axis=0)
# find max and min active neurons
# most active, second most active, third most active, least active
neurons_of_interest = [-1, -2, -3, 0]
idc_recurrent_activity_last_epoch_ordered = idc_recurrent_activity_first_epoch_ordered
recurrent_activity_ordered = data_dict_last_epoch["recurrent_activity"][:, 0,
                                                                        idc_recurrent_activity_last_epoch_ordered[neurons_of_interest]]  # most active neurons first
recurrent_traces_last_epoch_ordered = data_dict_last_epoch["trace_rec"][0,
                                                                        idc_recurrent_activity_last_epoch_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(recurrent_traces_last_epoch_ordered.transpose(),
         label=idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, recurrent_traces_last_epoch_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_last_epoch_subfig.pdf",
            bbox_inches="tight")
plt.close(fig)

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
# Primary axis for traces
ax1 = fig.add_subplot(111)
ax1.plot(recurrent_traces_last_epoch_ordered.transpose(),
         label=idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 1))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 1))
ax1.set_ylim(-5, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron")
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest][sorted_indices])
# Adjust y-axis limits to align with the shifted raster plot
ax2.set_ylim(-5, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5))
# Align and finalize the figure
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_last_epoch_shared_x.pdf",
            bbox_inches="tight")
plt.close(fig)

# HERE WE ADD TO THE MAIN FIGURE
# First plot (ax1)
ax1 = fig_final.add_subplot(gs_final[1, 0])
ax1.plot(recurrent_traces_last_epoch_ordered.transpose(),
         label=idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
ax1.set_title("Last epoch", fontsize=14)
ax1.set_ylabel("Trace", fontsize=12)
ax1.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
ax1.set_yticks(
    np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 2))
# list_of_empty_lists = [[] for _ in range(5)]
# for i in np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 1):
#     list_of_empty_lists.append(i)
ax1.set_yticklabels(
    np.arange(0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 2))
ax1.set_ylim(-5, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5))
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
ax1.legend(sorted_handles, sorted_labels, loc="upper left", fontsize=12)
# Secondary axis for spike raster plot
ax2 = ax1.twinx()
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new-4 for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices,
            color="black", s=10, label="Spikes")
# Adjust the secondary axis
ylabel = ax2.set_ylabel("Neuron", fontsize=12)
ylabel.set_rotation(270)  # Rotate the label 270 degrees
ax1.set_xlabel("Time (ms)", fontsize=12)
ax2.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
ax2.set_yticks(np.arange(-4, 0, 1))
ax2.set_yticklabels(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-5, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5))
# END MAIN FIGURE

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(recurrent_traces_last_epoch_ordered.transpose(),
        label=idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
ax.set_title("Last epoch")
ax.set_ylabel("Trace")
ax.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
for i in range(len(sorted_labels)):
    sorted_labels[i] = "Neuron ID: " + sorted_labels[i]
# Reassign the legend with sorted labels and corresponding handles
ax.legend(sorted_handles, sorted_labels)
# ax = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(
    recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array(
    [mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax.scatter(sorted_time_indices, sorted_neuron_indices -
           4, color="black", s=10, label="Spikes")
# ax.set_ylabel("Neuron")
ax.set_xlabel("Time (ms)")
ax.set_xlim(0, recurrent_traces_last_epoch_ordered.shape[1])
neuron_ticks = np.arange(-4, 0, 1)
traces_ticks = np.arange(
    0, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5), 2)
places_ticks = np.append(neuron_ticks, traces_ticks)
tick_labels = np.append(
    idc_recurrent_activity_last_epoch_ordered[neurons_of_interest][sorted_indices], traces_ticks)
ax.set_yticks(places_ticks)
ax.set_yticklabels(tick_labels)
ax.set_ylim(-4.5, int(np.max(recurrent_traces_last_epoch_ordered) + 1.5))
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/recurrent_analysis_last_epoch_single_fig.pdf",
            bbox_inches="tight")
plt.close(fig)
pass


####################
# WEIGHTS ANALYSIS #
####################
# EPOCH 0
file_path = "./results/weights/data_plot_3batch_size_epoch0"
files = os.listdir(file_path)
data_dict_batches = {}
data_dict_batches["epoch_0"] = {}
# Open the file in read-binary mode
for file in tqdm(files):
    data = []
    with open(f"{file_path}/{file}", 'rb') as inf:
        while True:
            try:
                obj = pickle.load(inf)
                obj = obj.detach().cpu().numpy()  # Load each object
                data.append(obj)  # Store it in the list
            except EOFError:
                break  # Stop when reaching the end of the file

    data_dict_batches["epoch_0"][file.split(".")[0]] = np.array(data)

# EPOCH 0
file_path = "./results/weights/data_plot_3batch_size_epoch21"
files = os.listdir(file_path)
data_dict_batches["epoch_21"] = {}
# Open the file in read-binary mode
for file in tqdm(files):
    data = []
    with open(f"{file_path}/{file}", 'rb') as inf:
        while True:
            try:
                obj = pickle.load(inf)
                obj = obj.detach().cpu().numpy()  # Load each object
                data.append(obj)  # Store it in the list
            except EOFError:
                break  # Stop when reaching the end of the file

    data_dict_batches["epoch_21"][file.split(".")[0]] = np.array(data)

sum_change_in = np.abs(
    np.sum(data_dict_batches["epoch_0"]["update_value_w_in"], axis=0))
idc_max_change_in = np.where(sum_change_in == np.max(sum_change_in))
sum_change_rec = np.abs(
    np.sum(data_dict_batches["epoch_0"]["update_value_w_rec"], axis=0))
idc_max_change_rec = np.where(sum_change_rec == np.max(sum_change_rec))
sum_change_out = np.abs(
    np.sum(data_dict_batches["epoch_0"]["update_value_w_out"], axis=0))
idc_max_change_out = np.where(sum_change_out == np.max(sum_change_out))

# let's plot the weight change over batches for the layers
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2, 1, 1)
w_in = data_dict_batches["epoch_0"]["w_in"][:,
                                            idc_max_change_in[0], idc_max_change_in[1]][:, 0]
w_rec = data_dict_batches["epoch_0"]["w_rec"][:,
                                              idc_max_change_rec[0], idc_max_change_rec[1]][:, 0]
w_out = data_dict_batches["epoch_0"]["w_out"][:,
                                              idc_max_change_out[0], idc_max_change_out[1]][:, 0]
label = r"$\mathcal{W}_{\mathrm{in}}^{(" + f"{idc_max_change_in[0][0]}, {idc_max_change_in[1][0]}" + ")}$"
ax1.plot(range(1, w_in.shape[0]+1), w_in, label=label)
label = r"$\mathcal{W}_{\mathrm{rec}}^{(" + f"{idc_max_change_rec[0][0]}, {idc_max_change_rec[1][0]}" + ")}$"
ax1.plot(range(1, w_in.shape[0]+1), w_rec, label=label)
label = r"$\mathcal{W}_{\mathrm{out}}^{(" + f"{idc_max_change_out[0][0]}, {idc_max_change_out[1][0]}" + ")}$"
ax1.plot(range(1, w_in.shape[0]+1), w_out, label=label)
ax1.set_title("First epoch")
ax1.set_ylabel("Weight")
ax1.set_xlim(1, w_in.shape[0])
ax1.legend(loc="upper right")

# HERE WE ADD TO THE MAIN FIGURE
ax5 = fig_final.add_subplot(gs_final[0, 1])
label = r"$\mathcal{W}_{\mathrm{in}}^{(" + f"{idc_max_change_in[0][0]}, {idc_max_change_in[1][0]}" + ")}$"
ax5.plot(range(1, w_in.shape[0]+1), w_in, label=label)
label = r"$\mathcal{W}_{\mathrm{rec}}^{(" + f"{idc_max_change_rec[0][0]}, {idc_max_change_rec[1][0]}" + ")}$"
ax5.plot(range(1, w_in.shape[0]+1), w_rec, label=label)
label = r"$\mathcal{W}_{\mathrm{out}}^{(" + f"{idc_max_change_out[0][0]}, {idc_max_change_out[1][0]}" + ")}$"
ax5.plot(range(1, w_in.shape[0]+1), w_out, label=label)
ax5.set_title("Weight change over batches", fontsize=14)
ax5.set_ylabel("Weight", fontsize=12)
ax5.set_xlabel("Batch", fontsize=12)
ax5.set_xlim(1, w_in.shape[0])
ax5.legend(loc="center right", fontsize=12)

ax2 = fig.add_subplot(2, 1, 2)
w_in = data_dict_batches["epoch_21"]["w_in"][:,
                                             idc_max_change_in[0], idc_max_change_in[1]][:, 0]
w_rec = data_dict_batches["epoch_21"]["w_rec"][:,
                                               idc_max_change_rec[0], idc_max_change_rec[1]][:, 0]
w_out = data_dict_batches["epoch_21"]["w_out"][:,
                                               idc_max_change_out[0], idc_max_change_out[1]][:, 0]
label = r"$\mathcal{W}_{\mathrm{in}}^{(" + f"{idc_max_change_in[0][0]}, {idc_max_change_in[1][0]}" + ")}$"
ax2.plot(range(1, w_in.shape[0]+1), w_in, label=label)
label = r"$\mathcal{W}_{\mathrm{rec}}^{(" + f"{idc_max_change_rec[0][0]}, {idc_max_change_rec[1][0]}" + ")}$"
ax2.plot(range(1, w_in.shape[0]+1), w_rec, label=label)
label = r"$\mathcal{W}_{\mathrm{out}}^{(" + f"{idc_max_change_out[0][0]}, {idc_max_change_out[1][0]}" + ")}$"
ax2.plot(range(1, w_in.shape[0]+1), w_out, label=label)
ax2.set_title("Last epoch (21)")
ax2.set_ylabel("Weight")
ax2.set_xlabel("Batch")
ax2.set_xlim(1, w_in.shape[0])
ax2.legend(loc="upper right")
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/weights_change_over_batches.pdf", bbox_inches="tight")
plt.close(fig)

# weights over epochs
# first epoch
file_path = "./results/weights/data_weight_20epoch"
files = os.listdir(file_path)
data_dict_epochs = {}
# Open the file in read-binary mode
for file in tqdm(files):
    data = []
    with open(f"{file_path}/{file}", 'rb') as inf:
        while True:
            try:
                obj = pickle.load(inf)
                obj = obj.detach().cpu().numpy()  # Load each object
                data.append(obj)  # Store it in the list
            except EOFError:
                break  # Stop when reaching the end of the file

    data_dict_epochs[file.split(".")[0]] = np.array(data)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
label = r"$\mathcal{W}_{\mathrm{in}}^{(" + f"{idc_max_change_in[0][0]}, {idc_max_change_in[1][0]}" + ")}$"
ax.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_in"], label=label)
label = r"$\mathcal{W}_{\mathrm{rec}}^{(" + f"{idc_max_change_rec[0][0]}, {idc_max_change_rec[1][0]}" + ")}$"
ax.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_rec"], label=label)
label = r"$\mathcal{W}_{\mathrm{out}}^{(" + f"{idc_max_change_out[0][0]}, {idc_max_change_out[1][0]}" + ")}$"
ax.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_out"], label=label)
ax.set_title("Weight change over epochs")
ax.set_ylabel("Weight")
ax.set_xlabel("Epoch")
ax.set_xticks(np.arange(1, data_dict_epochs["w_in"].shape[0]+1, 1))
ax.set_xticklabels(np.arange(1, data_dict_epochs["w_in"].shape[0]+1, 1))
ax.set_xlim(1, data_dict_epochs["w_in"].shape[0])
ax.legend(loc="upper right")
fig.savefig("./figures/e_prop_analysis/weights_change_over_epochs.pdf",
            bbox_inches="tight")
plt.close(fig)

# HERE WE ADD TO THE MAIN FIGURE
# ax6 = fig_final.add_subplot(gs_final[2:4, 1])
# ax6.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_in"],
#         label=f"w_in ({idc_max_change_in[0][0]}, {idc_max_change_in[1][0]})")
# ax6.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_rec"],
#         label=f"w_rec ({idc_max_change_rec[0][0]}, {idc_max_change_rec[1][0]})")
# ax6.plot(range(1, data_dict_epochs["w_in"].shape[0]+1), data_dict_epochs["w_out"],
#         label=f"w_out ({idc_max_change_out[0][0]}, {idc_max_change_out[1][0]})")
# ax6.set_title("Weight change over epochs", fontsize=14)
# ax6.set_ylabel("Weight", fontsize=12)
# ax6.set_xlabel("Epoch", fontsize=12)
# ax6.set_xticks(np.arange(1, data_dict_epochs["w_in"].shape[0]+1, 1))
# ax6.set_xticklabels(np.arange(1, data_dict_epochs["w_in"].shape[0]+1, 1))
# ax6.set_xlim(1, data_dict_epochs["w_in"].shape[0])
# ax6.legend(loc="upper right", fontsize=12)

# ACCURACY
file_path = "./results/accuracy"
# here we must read a txt file
fila_name = "test_eprop_5test.txt"
acc_data = []
new_data = []  # train, test
start_new_trial = True
# we need to read the file line by line
with open(f"{file_path}/{fila_name}", "r") as f:
    for line in tqdm(f):
        if "Final results" in line:
            acc_data.append(new_data)
            new_data = []
        if "Train acc" in line:
            # Split the line into parts and convert to float
            parts = line.strip().split()
            # Convert the parts to float and append to the list
            new_data.append([float(parts[2]), float(parts[5])])

# Convert the list to a numpy array
acc_data = np.array(acc_data)
mean_acc, std_acc = np.mean(acc_data, axis=0), np.std(
    acc_data, axis=0)  # train, test
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.fill_between(range(1, mean_acc.shape[0]+1),
                mean_acc[:, 0] - std_acc[:, 0], mean_acc[:, 0] + std_acc[:, 0], alpha=0.2, label="Train std")
ax.plot(range(1, mean_acc.shape[0]+1), mean_acc[:, 0], label="Train avg")
ax.fill_between(range(1, mean_acc.shape[0]+1),
                mean_acc[:, 1] - std_acc[:, 1], mean_acc[:, 1] + std_acc[:, 1], alpha=0.2, label="Test std")
ax.plot(range(1, mean_acc.shape[0]+1), mean_acc[:, 1], label="Test avg")
ax.set_xticks(np.arange(1, mean_acc.shape[0]+1, 1))
ax.set_xticklabels(np.arange(1, mean_acc.shape[0]+1, 1))
ax.set_ylim(0, 100)
ax.set_xlim(1, mean_acc.shape[0])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training")
ax.legend()
fig.tight_layout()
fig.savefig("./figures/e_prop_analysis/accuracy_mean_std.pdf", bbox_inches="tight")
plt.close(fig)

# HERE WE ADD TO THE MAIN FIGURE
ax7 = fig_final.add_subplot(gs_final[1, 1])
ax7.fill_between(range(1, mean_acc.shape[0]+1),
                mean_acc[:, 0] - std_acc[:, 0], mean_acc[:, 0] + std_acc[:, 0], alpha=0.2, label="Train std")
ax7.plot(range(1, mean_acc.shape[0]+1), mean_acc[:, 0], label="Train avg")
ax7.fill_between(range(1, mean_acc.shape[0]+1),
                mean_acc[:, 1] - std_acc[:, 1], mean_acc[:, 1] + std_acc[:, 1], alpha=0.2, label="Test std")
ax7.plot(range(1, mean_acc.shape[0]+1), mean_acc[:, 1], label="Test avg")
ax7.set_xticks(np.arange(1, mean_acc.shape[0]+1, 1))
ax7.set_xticklabels(np.arange(1, mean_acc.shape[0]+1, 1))
ax7.set_ylim(0, 100)
ax7.set_xlim(1, mean_acc.shape[0])
ax7.set_xlabel("Epoch", fontsize=12)
ax7.set_ylabel("Accuracy", fontsize=12)
ax7.set_title("Training", fontsize=14)
ax7.legend(loc="lower right", fontsize=12)

# FINALIZE THE FINAL FIGURE
fig_final.align_labels()
fig_final.tight_layout()
fig_final.savefig("./figures/e_prop_analysis/e_prop_learning_neuron_to_epoch.pdf", bbox_inches="tight")
plt.close(fig_final)
