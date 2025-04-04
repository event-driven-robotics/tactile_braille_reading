import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# let's create the final figure 
fig_final = plt.figure(figsize=(10, 6))
gs_final = GridSpec(8, 2, figure=fig_final)

#########################
# INPUT DATA AND TRACES #
#########################
# first epoch
file_path = "./results/traces/first_epoch"
files = os.listdir(file_path)
data_dict = {}
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
    data_dict[file.split(".")[0]] = data

# input is the time binned data from the two channels per sensor
# trace_in is the eligibility trace linked to the input neurons (two per channel) and is used to update the weights connecting the input neurons to the hidden layer
# trace_rec is the eligibility trace linked to the recurrent neurons and is used to update the weights connecting the recurrent neurons to the output layer
# recurrent_activity is the activity of the recurrent neurons (hidden layer)

# let's find the most and least active neurons in the input layer adn compare there traces
input_activity = np.sum(data_dict["input"][:, 0, :], axis=0)  # sum the spikes per neuron/channel over time
# find max and min active neurons
neurons_of_interest = [-1, -2, -3, 0]  # most active, second most active, third most active, least active
idc_input_activity_ordered = np.argsort(input_activity)
input_activity_ordered = data_dict["input"][:, 0, idc_input_activity_ordered[neurons_of_interest]]  # most active neurons first
traces_ordered = data_dict["trace_in"][0, idc_input_activity_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(traces_ordered.transpose(), label=idc_input_activity_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Trace")
ax1.set_xlim(0, traces_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(input_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(idc_input_activity_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array([mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, traces_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(idc_input_activity_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, traces_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/input_analysis_first_epoch.pdf", bbox_inches="tight")
plt.close(fig)


# let's find the most and least active neurons in the input layer adn compare there traces
recurrent_activity = np.sum(data_dict["recurrent_activity"], axis=0)  # sum the spikes per neuron/channel over time
# find max and min active neurons
neurons_of_interest = [-1, -2, -3, 0]  # most active, second most active, third most active, least active
idc_recurrent_activity_ordered = np.argsort(recurrent_activity[0, :])
recurrent_activity_ordered = data_dict["recurrent_activity"][:, 0, idc_recurrent_activity_ordered[neurons_of_interest]]  # most active neurons first
traces_ordered = data_dict["trace_rec"][0, idc_recurrent_activity_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(traces_ordered.transpose(), label=idc_recurrent_activity_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, traces_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(idc_recurrent_activity_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array([mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, traces_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(idc_recurrent_activity_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, traces_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/recurrent_analysis_first_epoch.pdf", bbox_inches="tight")
plt.close(fig)
pass

# HERE WE ADD TO THE MAIN FIGURE
# First plot (ax1)
ax1 = fig_final.add_subplot(gs_final[:3, 0])
ax1.plot(traces_ordered.transpose(), label=idc_recurrent_activity_ordered[neurons_of_interest])
ax1.set_title("First epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, traces_ordered.shape[1])
handles, labels = ax1.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
ax1.legend(sorted_handles, sorted_labels)
ax1.set_xticklabels([])  # Hide x-axis text

# Second plot (ax2)
ax2 = fig_final.add_subplot(gs_final[3, 0])
ax2.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlim(0, traces_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(idc_recurrent_activity_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, traces_ordered.shape[0] - 1.0 + 0.2)
ax2.set_xticklabels([])  # Hide x-axis text



# last epoch
file_path = "./results/traces/last_epoch"
files = os.listdir(file_path)
data_dict = {}
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
    data_dict[file.split(".")[0]] = data

# input is the time binned data from the two channels per sensor
# trace_in is the eligibility trace linked to the input neurons (two per channel) and is used to update the weights connecting the input neurons to the hidden layer
# trace_rec is the eligibility trace linked to the recurrent neurons and is used to update the weights connecting the recurrent neurons to the output layer
# recurrent_activity is the activity of the recurrent neurons (hidden layer)

# let's find the most and least active neurons in the input layer adn compare there traces
# input_activity = np.sum(data_dict["input"][:, 0, :], axis=0)  # sum the spikes per neuron/channel over time
# # find max and min active neurons
# neurons_of_interest = [-1, -2, -3, 0]  # most active, second most active, third most active, least active
# idc_input_activity_ordered = np.argsort(input_activity)
input_activity_ordered = data_dict["input"][:, 0, idc_input_activity_ordered[neurons_of_interest]]  # most active neurons first
traces_ordered = data_dict["trace_in"][0, idc_input_activity_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(traces_ordered.transpose(), label=idc_input_activity_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, traces_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(input_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(idc_input_activity_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array([mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax2.set_ylabel("Neuron")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, traces_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(idc_input_activity_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, traces_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/input_analysis_last_epoch.pdf", bbox_inches="tight")
plt.close(fig)


# let's find the most and least active neurons in the input layer adn compare there traces
# recurrent_activity = np.sum(data_dict["recurrent_activity"], axis=0)  # sum the spikes per neuron/channel over time
# # find max and min active neurons
# neurons_of_interest = [-1, -2, -3, 0]  # most active, second most active, third most active, least active
# idc_recurrent_activity_ordered = np.argsort(recurrent_activity[0, :])
recurrent_activity_ordered = data_dict["recurrent_activity"][:, 0, idc_recurrent_activity_ordered[neurons_of_interest]]  # most active neurons first
traces_ordered = data_dict["trace_rec"][0, idc_recurrent_activity_ordered[neurons_of_interest], :]  # most active neurons first

# let's plot the activity of the neurons of interest using gridspec with the first two rows for the traces and the third row for the recurrent activity rasterplot
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:2, 0])
ax1.plot(traces_ordered.transpose(), label=idc_recurrent_activity_ordered[neurons_of_interest])
ax1.set_title("Last epoch")
ax1.set_ylabel("Traces")
ax1.set_xlim(0, traces_ordered.shape[1])
# Retrieve the handles and labels from the legend
handles, labels = ax1.get_legend_handles_labels()
# Convert labels to integers (if they are not already) and sort them
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
# Reassign the legend with sorted labels and corresponding handles
ax1.legend(sorted_handles, sorted_labels)
ax2 = fig.add_subplot(gs[2, 0])
time_indices, neuron_indices = np.where(recurrent_activity_ordered == 1)  # Find where spikes occur
new_idc = np.argsort(idc_recurrent_activity_ordered[neurons_of_interest])
# Create a mapping from the original neuron indices to the new indices
mapping = {old: new for old, new in enumerate(new_idc)}
# Remap neuron_indices using the mapping
remapped_neuron_indices = np.array([mapping[neuron] for neuron in neuron_indices])
# Combine time_indices and remapped_neuron_indices for sorting
combined = np.array(list(zip(time_indices, remapped_neuron_indices)))
# Sort by remapped_neuron_indices (second column) and then by time_indices (first column)
sorted_combined = combined[np.lexsort((combined[:, 0], combined[:, 1]))]
# Extract the sorted time_indices and neuron_indices
sorted_time_indices = sorted_combined[:, 0]
sorted_neuron_indices = sorted_combined[:, 1]
# Use the sorted indices in the scatter plot
ax2.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax2.set_ylabel("Recurrent neurons")
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, traces_ordered.shape[1])
ax2.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax2.set_yticklabels(idc_recurrent_activity_ordered[neurons_of_interest][sorted_indices])
ax2.set_ylim(-0.2, traces_ordered.shape[0]-1.0+0.2)
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/recurrent_analysis_last_epoch.pdf", bbox_inches="tight")
plt.close(fig)
pass

# HERE WE ADD TO THE MAIN FIGURE
# Third plot (ax3)
ax3 = fig_final.add_subplot(gs_final[4:7, 0])
ax3.plot(traces_ordered.transpose(), label=idc_recurrent_activity_ordered[neurons_of_interest])
ax3.set_title("First epoch")
ax3.set_ylabel("Traces")
ax3.set_xlim(0, traces_ordered.shape[1])
handles, labels = ax3.get_legend_handles_labels()
sorted_indices = np.argsort([int(label) for label in labels])
sorted_handles = [handles[i] for i in sorted_indices]
sorted_labels = [labels[i] for i in sorted_indices]
ax3.legend(sorted_handles, sorted_labels)
ax3.set_xticklabels([])  # Hide x-axis text

# Second plot (ax4)
ax4 = fig_final.add_subplot(gs_final[7, 0])
ax4.scatter(sorted_time_indices, sorted_neuron_indices, color="black", s=10, label="Spikes")
ax4.set_ylabel("Neuron")
ax4.set_xlim(0, traces_ordered.shape[1])
ax4.set_yticks(np.arange(0, len(neurons_of_interest), 1))
ax4.set_yticklabels(idc_recurrent_activity_ordered[neurons_of_interest][sorted_indices])
ax4.set_ylim(-0.2, traces_ordered.shape[0] - 1.0 + 0.2)
ax4.set_xticklabels([])  # Hide x-axis text


####################
# WEIGHTS ANALYSIS #
####################
# EPOCH 0
file_path = "./results/weights/data_plot_3batch_size_epoch0"
files = os.listdir(file_path)
data_dict_epochs = {}
data_dict_epochs["epoch_0"] = {}
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
        
    data_dict_epochs["epoch_0"][file.split(".")[0]] = np.array(data)

# EPOCH 0
file_path = "./results/weights/data_plot_3batch_size_epoch21"
files = os.listdir(file_path)
data_dict_epochs["epoch_21"] = {}
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
        
    data_dict_epochs["epoch_21"][file.split(".")[0]] = np.array(data)

sum_change_in = np.abs(np.sum(data_dict_epochs["epoch_0"]["update_value_w_in"], axis=0))
idc_max_change_in = np.where(sum_change_in == np.max(sum_change_in))
sum_change_rec = np.abs(np.sum(data_dict_epochs["epoch_0"]["update_value_w_rec"], axis=0))
idc_max_change_rec = np.where(sum_change_rec == np.max(sum_change_rec))
sum_change_out = np.abs(np.sum(data_dict_epochs["epoch_0"]["update_value_w_out"], axis=0))
idc_max_change_out = np.where(sum_change_out == np.max(sum_change_out))

# let's plot the weight change over batches for the layers
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(2, 1, 1)
w_in = data_dict_epochs["epoch_0"]["w_in"][:, idc_max_change_in[0], idc_max_change_in[1]][:, 0]
w_rec = data_dict_epochs["epoch_0"]["w_rec"][:, idc_max_change_rec[0], idc_max_change_rec[1]][:, 0]
w_out = data_dict_epochs["epoch_0"]["w_out"][:, idc_max_change_out[0], idc_max_change_out[1]][:, 0]
ax.plot(w_in, label="w_in")
ax.plot(w_rec, label="w_rec")
ax.plot(w_out, label="w_out")
ax.set_title("First epoch")
ax.set_ylabel("Weight")
ax.set_xlim(0, w_in.shape[0])
ax.legend(loc="upper right")

# HERE WE ADD TO THE MAIN FIGURE
ax5 = fig_final.add_subplot(gs_final[:2, 1])
ax5.plot(w_in, label="w_in")
ax5.plot(w_rec, label="w_rec")
ax5.plot(w_out, label="w_out")
ax5.set_title("First epoch")
ax5.set_ylabel("Weight")
ax5.set_xlim(0, w_in.shape[0])
ax5.legend(loc="upper right")

ax = fig.add_subplot(2, 1, 2)
w_in = data_dict_epochs["epoch_21"]["w_in"][:, idc_max_change_in[0], idc_max_change_in[1]][:, 0]
w_rec = data_dict_epochs["epoch_21"]["w_rec"][:, idc_max_change_rec[0], idc_max_change_rec[1]][:, 0]
w_out = data_dict_epochs["epoch_21"]["w_out"][:, idc_max_change_out[0], idc_max_change_out[1]][:, 0]
ax.plot(w_in, label="w_in")
ax.plot(w_rec, label="w_rec")
ax.plot(w_out, label="w_out")
ax.set_title("Last epoch")
ax.set_ylabel("Weight")
ax.set_xlabel("Batch")
ax.set_xlim(0, w_in.shape[0])
ax.legend(loc="upper right")
fig.align_ylabels()
fig.tight_layout()
fig.savefig("./figures/weights_change_over_batches.pdf", bbox_inches="tight")
plt.close(fig)

# weights over epochs

# HERE WE ADD TO THE MAIN FIGURE
ax6 = fig_final.add_subplot(gs_final[2:4, 1])

# ACCURACY
file_path = "./results/accuracy"
# here we must read a txt file
fila_name = "test_eprop_lr_00008_tautrace_008_tautraceout_0105_threedecimal.txt"


# FINALIZE THE FINAL FIGURE
fig_final.align_ylabels()
fig_final.tight_layout()
fig_final.savefig("./figures/final_figure.pdf", bbox_inches="tight")
plt.close(fig_final)