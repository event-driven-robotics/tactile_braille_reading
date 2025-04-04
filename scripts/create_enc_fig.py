import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PLOT_ALL_LETTERS = False
enc_data_file = "./data/100Hz/data_braille_letters_100Hz_th2.pkl"
with open(enc_data_file, "rb") as f:
    enc_data = pickle.load(f)
pass

frequ = 100  # Hz
# samples = raw_data["taxel_data"].values
# raw_timestamp = raw_data["timestamp"].values
# raw_letter = raw_data["letter"].values

# given, that event data was created from three different recordings, we need to take only one of them

events = []
samples = []
letters = []
for item in enc_data:
    events.append(item["events"])
    letters.append(item["letter"])
    samples.append(item["samples"])
timestamps = np.arange(start=0, stop=(350 - 0.5)
                     * (1/frequ), step=1/frequ)  # create a timestamp array for the given frequency

if PLOT_ALL_LETTERS:
    # find the first occurrence of each unique letter in raw_letter
    unique_letters, first_indices = np.unique(letters, return_index=True)
    idc = sorted(first_indices)  # Sort indices to maintain the order of appearance
    # plot the enc data
    for idx in tqdm(idc):
        # create a figure with 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # plot the raw data
        axs[0].plot(timestamps, samples[idx], label=[str(i) for i in range(1, 13)])
        axs[0].set_title(f"Letter: {letters[idx]}")
        axs[0].set_ylabel("Taxel Value")
        axs[0].legend()
        # Enable and style the major grid
        axs[0].grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
        # Enable and style the minor grid
        axs[0].grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
        axs[0].minorticks_on()  # Turn on minor ticks
        axs[0].set_xlim([timestamps[0], timestamps[-1]])
        # event plot
        for i, sensor_events in enumerate(events[idx]):
            # ON events
            if len(sensor_events[0]):
                axs[1].eventplot(
                    sensor_events[0],
                    lineoffsets=i+1,
                    linelengths=0.8,
                    colors="green",
                )
            # OFF events
            if len(sensor_events[1]):
                axs[1].eventplot(
                    sensor_events[1],
                    lineoffsets=i+1,
                    linelengths=0.8,
                    colors="red",
                )
        # Enable and style the major grid
        axs[1].grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
        # Enable and style the minor grid
        axs[1].grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
        axs[1].minorticks_on()  # Turn on minor ticks
        axs[1].set_xlim([timestamps[0], timestamps[-1]])
        fig.savefig(
            f"./figures/enc_{letters[idx]}_th_2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        pass

# prepare a close up from a specific letter
letter = "A"
selected_taxel = 1
time_span = [1.9, 2.2]  # in seconds
idx = np.where(letter == np.array(letters))[0][0]
# only use data within the timespan of interest
mask1 = timestamps >= time_span[0]
local_timestamps = timestamps[mask1]
local_raw_data = samples[idx][mask1]
mask2 = local_timestamps <= time_span[1]
local_timestamps = local_timestamps[mask2]
local_raw_data = local_raw_data[mask2]
# create a mask for the enc data
mask1 = np.array(events[idx][selected_taxel][0]) >= time_span[0]
on_events = np.array(events[idx][selected_taxel][0])[mask1]
mask2 = on_events <= time_span[1]
on_events = on_events[mask2]
mask1 = np.array(events[idx][selected_taxel][1]) >= time_span[0]
off_events = np.array(events[idx][selected_taxel][1])[mask1]
mask2 = off_events <= time_span[1]
off_events = off_events[mask2]

# OPTION I
# create a figure with 2 subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# plot the raw data
axs[0].plot(local_timestamps, local_raw_data[:, selected_taxel]-local_raw_data[0, selected_taxel], '-o', label=f"Taxel {selected_taxel+1}", color="black")
axs[0].set_title(f"Letter: {letters[idx]}", fontsize=12)
axs[0].set_ylabel("Taxel Value", fontsize=12)
axs[0].legend()
# Enable and style the major grid
axs[0].grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
# Enable and style the minor grid
axs[0].grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
axs[0].minorticks_on()  # Turn on minor ticks
axs[0].set_xlim([local_timestamps[0], local_timestamps[-1]])
# ON events
axs[1].eventplot(
    on_events,
    lineoffsets=1,
    linelengths=2.0,
    colors="green",
)
# OFF events
axs[1].eventplot(
    off_events,
    lineoffsets=-1,
    linelengths=2.0,
    colors="red",
)
# Enable and style the major grid
axs[1].grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
# Enable and style the minor grid
axs[1].grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
axs[1].minorticks_on()  # Turn on minor ticks
axs[1].set_xlim([local_timestamps[0], local_timestamps[-1]])
axs[1].set_xlabel("Time [s]", fontsize=12)
axs[1].set_ylabel("Events", fontsize=12)
axs[1].set_yticks([-1, 1])
axs[1].set_yticklabels(["OFF", "ON"])
# Turn on minor ticks
fig.savefig(
    f"./figures/enc_{letters[idx]}_th_2_closeup_1.pdf",
    bbox_inches="tight",
)
plt.close(fig)


# OPTION II
from matplotlib.lines import Line2D  # Import Line2D for custom legend entries
mean = (local_raw_data[:-1, selected_taxel]-local_raw_data[0, selected_taxel] + local_raw_data[1:, selected_taxel]-local_raw_data[0, selected_taxel]) / 2
# create a figure with 2 subplots
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
# plot the raw data
ax.plot(local_timestamps, local_raw_data[:, selected_taxel]-local_raw_data[0, selected_taxel], '-o', label=f"Taxel {selected_taxel+1}", color="black")
ax.set_title(f"Letter: {letters[idx]}", fontsize=12)
ax.set_ylabel("Taxel Value", fontsize=12)
# ON events
# center = 0
linelengths = 10
on_event_heights = [mean[np.searchsorted(local_timestamps[:-1], t, side="right") - 1] for t in on_events]
for event, offset in zip(on_events, on_event_heights):
    ax.eventplot(
        positions=[event],
        lineoffsets=offset,
        linelengths=linelengths,
        colors="green",
    )
# OFF events
off_event_heights = [mean[np.searchsorted(local_timestamps[:-1], t, side="right") - 1] for t in off_events]
for event, offset in zip(off_events, off_event_heights):
    ax.eventplot(
        positions=[event],
        lineoffsets=offset,
        linelengths=linelengths,
        colors="red",
    )
# Enable and style the major grid
ax.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
# Enable and style the minor grid
ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
ax.minorticks_on()  # Turn on minor ticks
ax.set_xlim([local_timestamps[0], local_timestamps[-1]])
# Add custom legend entries for ON and OFF events
legend_elements = [
    Line2D([0], [0], color="green", marker="|", linestyle="None", markersize=10, label="ON"),
    Line2D([0], [0], color="red", marker="|", linestyle="None", markersize=10, label="OFF"),
    Line2D([0], [0], color="black", linestyle="-", marker="o", label=f"Taxel {selected_taxel+1}"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
ax.set_xlabel("Time [s]", fontsize=12)
# Turn on minor ticks
fig.savefig(
    f"./figures/enc_{letters[idx]}_th_2_closeup_2.pdf",
    bbox_inches="tight",
)
plt.close(fig)


# OPTION III
offset = -20 # np.mean(local_raw_data[:, selected_taxel]-local_raw_data[0, selected_taxel])
linelengths = 4
# create a figure with 2 subplots
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
# plot the raw data
ax.plot(local_timestamps, local_raw_data[:, selected_taxel]-local_raw_data[0, selected_taxel], '-o', label=f"Taxel {selected_taxel+1}", color="black")
# let's create lines to connect the events with the taxel values, for that we need linear interpolation between to samples to calculate the taxel value at the event time
# let's create the line equations
# y = mx + b
y = local_raw_data[:, selected_taxel]-local_raw_data[0, selected_taxel]
x = local_timestamps
m = np.diff(y) / np.diff(x)
b = y[:-1] - m * x[:-1]
# now we can calculate the taxel value at the event time    
on_event_heights = [m[np.searchsorted(x[:-1], t, side="right") - 1] * t + b[np.searchsorted(x[:-1], t, side="right") - 1] for t in on_events]
off_event_heights = [m[np.searchsorted(x[:-1], t, side="right") - 1] * t + b[np.searchsorted(x[:-1], t, side="right") - 1] for t in off_events]
# now we can plot the lines to connect the events with the taxel values
for event, line_offset in zip(on_events, on_event_heights):
    ax.plot([event, event], [offset, line_offset], color="black", linestyle="--", linewidth=0.5)
for event, line_offset in zip(off_events, off_event_heights):
    ax.plot([event, event], [offset, line_offset], color="black", linestyle="--", linewidth=0.5)
# ON events
# center = 0
ax.eventplot(
    positions=on_events,
    lineoffsets=offset,
    linelengths=linelengths,
    colors="green",
)
# OFF events
ax.eventplot(
    positions=off_events,
    lineoffsets=offset,
    linelengths=linelengths,
    colors="red",
)
ax.set_title(f"Letter: {letters[idx]}")
ax.set_ylabel("Taxel Value")
# Enable and style the major grid
ax.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.4, color="black")  # Thicker and solid for major grid
# Enable and style the minor grid
ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4, color="gray")  # Thinner and dashed for minor grid
ax.minorticks_on()  # Turn on minor ticks
ax.set_xlim([local_timestamps[0], local_timestamps[-1]])
ax.set_ylim([offset-4, 2])
ax.set_xlabel("Time [s]", fontsize=12)
# Add custom legend entries for ON and OFF events
legend_elements = [
    Line2D([0], [0], color="green", marker="|", linestyle="None", markersize=10, label="ON neuron"),
    Line2D([0], [0], color="red", marker="|", linestyle="None", markersize=10, label="OFF neuron"),
    Line2D([0], [0], color="black", linestyle="-", marker="o", label=f"Taxel {selected_taxel+1}"),
]
ax.legend(handles=legend_elements, loc="upper center", fontsize=10)
# Turn on minor ticks
fig.savefig(
    f"./figures/enc_{letters[idx]}_th_2_closeup.pdf",
    bbox_inches="tight",
)
plt.close(fig)