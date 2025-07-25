import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.neuron_models import LIF, RLIF, CuBaLIF, CuBaRLIF
from time import perf_counter


def main():
    start_time = perf_counter()
    # Define parameters
    nb_inputs = 10  # Number of input neurons
    nb_ff_neurons = 20  # Number of output neurons
    nb_rec_neurons = 20  # Number of recurrent neurons
    fwd_scale = 0.1  # Scaling factor for weight initialization
    rec_scale = 0.2  # Scaling factor for weight initialization
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # Use GPU if available
    nb_steps = 1000  # Number of time steps
    alpha = 0.94  # Synaptic decay constant
    beta = 0.8  # Membrane decay constant

    # Create input activity tensor (batch_size, nb_steps, nb_inputs)
    batch_size = 1
    input_activity = torch.rand(
        (batch_size, nb_steps, nb_inputs), device=device, dtype=torch.float)
    input_activity[input_activity > 0.8] = 1.0  # Binarize input activity
    input_activity[input_activity != 1.0] = 0.0  # Binarize input activity

    finish_init = perf_counter()
    print("Initialization time: ", finish_init - start_time)

    # Instantiate the feedforward layer
    feedforward_cuba_layer = CuBaLIF(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_ff_neurons,
                                     fwd_scale=fwd_scale, alpha=alpha, beta=beta, requires_grad=False)
    recurrent_cuba_layer = CuBaRLIF(batch_size=batch_size, nb_inputs=nb_ff_neurons, nb_neurons=nb_rec_neurons,
                                    fwd_scale=fwd_scale, rec_scale=rec_scale, alpha=alpha, beta=beta, requires_grad=False)
    finish_class_init = perf_counter()
    print("Class initialization time: ", finish_class_init - finish_init)

    # initialize the recording
    cuba_ff_spk_rec = np.zeros((nb_steps, batch_size, nb_ff_neurons))
    cuba_ff_syn_rec = np.zeros((nb_steps, batch_size, nb_ff_neurons))
    cuba_ff_mem_rec = np.zeros((nb_steps, batch_size, nb_ff_neurons))

    cuba_rec_spk_rec = np.zeros((nb_steps, batch_size, nb_rec_neurons))
    cuba_rec_syn_rec = np.zeros((nb_steps, batch_size, nb_rec_neurons))
    cuba_rec_mem_rec = np.zeros((nb_steps, batch_size, nb_rec_neurons))

    finish_out_init = perf_counter()
    print("Output initialization time: ", finish_out_init - finish_class_init)

    # Loop over time steps
    for t in range(nb_steps):
        # Extract input activity for the current time step
        input_activity_t = input_activity[:, t]
        ff_spk, ff_syn, ff_rec = feedforward_cuba_layer.update(
            input_activity_t)
        rec_spk, rec_syn, rec_mem = recurrent_cuba_layer.update(ff_spk)
        # Store the results
        cuba_ff_spk_rec[t] = ff_spk.detach().cpu().numpy()
        cuba_ff_syn_rec[t] = ff_syn.detach().cpu().numpy()
        cuba_ff_mem_rec[t] = ff_rec.detach().cpu().numpy()
        cuba_rec_spk_rec[t] = rec_spk.detach().cpu().numpy()
        cuba_rec_syn_rec[t] = rec_syn.detach().cpu().numpy()
        cuba_rec_mem_rec[t] = rec_mem.detach().cpu().numpy()
    pass
    finish_update = perf_counter()
    print("### Update time: ", finish_update - finish_out_init)

    # Instantiate the feedforward layer
    feedforward_layer = LIF(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_ff_neurons,
                            fwd_scale=fwd_scale, beta=beta, requires_grad=False)
    recurrent_layer = RLIF(batch_size=batch_size, nb_inputs=nb_ff_neurons, nb_neurons=nb_rec_neurons,
                           fwd_scale=fwd_scale, rec_scale=rec_scale, beta=beta, requires_grad=False)
    finish_class_init2 = perf_counter()
    print("Class initialization time: ", finish_class_init2 - finish_update)

    # new init for the non-CUBA version
    fwd_scale = 0.5  # Scaling factor for weight initialization
    rec_scale = 0.6  # Scaling factor for weight initialization
    beta = 0.99  # Membrane decay constant

    # initialize the recording
    ff_spk_rec = np.zeros((nb_steps, batch_size, nb_ff_neurons))
    ff_mem_rec = np.zeros((nb_steps, batch_size, nb_ff_neurons))

    rec_spk_rec = np.zeros((nb_steps, batch_size, nb_rec_neurons))
    rec_mem_rec = np.zeros((nb_steps, batch_size, nb_rec_neurons))

    finish_out_init2 = perf_counter()
    print("Output initialization time: ",
          finish_out_init2 - finish_class_init2)

    # Loop over time steps
    for t in range(nb_steps):
        # Extract input activity for the current time step
        input_activity_t = input_activity[:, t]
        ff_spk, ff_rec = feedforward_layer.update(input_activity_t)
        rec_spk, rec_mem = recurrent_layer.update(ff_spk)
        # Store the results
        ff_spk_rec[t] = ff_spk.detach().cpu().numpy()
        ff_mem_rec[t] = ff_rec.detach().cpu().numpy()
        rec_spk_rec[t] = rec_spk.detach().cpu().numpy()
        rec_mem_rec[t] = rec_mem.detach().cpu().numpy()
    pass
    finish_update2 = perf_counter()
    print("### Update time: ", finish_update2 - finish_out_init2)

    input_activity_np = input_activity.detach().cpu().numpy()
    # let's visualize the two results
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    # ax1.set_title("Neuron model comparison")
    # ax1.set_xlabel("Time")
    ax1.set_ylabel("Input")
    event_times = [np.where(input_activity_np[0, :, neuron] == 1)[
        0] for neuron in range(nb_inputs)]
    ax1.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax1.set_ylim(0, nb_inputs)
    ax1.set_xlim(0, nb_steps)
    ax1.set_xticks(np.arange(0, nb_steps, 100))
    ax1.set_yticks(np.arange(0, nb_inputs, 5))
    ax1.set_xticklabels(np.arange(0, nb_steps, 100))
    ax1.set_yticklabels(np.arange(0, nb_inputs, 5))
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    # ax2.set_title("CUBA LIF")
    # ax2.set_xlabel("Time")
    ax2.set_ylabel("CUBA LIF")
    event_times = [np.where(cuba_ff_spk_rec[:, 0, neuron] == 1)[0]
                   for neuron in range(nb_ff_neurons)]
    ax2.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax2.set_ylim(0, nb_ff_neurons)
    ax2.set_xlim(0, nb_steps)
    ax2.set_xticks(np.arange(0, nb_steps, 100))
    ax2.set_yticks(np.arange(0, nb_ff_neurons, 5))
    ax2.set_xticklabels(np.arange(0, nb_steps, 100))
    ax2.set_yticklabels(np.arange(0, nb_ff_neurons, 5))
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    # ax3.set_title("LIF")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("CUBA RLIF")
    event_times = [np.where(cuba_rec_spk_rec[:, 0, neuron] == 1)[0]
                   for neuron in range(nb_rec_neurons)]
    ax3.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax3.set_ylim(0, nb_rec_neurons)
    ax3.set_xlim(0, nb_steps)
    ax3.set_xticks(np.arange(0, nb_steps, 100))
    ax3.set_yticks(np.arange(0, nb_rec_neurons, 5))
    ax3.set_xticklabels(np.arange(0, nb_steps, 100))
    ax3.set_yticklabels(np.arange(0, nb_rec_neurons, 5))
    ax3.grid()
    plt.tight_layout()
    plt.savefig("cuba_lif.pdf")
    plt.close(fig)
    pass

    # input_activity = input_activity.detach().cpu().numpy()
    # let's visualize the two results
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    # ax1.set_title("Neuron model comparison")
    # ax1.set_xlabel("Time")
    ax1.set_ylabel("Input")
    event_times = [np.where(input_activity_np[0, :, neuron] == 1)[
        0] for neuron in range(nb_inputs)]
    ax1.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax1.set_ylim(0, nb_inputs)
    ax1.set_xlim(0, nb_steps)
    ax1.set_xticks(np.arange(0, nb_steps, 100))
    ax1.set_yticks(np.arange(0, nb_inputs, 5))
    ax1.set_xticklabels(np.arange(0, nb_steps, 100))
    ax1.set_yticklabels(np.arange(0, nb_inputs, 5))
    ax1.grid()

    ax2 = fig.add_subplot(3, 1, 2)
    # ax2.set_title("CUBA LIF")
    # ax2.set_xlabel("Time")
    ax2.set_ylabel("LIF")
    event_times = [np.where(ff_spk_rec[:, 0, neuron] == 1)[0]
                   for neuron in range(nb_ff_neurons)]
    ax2.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax2.set_ylim(0, nb_ff_neurons)
    ax2.set_xlim(0, nb_steps)
    ax2.set_xticks(np.arange(0, nb_steps, 100))
    ax2.set_yticks(np.arange(0, nb_ff_neurons, 5))
    ax2.set_xticklabels(np.arange(0, nb_steps, 100))
    ax2.set_yticklabels(np.arange(0, nb_ff_neurons, 5))
    ax2.grid()

    ax3 = fig.add_subplot(3, 1, 3)
    # ax3.set_title("LIF")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("RLIF")
    event_times = [np.where(rec_spk_rec[:, 0, neuron] == 1)[0]
                   for neuron in range(nb_rec_neurons)]
    ax3.eventplot(event_times, orientation='horizontal',
                  colors='black', linelengths=0.8)
    ax3.set_ylim(0, nb_rec_neurons)
    ax3.set_xlim(0, nb_steps)
    ax3.set_xticks(np.arange(0, nb_steps, 100))
    ax3.set_yticks(np.arange(0, nb_rec_neurons, 5))
    ax3.set_xticklabels(np.arange(0, nb_steps, 100))
    ax3.set_yticklabels(np.arange(0, nb_rec_neurons, 5))
    ax3.grid()
    plt.tight_layout()
    plt.savefig("lif.pdf")
    plt.close(fig)
    pass


if __name__ == "__main__":
    main()
