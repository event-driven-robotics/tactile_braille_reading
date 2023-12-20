'''
Here we have everything to prepare and run the training.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.neuron_models import (feedforward_layer, recurrent_layer,
                                 trainable_time_constants)


def check_cuda():
    if torch.cuda.device_count() > 1:
        torch.cuda.empty_cache()

        gpu_sel = 1
        gpu_av = [torch.cuda.is_available()
                  for ii in range(torch.cuda.device_count())]
        # LOG.debug("Detected {} GPUs. The load will be shared.".format(
        # torch.cuda.device_count()))
        for gpu in range(len(gpu_av)):
            if True in gpu_av:
                if gpu_av[gpu_sel]:
                    device = torch.device("cuda:"+str(gpu))
                    torch.cuda.set_per_process_memory_fraction(
                        0.5, device=device)
                    # LOG.debug("Selected GPUs: {}" .format("cuda:"+str(gpu)))
                else:
                    device = torch.device("cuda:"+str(gpu_av.index(True)))
            else:
                device = torch.device("cpu")
                # LOG.debug("No GPU detected. Running on CPU.")
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # LOG.debug("Single GPU detected. Setting up the simulation there.")
            device = torch.device("cuda:0")
            # torch.cuda.set_per_process_memory_fraction(0.5, device=device)
        else:
            device = torch.device("cpu")
            # LOG.warning("No GPU detected. Running on CPU.")
    return device


def build(params, nb_channels, nb_hidden, nb_outputs, time_step, ste_fn=None, use_trainable_tc=False, use_trainable_out=False, bit_resolution=None, dynamic_clamping=False, device='cpu', dtype=torch.float, logger=None):
    '''
    Here we set up the network.
    '''
    # init the dict for layers and time constants
    layers = []
    time_constants = []

    # Network parameters
    nb_inputs = nb_channels*params['nb_input_copies']

    tau_mem = params['tau_mem']  # ms
    tau_syn = tau_mem/params['tau_ratio']

    alpha = float(np.exp(-time_step/tau_syn))
    beta = float(np.exp(-time_step/tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = fwd_weight_scale*params['weight_scale_factor']

    # report some numbers back
    if logger is not None:
        logger.debug("Network parameters:")
        logger.debug("Number of inputs: {}" .format(nb_inputs))
        logger.debug("Number of hidden neurons: {}" .format(nb_hidden))
        logger.debug("Number of output neurons: {}" .format(nb_outputs))
        logger.debug("Time step: {}" .format(time_step))
        logger.debug("Forward weight scale: {}" .format(fwd_weight_scale))
        logger.debug("Recurrent weight scale: {}" .format(rec_weight_scale))
        logger.debug("Alpha: {}" .format(alpha))
        logger.debug("Beta: {}" .format(beta))
        if bit_resolution == "baseline":
            logger.debug("No weight discretization applied.\n")

    # recurrent layer
    w1, v1 = recurrent_layer.create_layer(
        nb_inputs, nb_hidden, fwd_weight_scale, rec_weight_scale, device=device, dtype=dtype)

    # readout layer
    w2 = feedforward_layer.create_layer(
        nb_hidden, nb_outputs, fwd_weight_scale, device=device, dtype=dtype)

    # write layers to dict
    layers.append(w1), layers.append(w2), layers.append(v1)

    if bit_resolution != "baseline":
        if dynamic_clamping:
            clamp_max, clamp_min = np.max([torch.max(w1).detach().cpu().numpy(), torch.max(w2).detach().cpu().numpy(), torch.max(v1).detach().cpu(
            ).numpy()]), np.min([torch.min(w1).detach().cpu().numpy(), torch.min(w2).detach().cpu().numpy(), torch.min(v1).detach().cpu().numpy()])
            clamp_max, clamp_min = 1.2*clamp_max, 1.2*clamp_min  # add some margin
            # calculate possible weight values
            # determines in how many increments we seperate values between min and max (inlc. both)
            number_of_increments = 2**bit_resolution
            possible_weight_values = np.linspace(
                clamp_min, clamp_max, number_of_increments)
            possible_weight_values = torch.as_tensor(
                possible_weight_values, device=device, dtype=dtype)
        else:
            # calculate possible weight values
            # determines in how many increments we seperate values between min and max (inlc. both)
            number_of_increments = 2**bit_resolution
            possible_weight_values = np.linspace(
                -0.5, 0.5, number_of_increments)
            possible_weight_values = torch.as_tensor(
                possible_weight_values, device=device, dtype=dtype)

    if bit_resolution != "baseline":
        if logger is not None:
            if dynamic_clamping:
                logger.debug("Dynamic clamping parameters:")
            else:
                logger.debug("Fix clamping parameters:")
            logger.debug("Lower bound of weight values: {}" .format(
                torch.min(possible_weight_values)))
            logger.debug("Upper bound of weight values: {}" .format(
                torch.max(possible_weight_values)))
            logger.debug("Number of increments of weight values: {}\n" .format(
                len(possible_weight_values)))
        # use thee STE function for quantization
        for layer_idx in range(len(layers)):
            try:
                # fast but memory intensive
                logger.debug(
                    f"Using fast but memory intensive STE function on layer {layer_idx+1}.")
                layers[layer_idx].data.copy_(
                    ste_fn(layers[layer_idx].data, possible_weight_values))
            except:
                logger.debug(
                    f"Fast STE function failed. Using slower version on layer {layer_idx+1}.")
                # slower but memory efficient
                for neuron_idx in range(len(layers[layer_idx])):
                    layers[layer_idx][neuron_idx].data.copy_(
                        ste_fn(layers[layer_idx][neuron_idx].data, possible_weight_values))

    if use_trainable_out:
        # include trainable output for readout layer (linear: y = out_scale * x + out_offset)
        out_scale = torch.empty(
            (nb_outputs),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.ones_(out_scale)
        layers.append(out_scale)
        out_offset = torch.empty(
            (nb_outputs),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.zeros_(out_offset)
        layers.append(out_offset)

    # create and write time constants to dict
    if use_trainable_tc:
        alpha1, beta1 = trainable_time_constants.create_time_constants(
            nb_hidden, alpha, beta, device=device, dtype=dtype)
        alpha1 = torch.clamp(alpha1, min=1E-12, max=None)
        beta1 = torch.clamp(beta1, min=1E-12, max=None)
        time_constants.append(alpha1), time_constants.append(beta1)

        alpha2, beta2 = trainable_time_constants.create_time_constants(
            nb_outputs, alpha, beta, device=device, dtype=dtype)
        alpha2 = torch.clamp(alpha1, min=1E-12, max=None)
        beta2 = torch.clamp(beta2, min=1E-12, max=None)
        time_constants.append(alpha2), time_constants.append(beta2)

    else:
        time_constants.append(alpha), time_constants.append(beta)

    return layers, time_constants


def train(params, spike_fn, dataset_train, ste_fn=None, batch_size=128, lr=0.0015, nb_epochs=300, layers=None, time_constants=None, dataset_validation=None, bit_resolution=None, dynamic_clamping=False, logger=None, use_trainable_out=False, use_trainable_tc=False, use_dropout=False, device='cpu', dtype=torch.float):
    """
    Here we have the training framework.
    """

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

    generator = DataLoader(dataset_train, batch_size=batch_size,
                           shuffle=True, num_workers=2)

    # The optimization loop
    loss_hist = []
    accs_hist = [[], []]
    for e in range(nb_epochs):
        # learning rate decreases over epochs
        optimizer = torch.optim.Adamax(layers, lr=lr, betas=(0.9, 0.995))
        # if e > nb_epochs/2:
        #     lr = lr * 0.9
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(device), y_local.to(device)

            spks_out, recs = _run_snn(inputs=x_local, layers=layers, time_constants=time_constants, spike_fn=spike_fn, nb_input_copies=params[
                'nb_input_copies'], device=device, dtype=dtype, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)
            # [mem_rec, spk_rec, out_rec]
            _, spk_rec, _ = recs

            # with output spikes
            if use_trainable_out:
                m = spks_out
            else:
                m = torch.sum(spks_out, 1)  # sum over time
            # cross entropy loss on the active read-out layer
            log_p_y = log_softmax_fn(m)

            # TODO change to loop!
            # Here we can set up our regularizer loss
            # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
            # L1 loss on total number of spikes (hidden layer 1)
            reg_loss = params['reg_spikes']*torch.mean(torch.sum(spk_rec, 1))
            # L1 loss on total number of spikes (output layer)
            # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spks_out, 1))
            # "L1: ", reg_loss)
            # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
            # L2 loss on spikes per neuron (hidden layer 1)
            reg_loss += params['reg_neurons'] * \
                torch.mean(torch.sum(torch.sum(spk_rec, dim=0), dim=0)**2)
            # L2 loss on spikes per neuron (output layer)
            # reg_loss += params['reg_neurons'] * \
            #     torch.mean(torch.sum(torch.sum(spks_out, dim=0), dim=0)**2)
            # "L1 + L2: ", reg_loss)

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

            if bit_resolution != "baseline":
                if dynamic_clamping:
                    # get the max and min value from the actual weights and include a safety margin
                    clamp_max, clamp_min = np.max([torch.max(layers[0]).detach().cpu().numpy(), torch.max(layers[1]).detach().cpu().numpy(), torch.max(layers[2]).detach().cpu(
                    ).numpy()]), np.min([torch.min(layers[0]).detach().cpu().numpy(), torch.min(layers[1]).detach().cpu().numpy(), torch.min(layers[2]).detach().cpu().numpy()])
                    clamp_max, clamp_min = 1.2*clamp_max, 1.2*clamp_min
                    # calculate possible weight values
                    # determines in how many increments we seperate values between min and max (inlc. both)
                    number_of_increments = 2**bit_resolution
                    possible_weight_values = np.linspace(
                        clamp_min, clamp_max, number_of_increments)
                    possible_weight_values = torch.as_tensor(
                        possible_weight_values, device=device, dtype=dtype)
                else:
                    # calculate possible weight values
                    # determines in how many increments we seperate values between min and max (inlc. both)
                    number_of_increments = 2**bit_resolution
                    possible_weight_values = np.linspace(
                        -0.5, 0.5, number_of_increments)
                    possible_weight_values = torch.as_tensor(
                        possible_weight_values, device=device, dtype=dtype)

                # use the STE function for quantization
                for layer_idx in range(len(layers)):
                    try:
                        # fast but memory intensive
                        # logger.debug(
                        #     f"Using fast but memory intensive STE function on layer {layer_idx+1}.")
                        layers[layer_idx].data.copy_(
                            ste_fn(layers[layer_idx].data, possible_weight_values))
                    except:
                        # logger.debug(
                        #     f"Fast STE function failed. Using slower version on layer {layer_idx+1}.")
                        # slower but memory efficient
                        for neuron_idx in range(len(layers[layer_idx])):
                            layers[layer_idx][neuron_idx].data.copy_(
                                ste_fn(layers[layer_idx][neuron_idx].data, possible_weight_values))

            # compare to labels
            _, spikes_max = torch.max(m, 1)  # argmax over output units
            acc = np.mean((y_local == spikes_max).detach().cpu().numpy())
            accs.append(acc)

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate validation accuracy in each epoch
        validation_acc, _, _, _ = validate_model(dataset=dataset_validation, layers=layers, time_constants=time_constants, batch_size=batch_size, spike_fn=spike_fn, nb_input_copies=params[
            'nb_input_copies'], device=device, dtype=torch.float, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)
        # only safe best validation
        accs_hist[1].append(np.mean(validation_acc))

        # save best test
        if np.max(validation_acc) >= np.max(accs_hist[1]):
            best_acc_layers = []
            for ii in layers:
                best_acc_layers.append(ii.detach().clone())

        logger.debug("Epoch {}/{} done. Train accuracy: {:.2f}%, Test accuracy: {:.2f}%, Loss: {:.5f}.".format(
            e + 1, nb_epochs, accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))

    return loss_hist, accs_hist, best_acc_layers


def validate_model(dataset, layers, time_constants, spike_fn, nb_input_copies=1, batch_size=128, device='cpu', dtype=torch.float, use_trainable_out=False, use_trainable_tc=False, use_dropout=False):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    accs = []
    trues = []
    preds = []
    activity_record = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        spks_out, recs = _run_snn(inputs=x_local, layers=layers, time_constants=time_constants, spike_fn=spike_fn, nb_input_copies=nb_input_copies,
                                  device=device, dtype=dtype, use_trainable_out=use_trainable_out, use_trainable_tc=use_trainable_tc, use_dropout=use_dropout)
        # [mem_rec, spk_rec, out_rec]
        _, spk_rec, _ = recs
        # with output spikes
        if use_trainable_out:
            sum_over_time = spks_out
        else:
            sum_over_time = torch.sum(spks_out, 1)  # sum over time
        _, out_max = torch.max(sum_over_time, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == out_max).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(out_max.detach().cpu().numpy())
        activity_record.append(
            [spk_rec.detach().cpu().numpy(), spks_out.detach().cpu().numpy()])

    return accs, trues, preds, activity_record


def _run_snn(inputs, layers, time_constants, spike_fn, nb_input_copies=1, device='cpu', dtype=torch.float, use_trainable_out=False, use_trainable_tc=False, use_dropout=False):
    """
    Here we set up the layers and run the SNN.
    """

    if use_trainable_out:
        w1, w2, v1, out_scale, out_offset = layers
    else:
        w1, w2, v1 = layers
    if use_trainable_tc:
        alpha1, beta1, alpha2, beta2 = time_constants
    else:
        alpha, beta = time_constants

    if use_dropout:
        dropout = nn.Dropout(p=0.25)  # using dropout on n % of time steps

    bs, nb_hidden, nb_outputs, nb_steps = inputs.shape[0], w2.shape[0], w2.shape[1], len(
        inputs[0, :])

    h1 = torch.einsum(
        "abc,cd->abd", (inputs.tile((nb_input_copies,)), w1))
    if use_dropout:
        h1 = dropout(h1)

    if use_trainable_tc:
        spk_rec, mem_rec = recurrent_layer.compute_activity(spike_fn=spike_fn, nb_input=bs, nb_neurons=nb_hidden,
                                                            input_activity=h1, layer=v1, alpha=alpha1, beta=beta1, nb_steps=nb_steps, device=device, dtype=dtype)
    else:
        spk_rec, mem_rec = recurrent_layer.compute_activity(spike_fn=spike_fn, nb_input=bs, nb_neurons=nb_hidden,
                                                            input_activity=h1, layer=v1, alpha=alpha, beta=beta, nb_steps=nb_steps, device=device, dtype=dtype)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    if use_dropout:
        h2 = dropout(h2)
    if use_trainable_tc:
        s_out_rec, out_rec = feedforward_layer.compute_activity(
            spike_fn=spike_fn, nb_input=bs, nb_neurons=nb_outputs, input_activity=h2, alpha=alpha2, beta=beta2, nb_steps=nb_steps, device=device, dtype=dtype)
    else:
        s_out_rec, out_rec = feedforward_layer.compute_activity(
            spike_fn=spike_fn, nb_input=bs, nb_neurons=nb_outputs, input_activity=h2, alpha=alpha, beta=beta, nb_steps=nb_steps, device=device, dtype=dtype)

    if use_trainable_out:
        # trainable output spike scaling
        # mean_firing_rate = torch.div(torch.sum(s_out_rec,1), s_out_rec.shape[1]) # mean firing rate
        s_out_rec = torch.sum(s_out_rec, 1)*out_scale + \
            out_offset  # sum spikes

    other_recs = [mem_rec, spk_rec, out_rec]

    return s_out_rec, other_recs
