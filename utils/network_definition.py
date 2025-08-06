import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.neuron_models import CuBaLIF_HW_Aware, CuBaRLIF_HW_Aware, CuBaLIF_HW_Aware_OG, CuBaRLIF_HW_Aware_OG, STEFunction

ste_fn = STEFunction.apply()

class SRNN:
    """
    Spiking Recurrent Neural Network (SRNN) model using CUBA LIF neurons.

    This class implements a two-layer spiking network with a recurrent hidden
    layer and a feedforward readout layer.  The network can be trained either
    by Backpropagation Through Time (BPTT) or by Eligibility Propagation (e-prop),
    as selected via a configuration dictionary.

    Attributes:
        nb_inputs (int):    Number of input channels.
        nb_hidden (int):    Number of neurons in the recurrent (hidden) layer.
        nb_output (int):    Number of neurons in the output (readout) layer.
        time_bin_size (int): Time bin size in milliseconds used to discretize inputs.
        eprop (bool):       If True, use e-prop for training; otherwise use BPTT.
        alpha (float):      Synaptic decay constant, exp(-Δt/τ_syn).
        beta (float):       Membrane decay constant, exp(-Δt/τ_mem).
        beta_adaptive_thr (float): Adaptive threshold decay, exp(-Δt/τ_adaptive_thr).
        beta_trace (float): Eligibility trace decay for feedforward weights.
        beta_trace_rec (float): Eligibility trace decay for recurrent weights.
        device (torch.device): Device on which tensors are allocated.
        dtype (torch.dtype):   Data type for network tensors.
        rec_layer (CuBaRLIF):  Instance of the recurrent LIF layer.
        ff_layer (CuBaLIF):    Instance of the feedforward (readout) LIF layer.

    Methods:
        forward(input):
            Run one forward pass over a batch of input spike-trains.
        train_bptt(dataset_train, dataset_test):
            Train the network using Backpropagation Through Time.
        train_eprop(dataset_train, dataset_test, labels):
            Train the network using Eligibility Propagation.
        grads_batch(x, yo, yt, v, z, data_steps, nb_inputs, nb_hidden):
            Compute e-prop gradients for a single batch.
        compute_classification_accuracy(dataset):
            Evaluate classification accuracy on a dataset.
    """

    def __init__(self, nb_inputs, nb_hidden, nb_output, dict_args):
        """
        Initialize the SRNN model with architecture parameters, dynamics constants,
        and instantiate its recurrent and feedforward layers.

        Args:
            nb_inputs (int): Number of input channels.
            nb_hidden (int): Number of neurons in the recurrent (hidden) layer.
            nb_output (int): Number of neurons in the output (readout) layer.
            dict_args (dict): Configuration dictionary containing training and model
                parameters. Expected keys include:
                    - time_bin_size (int): Bin size in ms for input discretization.
                    - eprop (bool): If True, use eligibility propagation; else BPTT.
                    - tau_syn (float): Synaptic time constant in ms.
                    - tau_mem (float): Membrane time constant in ms.
                    - tau_adaptive_thr (float): Adaptive threshold time constant in ms.
                    - tau_trace (float): Feedforward eligibility trace decay.
                    - tau_trace_rec (float): Recurrent eligibility trace decay.
                    - device (str): Torch device identifier ('cpu' or 'cuda').
                    - batch_size (int): Batch size for training and evaluation.
                    - epochs (int): Number of training epochs.
                    - lr (float): Learning rate.
                    - reg_spikes (float): Spike count regularization coefficient.
                    - reg_neurons (float): Neuron activity regularization coefficient.
                    - fwd_weight_scale (float): Initialization scale for feedforward weights.
                    - weight_scale_factor (float): Initialization scale for recurrent weights.
                    - ref_per_timesteps (int): Neuron refractory period in time steps.
                    - lower_bound (float): Minimum allowed membrane potential.
                Any additional entries in dict_args will be set as attributes.

        Raises:
            ValueError: If dict_args contains a key that matches an existing attribute.
        """

        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output

        for k, v in dict_args.items():
            if hasattr(self, k):
                raise ValueError(f"Cannot override existing attribute {k!r}")
            setattr(self, k, v)

        if self.eprop:
            self.train = self.train_eprop
        else:
            self.train = self.train_bptt

        time_step = self.time_bin_size / 1000.0

        if self.tau_syn == 0.0:
            self.alpha = 0.0
        else:
            self.alpha = float(np.exp(-time_step / self.tau_syn))

        self.device = torch.device(dict_args['device'])
        self.dtype = torch.float

        self.beta = float(np.exp(-time_step / self.tau_mem))
        self.beta_adaptive_thr = float(
            np.exp(-time_step / self.tau_adaptive_thr))
        self.beta_trace = float(np.exp(-time_step / self.tau_trace))
        self.beta_trace_rec = float(np.exp(-time_step / self.tau_trace_rec))

        # with open('test_init_weight.pkl', 'rb') as f:
        #     layers = pickle.load(f)

        self.ff_layer = CuBaLIF_HW_Aware(batch_size=self.batch_size, nb_inputs=self.nb_hidden, nb_neurons=self.nb_output,
                                         fwd_scale=self.fwd_weight_scale, alpha=self.alpha, firing_threshold=self.firing_threshold,
                                         beta=self.beta, device=self.device, dtype=self.dtype, lower_bound=self.lower_bound,
                                         ref_per_timesteps=self.ref_per_timesteps, weights=None, requires_grad=True)

        self.rec_layer = CuBaRLIF_HW_Aware(batch_size=self.batch_size, nb_inputs=self.nb_inputs, nb_neurons=self.nb_hidden,
                                           fwd_scale=self.fwd_weight_scale, rec_scale=self.weight_scale_factor, alpha=self.alpha,
                                           firing_threshold=self.firing_threshold, beta_thr=self.beta_adaptive_thr,
                                           dump_thr=self.dump_thr, beta=self.beta, device=self.device, dtype=self.dtype,
                                           lower_bound=self.lower_bound, ref_per_timesteps=self.ref_per_timesteps, weights=None,
                                           requires_grad=True)

    def forward(self, input):
        """
        Run one forward pass through the SRNN on a batch of input spike-trains.

        This method resets the internal states of both the recurrent and
        feedforward readout layers, then processes the input through the
        recurrent hidden layer followed by the feedforward output layer.

        Args:
            input (torch.Tensor):
                Input tensor of shape (batch_size, timesteps, nb_inputs),
                containing the spike-train or rate-coded inputs per time bin.

        Returns:
            tuple:
                rec_outputs (list of torch.Tensor):
                    [rec_spk_rec, rec_syn_rec, rec_mem_rec]
                    Each tensor has shape (batch_size, timesteps, nb_hidden)
                    containing, respectively, the hidden-layer spike trains,
                    synaptic currents, and membrane potentials.
                ff_outputs (list of torch.Tensor):
                    [ff_spk_rec, ff_syn_rec, ff_mem_rec]
                    Each tensor has shape (batch_size, timesteps, nb_output)
                    where the last entry is the per-time-step spike count.
        """
        nb_steps = self.max_time // self.time_bin_size
        bs = input.shape[0]

        # Reset from previous batch
        # self.rec_layer.syn = torch.zeros(
        #     (bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        # self.rec_layer.mem = torch.zeros(
        #     (bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        # self.rec_layer.rst = torch.zeros(
        #     (bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        # self.rec_layer.ref_per_counter = torch.zeros(
        #     (bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        # self.rec_layer.firing_threshold = self.rec_layer.firing_threshold * \
        #     torch.ones((bs, self.nb_hidden),
        #                device=self.device, dtype=self.dtype)
        # self.rec_layer.out = torch.zeros(
        #     (bs, self.nb_hidden), device=self.device, dtype=self.dtype)

        # self.ff_layer.syn = torch.zeros(
        #     (bs, self.nb_output), device=self.device, dtype=self.dtype)
        # self.ff_layer.mem = torch.zeros(
        #     (bs, self.nb_output), device=self.device, dtype=self.dtype)
        # self.ff_layer.rst = torch.zeros(
        #     (bs, self.nb_output), device=self.device, dtype=self.dtype)
        # self.ff_layer.ref_per_counter = torch.zeros(
        #     (bs, self.nb_output), device=self.device, dtype=self.dtype)
        # self.ff_layer.firing_threshold = self.ff_layer.firing_threshold * \
        #     torch.ones((bs, self.nb_output),
        #                device=self.device, dtype=self.dtype)
        # self.ff_layer.n_spike = torch.zeros(
        #     (bs, self.nb_output), device=self.device, dtype=self.dtype)

        self.ff_layer.reset()
        self.rec_layer.reset()
        # add them as rsnn attribute?
        rec_spk_rec = torch.zeros(
            (bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_syn_rec = torch.zeros(
            (bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_mem_rec = torch.zeros(
            (bs, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device)

        ff_spk_rec = torch.zeros(
            (bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_syn_rec = torch.zeros(
            (bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_mem_rec = torch.zeros(
            (bs, nb_steps, self.nb_output), dtype=self.dtype, device=self.device)

        for t in range(nb_steps):
            # input to hidden
            rec_spk, rec_syn, rec_mem = self.rec_layer.step(input[:, t, :])
            rec_spk_rec[:, t, :] = rec_spk
            rec_syn_rec[:, t, :] = rec_syn
            rec_mem_rec[:, t, :] = rec_mem
            # hidden to output

            ff_spk, ff_syn, ff_mem = self.ff_layer.step(rec_spk)
            ff_spk_rec[:, t, :] = ff_spk
            ff_syn_rec[:, t, :] = ff_syn
            ff_mem_rec[:, t, :] = ff_mem

        '''
        with open('other_recs.pkl', 'rb') as f:
            other_recs = pickle.load(f)

        with open('spk_rec_readout.pkl', 'rb') as f:
            spk_rec_readout = pickle.load(f)

        [mem_rec_hidden, spk_rec_hidden, mem_rec_readout] = other_recs

        print("mem_rec_hidden are equal: ", torch.equal(rec_mem_rec, mem_rec_hidden))
        print("spk_rec_hidden are equal: ", torch.equal(rec_spk_rec, spk_rec_hidden))
        print("mem_rec_readout are equal: ", torch.equal(ff_mem_rec, mem_rec_readout))
        print("spk_rec_readout are equal: ", torch.equal(ff_spk_rec, spk_rec_readout))
        '''
        return [rec_spk_rec, rec_syn_rec, rec_mem_rec], [ff_spk_rec, ff_syn_rec, ff_mem_rec]

    def train_bptt(self, dataset_train, dataset_test):
        """
        Train the SRNN using Backpropagation Through Time (BPTT) and record performance.

        This method performs BPTT over a specified number of epochs. For each batch, it
        runs a forward pass, computes the supervised classification loss combined with
        spike-based regularization terms, backpropagates gradients through time, and
        updates the network weights using an optimizer. Training and test accuracies are
        computed and stored at each epoch. The set of network weights achieving the best
        test accuracy (or training accuracy if no test set provided) is saved.

        Args:
            dataset_train (torch.utils.data.Dataset):
                Training dataset yielding (input_spikes, target_labels) pairs.
            dataset_test (torch.utils.data.Dataset or None):
                Test dataset for evaluating generalization after each epoch.
                If None, only training accuracy is monitored.

        Returns:
            tuple:
                - loss_hist (list of float):
                  Mean training loss per epoch.
                - accs_hist (tuple of lists):
                  Two lists: training accuracies and test accuracies per epoch.
                - best_acc_layers (list of torch.Tensor):
                  Cloned weight tensors corresponding to the epoch with highest monitored accuracy.
        """
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                               shuffle=True, num_workers=2)
        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(
            self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                             total=self.epochs, leave=False)
        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(
                layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(
                    self.device), y_local.to(self.device)
                recs, ff = self.forward(x_local)
                # [rec_spk_rec, rec_syn_rec, rec_mem_rec]
                spk_rec_readout = ff[0]
                # [rec_spk_rec, rec_syn_rec, rec_mem_rec]
                spk_rec_hidden = recs[0]
                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                # print("m: ", m.sum())
                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes * \
                    torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                # print("reg_loss: ", reg_loss)
                reg_loss += self.reg_neurons * \
                    torch.mean(
                        torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                # print(f"{loss_val:.15f}")  # prints 10 decimal places

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                layers_update = [self.ff_layer.ff_weights.detach().clone(
                ), self.rec_layer.ff_weights.detach().clone(), self.rec_layer.rec_weights.detach().clone()]

                local_loss.append(loss_val.item())
                max_val, am = torch.max(m, 1)     # argmax over output units

                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(dataset_test)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]
                  * 100, "Test acc", accs_hist[1][-1]*100)

        return loss_hist, accs_hist, best_acc_layers

    def train_eprop(self, dataset_train, dataset_test, labels):
        """
        Train the SRNN using Eligibility Propagation (e-prop).

        This method performs e-prop over a specified number of epochs. For each batch,
        it runs a forward pass, computes surrogate gradients via eligibility traces and
        learning signals, accumulates weight gradients, and updates network parameters
        using an optimizer. Training and test accuracies (if provided) are computed and
        recorded at each epoch. The network weights yielding the highest monitored
        accuracy are saved.

        Args:
            dataset_train (torch.utils.data.Dataset):
                Training dataset yielding (input_spikes, target_labels) pairs.
            dataset_test (torch.utils.data.Dataset or None):
                Optional test dataset for evaluating performance after each epoch.
                If None, only training accuracy is monitored.
            labels (list or array-like):
                Class names used for one-hot encoding and for labeling accuracy metrics.

        Returns:
            tuple:
                - loss_hist (list of float):
                  Mean training loss per epoch.
                - accs_hist (tuple of lists):
                  Two lists containing training accuracies and test accuracies per epoch.
                - best_acc_layers (list of torch.Tensor):
                  Cloned weight tensors corresponding to the epoch with highest monitored accuracy.
        """
        print("TO DO")
        # The log softmax function across output units
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                               shuffle=True, num_workers=2)

        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(
            self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        # The optimization loop
        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                             total=self.epochs, leave=False)

        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(
                layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(
                    self.device), y_local.to(self.device)
                # reset refractory period counter for each batch
                optimizer.zero_grad()

                with open('x.pkl', 'rb') as f:
                    x_orig = pickle.load(f)

                with open('yo.pkl', 'rb') as f:
                    yo_orig = pickle.load(f)

                with open('yt.pkl', 'rb') as f:
                    yt_orig = pickle.load(f)

                with open('v.pkl', 'rb') as f:
                    v_orig = pickle.load(f)

                with open('z.pkl', 'rb') as f:
                    z_orig = pickle.load(f)

                one_hot_encoded = torch.nn.functional.one_hot(
                    y_local, num_classes=len(np.unique(labels)))

                rec, ff = self.forward(x_local)
                max_spikes, _ = torch.max(
                    ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]
                '''

                # Crea una maschera booleana per identificare i neuroni con il valore massimo
                is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

                # Genera indici casuali per scegliere tra i massimi
                rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
                rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi
                # Prendi l'indice con il valore massimo tra quelli rimasti
                _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
                '''
                with open('nb_spikes.pkl', 'rb') as f:
                    nb_spikes = pickle.load(f)

                # print("nb_spikes are equals: ", torch.equal(ff[3], nb_spikes))
                _, am = torch.max(ff[3], 2)
                with open('am.pkl', 'rb') as f:
                    am_orig = pickle.load(f)

                print("am are equal: ", torch.equal(am, am_orig))
                print("nb_spikes are equal: ", torch.equal(ff[3], nb_spikes))
                yo = torch.nn.functional.one_hot(
                    am, num_classes=len(np.unique(labels))).to(self.device)

                spk_rec_hidden = rec[0]
                spk_rec_readout = ff[0]

                print("x are equal: ", torch.equal(
                    x_local.permute(1, 0, 2), x_orig))
                print("yo are equal: ", torch.equal(
                    yo.permute(1, 0, 2), yo_orig))
                print("yt are equal: ", torch.equal(one_hot_encoded, yt_orig))
                print("v are equal: ", torch.equal(
                    rec[2].permute(1, 0, 2), v_orig))
                print("z are equal: ", torch.equal(
                    rec[0].permute(1, 0, 2), z_orig))

                self.grads_batch(x_local.permute(1, 0, 2), yo.permute(
                    1, 0, 2), one_hot_encoded, rec[2].permute(1, 0, 2), rec[0].permute(1, 0, 2))

                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes * \
                    torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                reg_loss += self.reg_neurons * \
                    torch.mean(
                        torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                optimizer.step()
                layers_update = [self.ff_layer.ff_weights.detach().clone(
                ), self.rec_layer.ff_weights.detach().clone(), self.rec_layer.rec_weights.detach().clone()]

                local_loss.append(loss_val.item())

                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            # mean_accs: mean training accuracy of current epoch (average over all batches)
            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(
                    dataset=dataset_test)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]
                  * 100, "Test acc", accs_hist[1][-1]*100)
        return loss_hist, accs_hist, best_acc_layers

    def grads_batch(self, x, yo, yt, v, z, data_steps, nb_inputs, nb_hidden):
        """
        Compute and accumulate eligibility-propagation (e-prop) gradients for one batch.

        This method uses surrogate derivatives and convolutions of eligibility traces
        to compute weight gradients for both the feedforward and recurrent layers.
        Gradients are accumulated in the `.grad` attributes of:
          - self.ff_layer.ff_weights
          - self.rec_layer.ff_weights
          - self.rec_layer.rec_weights

        Args:
            x (torch.Tensor):
                Input spike tensor of shape (time_steps, batch_size, nb_inputs).
            yo (torch.Tensor):
                Readout spike tensor (assigned outputs) of shape
                (time_steps, batch_size, nb_output).
            yt (torch.Tensor):
                Target one-hot output tensor of shape (batch_size, nb_output).
            v (torch.Tensor):
                Membrane potentials over time of shape
                (time_steps, batch_size, nb_hidden).
            z (torch.Tensor):
                Hidden-layer spike tensor over time of shape
                (time_steps, batch_size, nb_hidden).
            data_steps (int):
                Number of time steps in the batch.
            nb_inputs (int):
                Number of input neurons.
            nb_hidden (int):
                Number of hidden (recurrent) neurons.

        Returns:
            None
        """
        if self.ff_layer.ff_weights.grad is None:
            self.ff_layer.ff_weights.grad = torch.zeros_like(
                self.ff_layer.ff_weights)
        if self.rec_layer.ff_weights.grad is None:
            self.rec_layer.ff_weights.grad = torch.zeros_like(
                self.rec_layer.ff_weights)
        if self.rec_layer.rec_weights.grad is None:
            self.rec_layer.rec_weights.grad = torch.zeros_like(
                self.rec_layer.rec_weights)
        # Surrogate derivatives
        h = self.gamma * torch.max(torch.zeros_like(v), 1 - torch.abs(
            (v - self.firing_threshold) / self.firing_threshold))

        # Crea una variabile di errore vuota con le stesse dimensioni di yo
        err = torch.zeros_like(yo)

        '''
        print('x are equal: ', torch.equal(x, x_orig))
        print('yo are equal: ', torch.equal(yo, yo_orig))
        print('yt are equal: ', torch.equal(yt, yt_orig))
        print('v are equal: ', torch.equal(v, v_orig))
        print('z are equal: ', torch.equal(z, z_orig))
        '''
        # Eligibility traces convolution
        beta_conv = torch.tensor([self.beta_trace_rec ** (data_steps - i - 1)
                                 for i in range(data_steps)]).float().view(1, 1, -1).to(self.device)
        beta_rec_conv = torch.tensor([self.beta_trace ** (data_steps - i - 1)
                                     for i in range(data_steps)]).float().view(1, 1, -1).to(self.device)

        # Convoluzione Input eligibility traces
        trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(
            nb_inputs, -1, -1), padding=data_steps, groups=nb_inputs)[:, :, 1:data_steps+1]
        trace_in = trace_in.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
        trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

        # Convoluzione Recurrent eligibility traces
        trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(
            nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, :data_steps]
        trace_rec = trace_rec.unsqueeze(1).expand(-1, nb_hidden, -1, -1)
        trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

        # Output eligibility vector
        trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(
            nb_hidden, -1, -1), padding=data_steps, groups=nb_hidden)[:, :, 1:data_steps+1]

        # Ottimizzazione convoluzioni batch-wise
        trace_in = F.conv1d(trace_in.reshape(self.batch_size, nb_inputs * nb_hidden, data_steps),
                            beta_conv.expand(nb_inputs * nb_hidden, -1, -1),
                            padding=data_steps, groups=nb_inputs * nb_hidden)[:, :, 1:data_steps+1]
        trace_in = trace_in.reshape(
            self.batch_size, nb_hidden, nb_inputs, data_steps)

        trace_rec = F.conv1d(trace_rec.reshape(self.batch_size, nb_hidden * nb_hidden, data_steps),
                             beta_conv.expand(nb_hidden * nb_hidden, -1, -1),
                             padding=data_steps, groups=nb_hidden * nb_hidden)[:, :, 1:data_steps+1]
        trace_rec = trace_rec.reshape(
            self.batch_size, nb_hidden, nb_hidden, data_steps)

        # Ciclo for per calcolare l'errore 'err'
        for i in range(yo.shape[0]):
            err[i, :, :] = yo[i, :, :] - yt
        err = err.to(self.dtype)
        # Calcolo dei segnali di apprendimento
        L = torch.einsum('tbo,or->brt', err, self.ff_layer.ff_weights.t())

        # Weight gradient updates
        self.rec_layer.ff_weights.grad += (torch.sum(L.unsqueeze(
            2).expand(-1, -1, nb_inputs, -1) * trace_in, dim=(0, 3))).t()
        self.rec_layer.rec_weights.grad += (torch.sum(L.unsqueeze(
            2).expand(-1, -1, nb_hidden, -1) * trace_rec, dim=(0, 3))).t()
        self.ff_layer.ff_weights.grad += (
            torch.einsum('tbo,brt->or', err, trace_out)).t()

    def compute_classification_accuracy(self, dataset):
        """
        Evaluate classification accuracy on a dataset in batches.

        This method runs the SRNN in inference mode (no gradients), iterates over
        the provided dataset in mini-batches, performs a forward pass, sums the
        readout-layer spike counts over time to obtain class scores, picks the
        class with the highest spike count as the prediction, and computes the
        overall mean accuracy.

        Args:
            dataset (torch.utils.data.Dataset):
                A dataset yielding (input_spikes, target_labels) pairs, where
                input_spikes is a tensor of shape (batch_size, timesteps, nb_inputs)
                and target_labels is a tensor of shape (batch_size,).

        Returns:
            float:
                Mean classification accuracy (0.0 to 1.0) over all samples in the dataset.
        """
        generator = DataLoader(dataset=dataset, batch_size=self.batch_size_test, pin_memory=True,
                               shuffle=False, num_workers=2)
        accs = []

        for x_local, y_local in generator:
            x_local, y_local = x_local.to(self.device), y_local.to(self.device)
            with torch.no_grad():
                rec, ff = self.forward(x_local)

            spk_rec_readout = ff[0]  # [rec_spk_rec, rec_syn_rec, rec_mem_rec]
            m = torch.sum(spk_rec_readout, 1)  # sum over time

            max_val, am = torch.max(m, 1)     # argmax over output units

            # compare to labels
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        return np.mean(accs)


class SRNN_OG:

    def __init__(self, nb_inputs, nb_hidden, nb_output, dict_args):
        """
        Initialize the SRNN model with parameters and layers.
        All entries from dict_args are set as attributes.
        """

        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output

        for k, v in dict_args.items():
            if hasattr(self, k):
                raise ValueError(f"Cannot override existing attribute {k!r}")
            setattr(self, k, v)

        if self.eprop:
            self.train = self.train_eprop
        else:
            self.train = self.train_bptt

        time_step = self.time_bin_size / 1000.0

        if self.tau_syn == 0.0:
            self.alpha = 0.0
        else:
            self.alpha = float(np.exp(-time_step / self.tau_syn))

        self.device = torch.device(dict_args['device'])
        self.dtype = torch.float

        self.beta = float(np.exp(-time_step / self.tau_mem))
        self.beta_trace = float(np.exp(-time_step / self.tau_trace))
        self.beta_trace_rec = float(np.exp(-time_step / self.tau_trace_rec))

        self.data_steps = self.max_time // self.time_bin_size

        with open('test_init_weight.pkl', 'rb') as f:
            layers = pickle.load(f)

        self.ff_layer = CuBaLIF_HW_Aware_OG(batch_size=self.batch_size, nb_inputs=self.nb_hidden, nb_neurons=self.nb_output,
                            fwd_scale=self.fwd_weight_scale, alpha=self.alpha, firing_threshold=self.firing_threshold,
                            beta=self.beta, device=self.device, dtype=self.dtype, lower_bound=self.lower_bound,
                            ref_per_timesteps=self.ref_per_timesteps,weights=layers[1], requires_grad=True)

        self.rec_layer = CuBaRLIF_HW_Aware_OG(batch_size=self.batch_size, nb_inputs=self.nb_inputs, nb_neurons=self.nb_hidden,
                           fwd_scale=self.fwd_weight_scale, rec_scale=self.weight_scale_factor, alpha=self.alpha,
                           firing_threshold=self.firing_threshol, beta=self.beta, device=self.device, dtype=self.dtype,
                           lower_bound=self.lower_bound, ref_per_timesteps=self.ref_per_timesteps, weights=[layers[0], layers[2]],
                           requires_grad=True)


    def forward(self, input, weights):
        self.data_steps = self.max_time // self.time_bin_size
        bs = input.shape[0]
        rec_layer_ff_weight, ff_layer_weights, rec_layer_rec_weight = weights

        layers_update = weights
        # Reset from previous batch
        self.rec_layer.syn = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.mem = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.rst = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.ref_per_counter = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.firing_threshold = self.rec_layer.theta * torch.ones((bs, self.nb_hidden), device=self.device, dtype=self.dtype)
        self.rec_layer.out = torch.zeros((bs, self.nb_hidden), device=self.device, dtype=self.dtype)

        self.ff_layer.syn = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.mem = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.rst = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.ref_per_counter = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.firing_threshold = self.ff_layer.theta * torch.ones((bs, self.nb_output), device=self.device, dtype=self.dtype)
        self.ff_layer.n_spike = torch.zeros((bs, self.nb_output), device=self.device, dtype=self.dtype)

        # add them as rsnn attribute?
        rec_spk_tot = torch.zeros((bs, self.data_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_syn_tot = torch.zeros((bs, self.data_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_syn_tot = torch.zeros((bs, self.data_steps, self.nb_hidden), dtype=self.dtype, device=self.device)
        rec_mem_tot = torch.zeros((bs, self.data_steps, self.nb_hidden), dtype=self.dtype, device=self.device)

        ff_spk_tot = torch.zeros((bs, self.data_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_syn_tot = torch.zeros((bs, self.data_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_mem_tot = torch.zeros((bs, self.data_steps, self.nb_output), dtype=self.dtype, device=self.device)
        ff_nb_spk_tot = torch.zeros((bs, self.data_steps, self.nb_output), dtype=self.dtype, device=self.device)

        h = torch.einsum("abc,cd->abd", input, rec_layer_ff_weight)

        for t in range(self.data_steps):
            rec_spk, rec_syn, rec_mem = self.rec_layer.step(h[:,t,:], rec_layer_rec_weight)
            rec_spk_tot[:,t,:] = rec_spk
            rec_syn_tot[:,t,:] = rec_syn
            rec_mem_tot[:,t,:] = rec_mem

        h1 = torch.einsum("abc,cd->abd", rec_spk_tot, ff_layer_weights)

        for t in range(self.data_steps):
            ff_spk, ff_syn, ff_mem, ff_nb_spk = self.ff_layer.step(h1[:,t,:])

            ff_spk_tot[:,t,:] = ff_spk
            ff_syn_tot[:,t,:] = ff_syn
            ff_mem_tot[:,t,:] = ff_mem
            ff_nb_spk_tot[:,t,:] = ff_nb_spk

        return[rec_spk_tot, rec_syn_tot, rec_mem_tot], [ff_spk_tot, ff_syn_tot, ff_mem_tot, ff_nb_spk_tot], layers_update

    def train_bptt(self, dataset_train, dataset_test, possible_weight):
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2)
        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                            total=self.epochs, leave=False)
        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(self.device), y_local.to(self.device)
                recs, ff, layers_update = self.forward(x_local)
                if self.quantization:
                    layers_update = [ste_fn(layer, possible_weight) for layer in layers_update]
                    layers_update = [layer.to(self.dtype) for layer in layers_update]
                spk_rec_readout = ff[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
                spk_rec_hidden = recs[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                #print("m: ", m.sum())
                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes*torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                #print("reg_loss: ", reg_loss)
                reg_loss += self.reg_neurons * \
                    torch.mean(torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                #print(f"{loss_val:.15f}")  # prints 10 decimal places

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())
                max_val, am = torch.max(m, 1)     # argmax over output units
                '''
                # This is a workaround to randomly select one of the neurons that have the maximum spikes
                max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

                is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

                rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
                rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

                _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
                '''
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(dataset_test, layers_update)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]*100, "Test acc", accs_hist[1][-1]*100)

        return loss_hist, accs_hist, best_acc_layers



    def train_eprop(self, dataset_train, dataset_test, labels, possible_weight):
        # The log softmax function across output units
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()  # The negative log likelihood loss function

        generator = DataLoader(dataset=dataset_train, batch_size=self.batch_size, pin_memory=True,
                            shuffle=True, num_workers=2)

        layers = []
        layers.append(self.rec_layer.ff_weights), layers.append(self.ff_layer.ff_weights), layers.append(self.rec_layer.rec_weights)

        # The optimization loop
        loss_hist = []
        accs_hist = [[], []]
        pbar_training = tqdm(range(self.epochs), position=1,
                            total=self.epochs, leave=False)

        for _ in pbar_training:
            # learning rate decreases over epochs
            optimizer = torch.optim.Adamax(layers, lr=self.lr, betas=(0.9, 0.995))
            # if e > nb_epochs/2:
            #     lr = lr * 0.9
            local_loss = []
            # accs: mean training accuracies for each batch
            accs = []
            pbar_batches = tqdm(generator, position=2,
                                total=len(generator), leave=False)
            for x_local, y_local in pbar_batches:
                x_local, y_local = x_local.to(self.device), y_local.to(self.device)
                # reset refractory period counter for each batch
                optimizer.zero_grad()
                one_hot_encoded = torch.nn.functional.one_hot(y_local, num_classes=len(np.unique(labels)))

                rec, ff, layers_update = self.forward(x_local, layers)
                if self.quantization:
                    layers_update = [ste_fn(layer, possible_weight) for layer in layers_update]
                    layers_update = [layer.to(self.dtype) for layer in layers_update]
                ff[3][:] = ff[3][:, -1:, :].expand_as(ff[3])
                _, am = torch.max(ff[3], 2)

                '''
                # This is a workaround to randomly select one of the neurons that have the maximum spikes
                max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

                is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

                rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
                rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

                _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
                '''


                yo = torch.nn.functional.one_hot(am, num_classes=len(np.unique(labels))).to(self.device)

                spk_rec_hidden= rec[0]
                spk_rec_readout = ff[0]

                self.grads_batch(x_local.permute(1,0,2), yo.permute(1,0,2), one_hot_encoded, rec[2].permute(1,0,2), rec[0].permute(1,0,2))

                m = torch.sum(spk_rec_readout, 1)  # sum over time

                # cross entropy loss on the active read-out layer
                log_p_y = log_softmax_fn(m)

                # Here we can set up our regularizer loss
                # reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks1,1)) # L1 loss on spikes per neuron (original)
                # L1 loss on total number of spikes (hidden layer 1)
                reg_loss = self.reg_spikes * torch.mean(torch.sum(spk_rec_hidden, 1))
                # L1 loss on total number of spikes (output layer)
                # reg_loss += params['reg_spikes']*torch.mean(torch.sum(spk_rec_readout, 1))
                # print("L1: ", reg_loss)
                # reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks1,dim=0),dim=0)**2) # e.g., L2 loss on total number of spikes (original)
                # L2 loss on spikes per neuron (hidden layer 1)
                reg_loss += self.reg_neurons * \
                    torch.mean(torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0)**2)
                # L2 loss on spikes per neuron (output layer)
                # reg_loss += params['reg_neurons'] * \
                #     torch.mean(torch.sum(torch.sum(spk_rec_readout, dim=0), dim=0)**2)
                # print("L1 + L2: ", reg_loss)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local) + reg_loss
                optimizer.step()
                local_loss.append(loss_val.item())

                # compare to labels
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean((y_local == am).detach().cpu().numpy())
                accs.append(tmp)

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            # mean_accs: mean training accuracy of current epoch (average over all batches)
            mean_accs = np.mean(accs)
            accs_hist[0].append(mean_accs)

            # Calculate test accuracy in each epoch
            if dataset_test is not None:
                test_acc = self.compute_classification_accuracy(
                    dataset_test, layers_update)
                accs_hist[1].append(test_acc)  # only safe best test

                # save best training
                if mean_accs >= np.max(accs_hist[0]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())
            else:
                # save best test
                if np.max(test_acc) >= np.max(accs_hist[1]):
                    best_acc_layers = []
                    for ii in layers_update:
                        best_acc_layers.append(ii.detach().clone())

            pbar_training.set_description("{:.2f}%, {:.2f}%, {:.2f}.".format(
                accs_hist[0][-1]*100, accs_hist[1][-1]*100, loss_hist[-1]))
            print("Train acc: ", accs_hist[0][-1]*100, "Test acc", accs_hist[1][-1]*100)
        return loss_hist, accs_hist, best_acc_layers

    def grads_batch(self, x, yo, yt, v, z):

        if self.ff_layer.ff_weights.grad is None:
            self.ff_layer.ff_weights.grad = torch.zeros_like(self.ff_layer.ff_weights)
        if self.rec_layer.ff_weights.grad is None:
            self.rec_layer.ff_weights.grad = torch.zeros_like(self.rec_layer.ff_weights)
        if self.rec_layer.rec_weights.grad is None:
            self.rec_layer.rec_weights.grad = torch.zeros_like(self.rec_layer.rec_weights)

        # Surrogate derivatives
        h = self.gamma * torch.max(torch.zeros_like(v), 1 - torch.abs((v - self.firing_threshold) / self.firing_threshold))

        err = torch.zeros_like(yo)

        # Eligibility traces convolution
        beta_conv     = torch.tensor([self.beta_trace_rec ** (self.data_steps - i - 1) for i in range(self.data_steps)]).float().view(1, 1, -1).to(self.device)
        beta_rec_conv = torch.tensor([self.beta_trace ** (self.data_steps - i - 1) for i in range(self.data_steps)]).float().view(1, 1, -1).to(self.device)

        # Convoluzione Input eligibility traces
        trace_in = F.conv1d(x.permute(1, 2, 0), beta_rec_conv.expand(self.nb_inputs, -1, -1), padding=self.data_steps, groups=self.nb_inputs)[:, :, 1:self.data_steps+1]
        trace_in = trace_in.unsqueeze(1).expand(-1, self.nb_hidden, -1, -1)
        trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

        trace_rec = F.conv1d(z.permute(1, 2, 0), beta_rec_conv.expand(self.nb_hidden, -1, -1), padding=self.data_steps, groups=self.nb_hidden)[:, :, :self.data_steps]
        trace_rec = trace_rec.unsqueeze(1).expand(-1, self.nb_hidden, -1, -1)
        trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)

        # Output eligibility vector
        trace_out = F.conv1d(z.permute(1, 2, 0), beta_conv.expand(self.nb_hidden, -1, -1), padding=self.data_steps, groups=self.nb_hidden)[:, :, 1:self.data_steps+1]

        trace_in = F.conv1d(trace_in.reshape(self.batch_size, self.nb_inputs * self.nb_hidden, self.data_steps),
                            beta_conv.expand(self.nb_inputs * self.nb_hidden, -1, -1),
                            padding=self.data_steps, groups=self.nb_inputs * self.nb_hidden)[:, :, 1:self.data_steps+1]
        trace_in = trace_in.reshape(self.batch_size, self.nb_hidden, self.nb_inputs, self.data_steps)

        trace_rec = F.conv1d(trace_rec.reshape(self.batch_size, self.nb_hidden * self.nb_hidden, self.data_steps),
                            beta_conv.expand(self.nb_hidden * self.nb_hidden, -1, -1),
                            padding=self.data_steps, groups=self.nb_hidden * self.nb_hidden)[:, :, 1:self.data_steps+1]
        trace_rec = trace_rec.reshape(self.batch_size, self.nb_hidden, self.nb_hidden, self.data_steps)

        for i in range(yo.shape[0]):
            err[i,:,:] = yo[i,:,:] - yt
        err = err.to(self.dtype)
        # Learning signal
        L = torch.einsum('tbo,or->brt', err, self.ff_layer.ff_weights.t())

        # Weight gradient updates
        self.rec_layer.ff_weights.grad += (torch.sum(L.unsqueeze(2).expand(-1, -1, self.nb_inputs, -1) * trace_in, dim=(0, 3))).t()
        self.rec_layer.rec_weights.grad += (torch.sum(L.unsqueeze(2).expand(-1, -1, self.nb_hidden, -1) * trace_rec, dim=(0, 3))).t()
        self.ff_layer.ff_weights.grad += (torch.einsum('tbo,brt->or', err, trace_out)).t()

    def compute_classification_accuracy(self, dataset, weights):
        """ Computes classification accuracy on supplied data in batches. """

        generator = DataLoader(dataset=dataset, batch_size=self.batch_size_test, pin_memory=True,
                            shuffle=False, num_workers=2)
        accs = []

        for x_local, y_local in generator:
            x_local, y_local = x_local.to(self.device), y_local.to(self.device)
            with torch.no_grad():
                rec, ff, _ = self.forward(x_local, weights)

            spk_rec_readout = ff[0]  # [rec_spk_tot, rec_syn_tot, rec_mem_tot]
            m = torch.sum(spk_rec_readout, 1)  # sum over time

            max_val, am = torch.max(m, 1)     # argmax over output units
            '''
            # This is a workaround to randomly select one of the neurons that have the maximum spikes
            max_spikes, _ = torch.max(ff[3], dim=2, keepdim=True)  # [batch, tempo, 1]

            is_max = ff[3] == max_spikes  # [batch, tempo, neurone]

            rand_vals = torch.rand(ff[3].shape)  # Numeri casuali [0,1] per ogni neurone
            rand_vals[~is_max] = -1  # Imposta a -1 i neuroni che non sono massimi

            _, am = torch.max(rand_vals, dim=2)  # Ora il massimo è scelto casualmente tra i pari
            '''

            # compare to labels
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)

        return np.mean(accs)
