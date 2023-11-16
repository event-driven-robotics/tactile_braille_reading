'''
SpyTorch neuron models.
'''

import torch
from numpy import sqrt


class feedforward_layer:
    '''
    class to initialize and compute spiking feedforward layer
    '''
    def create_layer(nb_inputs, nb_outputs, fwd_scale, device='cpu', dtype=torch.float):
        ff_layer = torch.empty(
            (nb_inputs, nb_outputs),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0, std=fwd_scale/sqrt(nb_inputs))
        return ff_layer

    def compute_activity(spike_fn, nb_input, nb_neurons, input_activity, alpha, beta, nb_steps, device='cpu', dtype=torch.float):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        mem_rec = []
        spk_rec = []

        # Compute feedforward layer activity
        for t in range(nb_steps):
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst_out = out.detach()

            new_syn = alpha*syn + input_activity[:, t]
            new_mem = (beta*mem + syn)*(1.0-rst_out)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


class recurrent_layer:
    '''
    class to initialize and compute spiking recurrent layer
    '''
    def create_layer(nb_inputs, nb_outputs, fwd_scale, rec_scale, device='cpu', dtype=torch.float):
        ff_layer = torch.empty(
            (nb_inputs, nb_outputs),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(ff_layer, mean=0.0, std=fwd_scale/sqrt(nb_inputs))

        rec_layer = torch.empty(
            (nb_outputs, nb_outputs),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(rec_layer, mean=0.0, std=rec_scale/sqrt(nb_inputs))
        return ff_layer,  rec_layer

    def compute_activity(spike_fn, nb_input, nb_neurons, input_activity, layer, alpha, beta, nb_steps, device='cpu', dtype=torch.float):
        syn = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        mem = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        out = torch.zeros((nb_input, nb_neurons),
                          device=device, dtype=dtype)
        mem_rec = []
        spk_rec = []

        # Compute recurrent layer activity
        for t in range(nb_steps):
            # input activity plus last step output activity
            h1 = input_activity[:, t] + \
                torch.einsum("ab,bc->ac", (out, layer))
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = alpha*syn + h1
            new_mem = (beta*mem + syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        # Now we merge the recorded membrane potentials into a single tensor
        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)
        return spk_rec, mem_rec


class trainable_time_constants:
    def create_time_constants(nb_neurons, alpha_mean, beta_mean, device='cpu', dtype=torch.float):
        alpha = torch.empty((nb_neurons),  device=device,
                            dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(
            alpha, mean=alpha_mean, std=alpha_mean/10)

        beta = torch.empty((nb_neurons),  device=device,
                           dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(
            beta, mean=beta_mean, std=beta_mean/10)
        return alpha, beta
