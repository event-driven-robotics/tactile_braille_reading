# E-prop implementation audit: repository versus Bellec and Frenkel

## 1. Purpose and scope

This document compares the repository's current supervised LIF e-prop implementation with:

1. G. Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons* (2020), available in [`Bellec et al. - A solution to the learning dilemma for recurrent networks of spiking neurons.pdf`](Bellec%20et%20al.%20-%20A%20solution%20to%20the%20learning%20dilemma%20for%20recurrent%20networks%20of%20spiking%20neurons.pdf); and
2. C. Frenkel and G. Indiveri, *ReckOn: A 28nm Sub-mm² Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales* (2022), available in [`Frenkel, Indiveri - 2022 - ReckOn...pdf`](Frenkel%2C%20Indiveri%20-%202022%20-%20ReckOn%20A%2028nm%20Sub-mm2%20Task-Agnostic%20Spiking%20Recurrent%20Neural%20Network%20Processor%20Enabling%20On-Chip%20Learning%20over.pdf).

The equations and source locations used here are reconstructed in [`eprop_algorithm_reference.md`](eprop_algorithm_reference.md). The implementation under review is primarily:

- `utils/train_snn.py::grads_batch`, which constructs the manual gradients;
- `utils/neuron_models.py::recurrent_layer.compute_activity`, which determines the hidden-neuron timing, reset, refractory, and synaptic dynamics;
- `utils/snn.py::run_snn`, which constructs the leaky non-spiking readout; and
- `utils/train_snn.py::train`, which defines the loss signal and update schedule.

This is a code-to-equation audit. It does not claim that either paper's exact task, preprocessing, hyperparameters, or hardware has been reproduced.

## 2. Executive verdict

The central algebraic distinction between the two modes is implemented correctly:

- `bellec` maintains the synapse-indexed, output-leak-filtered eligibility $\bar e_{ji}^t$;
- `frenkel` omits that second hidden-weight filter and uses the instantaneous product of learning signal, surrogate/STE, and a presynaptic trace; and
- both modes retain the output-leak-filtered hidden-spike trace for output weights.

The audit initially found several forward/eligibility inconsistencies. They have
now been resolved as follows:

| Priority | Finding | Resolution |
|---|---|---|
| P0 | Forward/eligibility timing | The forward loop now integrates `x[t]` and `z[t-1]` into recorded `v[t]` before computing `z[t]`. |
| P0 | Multiplicative hard reset | Both modes use the exact stopped-reset factor $1-z_j^{t-1}$; a warning identifies this as a noncanonical, synapse-state-expanding choice. |
| P0 | Refractory behavior | Membrane/input integration continues biologically while spike emission and the e-prop spike derivative are masked. |
| P1 | Recurrent diagonal gradients | Manual recurrent gradients are explicitly zeroed on the diagonal. |
| P1 | Synaptic-current eligibility | `--synapse` is rejected for e-prop until the required two-state eligibility is implemented. |
| P1 | Bellec pseudo-derivative | Bellec mode now uses the named fixed factor $0.3/v_{\mathrm{th}}$ without adding a CLI argument. |
| P2 | Batched Adamax execution | Deliberately retained; it remains different from ReckOn's physical online updates. |

Focused numerical tests now cover causal timing, hard-reset eligibility, refractory
masking, and recurrent diagonal gradients. The code is a software realization of
the two learning-rule structures, while Frenkel mode remains **ReckOn-inspired**
rather than a reproduction of the chip's update and quantization machinery.

Biological convention used for refractoriness: the absolute refractory period
is caused primarily by transient sodium-channel inactivation, with potassium
conductance contributing to repolarization; it prevents another action
potential rather than disconnecting synaptic input. See the NCBI summaries on
[the neuronal action potential](https://www.ncbi.nlm.nih.gov/books/NBK546639/)
and [the refractory period](https://www.ncbi.nlm.nih.gov/books/NBK11146/).

## 3. Shared forward model: the timing that eligibility must differentiate

### 3.1 What the forward loop actually computes

With synaptic-current dynamics disabled, the recurrent loop now performs these
operations at code index $t$:

1. construct the drive from current input `x[t]` and retained spike `z[t-1]`;
2. integrate that drive into the membrane recorded as $v^t$;
3. threshold $v^t$ to obtain and record $z^t$; and
4. reset the state carried into the next iteration.

Writing the post-reset state as $u^t$, the implemented recurrence is

$$
v_j^t=\beta u_j^{t-1}
+\sum_i W_{ji}^{\mathrm{in}}x_i^t
+\sum_i W_{ji}^{\mathrm{rec}}z_i^{t-1},
$$

$$
z_j^t=H(v_j^t-v_{\mathrm{th}}),
$$

followed by either subtractive

$$
u_j^t=v_j^t-v_{\mathrm{th}}\operatorname{stopgrad}(z_j^t),
$$

or multiplicative reset

$$
u_j^t=v_j^t(1-\operatorname{stopgrad}(z_j^t)).
$$

Both e-prop branches pair the recorded $(v^t,z^t)$ with direct terms

$$
p_{\mathrm{in}}^t=\beta p_{\mathrm{in}}^{t-1}+x^t,
\qquad
p_{\mathrm{rec}}^t=\beta p_{\mathrm{rec}}^{t-1}+z^{t-1}.
$$

The direct terms are therefore causally aligned with the membrane whose spike
derivative they are paired with.

### 3.2 Comparison with Bellec's timing

Bellec writes

$$
v_j^{t+1}
=\alpha v_j^t
+\sum_iW_{ji}^{\mathrm{rec}}z_i^t
+\sum_iW_{ji}^{\mathrm{in}}x_i^{t+1}
-z_j^t v_{\mathrm{th}}.
$$

Relabeling Bellec's displayed $t+1$ state as repository code-time `t` gives
exactly the implemented dependency on `x[t]` and `z[t-1]`. The original extra
recurrent delay has been removed.

### 3.3 Readout timing is internally consistent

The e-prop readout computes

$$
y^t=\kappa y^{t-1}+W^{\mathrm{out}}z^t+b,
\qquad
\pi^t=\operatorname{softmax}(y^t).
$$

Both modes update $q^t=\kappa q^{t-1}+z^t$ before applying the output error at $t$. That is the correct derivative of the implemented leaky readout with respect to $W^{\mathrm{out}}$. No corresponding off-by-one error was found in the output-weight trace.

## 4. Bellec implementation audit

### 4.1 What matches Bellec

Subject to the forward-timing and neuron-model qualifications below, the `bellec` branch correctly implements the important LIF structure:

- input and recurrent presynaptic traces use the recurrent membrane retention factor `beta_mem_rec`;
- the triangular pseudo-derivative is evaluated from the recorded hidden membrane;
- `e_in` and `e_rec` pair the current postsynaptic pseudo-derivative with the corresponding presynaptic trace;
- `bar_e_in` and `bar_e_rec` apply the readout leak `beta_mem` to the complete synapse-specific product;
- the symmetric learning signal is $(W^{\mathrm{out}})^T(\pi-y^*)$ using the live mathematical output weights;
- the output gradient uses a `beta_mem`-filtered hidden-spike trace; and
- input, recurrent, and output tensor orientations agree with the paper's postsynaptic-by-presynaptic convention.

Using `beta_mem_rec` for hidden eligibility is faithful to Bellec's basic LIF
recursion, and using `beta_mem` is faithful to the implemented leaky readout.
The former independent `tau_trace` and `tau_trace_out` controls have been
removed rather than retained as inactive debug parameters.

Prediction-minus-target error followed by `optimizer.step()` has the correct sign. It is the gradient-sign counterpart of papers that display target-minus-prediction as a direct weight-change direction.

### 4.2 Timing alignment: resolved

The branch updates `pre_in` with `x[t]` and `pre_rec` with `z[t-1]` and pairs
them with the surrogate at `v[t]`. The revised forward loop gives `v[t]` exactly
those direct dependencies. Deterministic forward and e-prop tests lock this
convention.

### 4.3 Reset handling: resolved with two variants

Bellec's default LIF eligibility stops the derivative through the spike in a **subtractive** reset:

$$
v^{t+1}=\beta v^t+I^t-v_{\mathrm{th}}\operatorname{stopgrad}(z^t),
$$

which leaves the local state Jacobian equal to $\beta$.

The repository defaults to a **multiplicative** hard reset:

$$
v^{t+1}=(\beta v^t+I^t)(1-\operatorname{stopgrad}(z^t)).
$$

Even though the spike is detached, this Jacobian is

$$
\frac{\partial v^{t+1}}{\partial v^t}=\beta(1-z^t),
$$

not $\beta$.

With `--soft_reset`, the repository uses Bellec's subtractive stopped-reset
recursion. With hard reset, it now maintains the required postsynaptic,
synapse-indexed eligibility and multiplies its history by $1-z_j^{t-1}$. A
runtime warning explains that the hard-reset path is an extension rather than
Bellec's default printed LIF rule.

### 4.4 Refractory handling: resolved

The forward model now continues synaptic and membrane integration during the
refractory interval while suppressing spike emission. The hidden refractory
mask is recorded and passed to `grads_batch`, where the Bellec pseudo-derivative
and Frenkel STE are zeroed. This is both compatible with Bellec's stated
pseudo-derivative convention and closer to biological refractoriness than
discarding all incoming drive.

### 4.5 Pseudo-derivative normalization: resolved

Bellec mode now uses

$$
\psi_j^t
=\frac{0.3}{v_{\mathrm{th}}}
\max\left(0,1-\left|\frac{v_j^t-v_{\mathrm{th}}}{v_{\mathrm{th}}}\right|\right)
$$

through the internal named constant
`BELLEC_PSEUDO_DERIVATIVE_DAMPENING = 0.3` and applies the full
$0.3/v_{\mathrm{th}}$ normalization. No new CLI argument is needed. The
existing `gamma` remains the BPTT fast-sigmoid scale and Frenkel triangular-STE
amplitude.

### 4.6 Loss and learning signal: valid adaptation with caveats

The repository uses time-resolved softmax cross entropy:

$$
\delta^t=\pi^t-y^*,
\qquad
L^t=(W^{\mathrm{out}})^T\delta^t.
$$

This is a valid classification version of symmetric e-prop. When `delayed_output` is set, gradients are accumulated only for the selected final error times while eligibility continues to include earlier causal activity. That is coherent for a loss defined only on the final window.

Differences that should be explicit are:

- the same sequence label is imposed at every selected error time;
- Bellec and Frenkel gradients are intentionally summed over the batch and all
  selected error timesteps, with no $1/B$ or $1/T$ normalization;
- `reg_spikes` and `reg_neurons` are ignored in the e-prop branch even though their nonzero CLI defaults may imply otherwise;
- the displayed tracking loss is computed from summed probabilities and is not the objective whose manual gradient is applied; and
- only symmetric feedback through current output weights is implemented, not random e-prop or an ideal total learning signal.

None of these invalidates the central e-prop factorization, but they affect comparison with paper experiments and with BPTT baselines.

### 4.7 Bellec conclusion

The synapse-indexed `bar_e` implementation, forward timing, reset Jacobian, and
refractory mask are now mutually consistent. The closest paper-faithful
configuration is:

- exponential decay;
- no synaptic-current state;
- subtractive `soft_reset` (hard reset is a tested extension);
- continued integration with the pseudo-derivative masked during refractoriness;
- no lower-bound clamp or threshold noise unless their derivative conventions are implemented; and
- the current aligned forward/eligibility indexing.

## 5. Frenkel/ReckOn implementation audit

### 5.1 What matches Frenkel's modified rule

The `frenkel` branch correctly captures the distinguishing ReckOn factorization:

$$
g_{ji}^{\mathrm{in},t}=L_j^t h_j^t p_i^{\mathrm{in},t},
\qquad
g_{ji}^{\mathrm{rec},t}=L_j^t h_j^t p_i^{\mathrm{rec},t},
$$

with no readout-leak recursion around $h_j^tp_i^t$. It also correctly retains

$$
q_j^t=\kappa q_j^{t-1}+z_j^t,
\qquad
g_{kj}^{\mathrm{out},t}=\delta_k^tq_j^t.
$$

Using the hidden membrane leak for the input/recurrent presynaptic trace and the readout leak for the output trace matches the printed ReckOn equations. Its outer-product form has neuron-scaled eligibility **state**, even though computing dense updates still touches every weight.

### 5.2 Shared timing, reset, and refractory handling: resolved

Frenkel mode now uses the same causally aligned `x[t]`, `z[t-1]`, and `v[t]`
convention. Its STE is zero during recorded refractory steps. With multiplicative
hard reset, it uses a reset-aware synapse-indexed trace; this preserves the
correct stopped-reset Jacobian but deliberately gives up Frenkel's neuron-scaled
ET storage advantage. A warning makes that tradeoff explicit.

### 5.3 STE choice: permissible but not a ReckOn reproduction

Frenkel describes a programmable five-segment, 5-bit signed LUT for the spiking activation STE; the paper does not prescribe the repository's triangular analytic function. The triangular surrogate is therefore a reasonable software choice, but `gamma=15` is not derivable from the paper and should be treated as a tuned hyperparameter.

This activation STE is distinct from `STEFunction`, which passes gradients through deterministic forward weight quantization.

### 5.4 Update schedule and quantization: deliberately different

ReckOn applies local weight changes at each time step to stored 8-bit weights using stochastic updates. The repository instead:

- stores the entire forward trajectory needed by `grads_batch`;
- accumulates all selected time and batch contributions;
- performs one Adamax step per batch;
- keeps floating-point master weights; and
- optionally uses a deterministic nearest-level quantized proxy in the forward pass.

Consequently, `quantize_weights=True` does not reproduce ReckOn's stochastic rounding, online update schedule, optimizer, or no-extra-weight-memory property. Calling the branch `frenkel` is useful for identifying its algebra, but `ReckOn-style` or `ReckOn-inspired` is more accurate for the end-to-end training system.

### 5.5 Long-timescale model choice is not reproduced by default

ReckOn demonstrates its simplified rule with deliberately long LIF leakage on long-memory tasks, including a 2 s leak in the delayed-navigation comparison discussed in the paper. The repository default is `tau_mem_rec=0.06` s. This is not intrinsically wrong for tactile Braille, but selecting `frenkel` does not by itself reproduce ReckOn's long-timescale neuron model.

There is no independent trace-time control. Users seeking a longer Frenkel ET
must change `tau_mem_rec`, because the ET uses the LIF membrane decay as in the
printed rule.

### 5.6 Frenkel conclusion

The branch implements the simplified hidden-weight factorization and output
trace consistently with the revised default forward dynamics. It remains a
software realization of Frenkel's modified e-prop algebra with a triangular STE
and batched Adamax updates, not a reproduction of ReckOn's hardware system.

## 6. Additional implementation defects and unsupported options

### 6.1 Masked recurrent self-connections: resolved

The forward pass masks recurrent self-connections, and both manual-gradient
branches now zero the recurrent gradient diagonal before optimization. A
focused regression test covers this invariant.

### 6.2 Synaptic-current dynamics are explicitly unsupported

With `--synapse`, the forward neuron has both synaptic-current and membrane state:

$$
s^{t+1}=a s^t+W u^t,
\qquad
v^{t+1}=\beta v^t+s^t-\text{reset}.
$$

The eligibility is then a state vector whose recursion contains both $a$ and
$\beta$. Neither e-prop branch includes that vector; both implement the
one-state LIF trace only.

Both e-prop modes now reject `synapse=True` with a clear error until the
two-state eligibility is derived and tested. BPTT can continue using synaptic
current dynamics.

### 6.3 Lower clamp and threshold noise are not differentiated: P1/P2

If `lower_bound` clips the membrane, the local derivative through the clamp is zero in the clipped region, but eligibility continues unchanged. If threshold noise is enabled, the forward spike uses the sampled noisy threshold while the surrogate uses the fixed nominal threshold. Both options therefore train a surrogate graph different from the executed forward graph.

These can be accepted approximations, but should be opt-in and documented. A strict-validation mode should reject them.

### 6.4 Input copies: resolved

The training path now passes the same expanded presynaptic activity used by the
forward pass into `grads_batch` when `nb_input_copies > 1`.

### 6.5 “Online” describes the equations, not current execution

The current forward pass records complete `v`, `z`, readout, and input sequences before `grads_batch` replays them. Frenkel mode reduces the size of the recurrent eligibility state within that replay, but the training path still stores time-indexed activations. Bellec mode additionally allocates batch-by-synapse `bar_e` tensors.

This is suitable for equation validation and GPU batching. It should not be used as evidence of end-to-end online memory scaling until forward simulation, trace updates, errors, and optimizer updates are fused into a streaming loop.

## 7. Implementation and remaining validation order

### Phase 1: establish a single causal convention — complete

1. The intended recurrence is documented with code-time indices.
2. The recurrent forward loop integrates `x[t]` and `z[t-1]` into recorded `v[t]` before computing `z[t]`.
3. Both eligibility branches use that recurrence.
4. Deterministic timing and current-input eligibility tests cover the convention.

### Phase 2: make the neuron model compatible — complete for supported options

1. Subtractive reset uses the canonical recursion; multiplicative reset uses its derived Jacobian and emits a warning.
2. Refractory masks are exposed and applied while membrane integration continues.
3. Synaptic dynamics are rejected; lower clamping and threshold noise remain documented approximations.
4. Recurrent diagonal gradients are zeroed.

### Phase 3: validate equations numerically — complete

1. With $\kappa=0$ and identical surrogates, Bellec and Frenkel hidden gradients are equal.
2. With $\kappa\ne0$, a two-step test confirms Bellec's historical eligibility contribution and its intended absence in Frenkel.
3. Output gradients are equal between modes for identical $\delta,z,\kappa$.
4. Bellec's recursion matches automatic differentiation of a small triangular-surrogate, stopped-reset reference graph.
5. Separate tests cover spike/reset behavior, refractory masking and continued integration, zero recurrent diagonals, a delayed-output window, and `nb_input_copies=2`.

### Phase 4: calibration and variant policy

1. Bellec's pseudo-derivative normalization is implemented independently of the existing `gamma=15` setting.
2. The inactive `tau_trace`/`tau_trace_out` CLI parameters have been removed.
3. No `reckon_hardware` variant or per-step stochastic integer update is planned;
   Frenkel mode implements only the modified e-prop algebra in the repository's
   batched floating-point training system.
4. Gradient normalization is an explicit algorithm policy: BPTT retains
   PyTorch's batch-mean NLL objective, while Bellec and Frenkel retain summed
   manual gradients over batch and selected timesteps.

The summed e-prop convention makes gradient magnitude depend on batch size and
the number of selected error timesteps. Learning rates must therefore be tuned
for the configured batch/window size. A partial final batch contributes a
smaller sum; fixed-size batches (`drop_last`) or reference-size learning-rate
rescaling would be required if invariant update magnitude becomes more
important than retaining every sample.

## 8. Validation status matrix

| Component | Bellec mode | Frenkel mode |
|---|---|---|
| Hidden algebraic factorization | Matches intended structure | Matches intended simplified structure |
| Output trace and gradient | Matches implemented readout | Matches implemented readout |
| Error sign and symmetric feedback | Correct | Correct |
| Forward/eligibility timing | Aligned and tested | Aligned and tested |
| Subtractive reset | Canonical recursion | Preserves neuron-scaled trace |
| Multiplicative reset | Exact reset-aware extension; warning | Exact reset-aware extension; warning; synapse-sized state |
| Default refractory behavior | Continued integration; derivative masked | Continued integration; STE masked |
| Recurrent diagonal constraint | Enforced | Enforced |
| Synaptic-current option | Rejected until supported | Rejected until supported |
| Paper surrogate/STE | Paper normalization implemented | Analytic implementation choice; hardware LUT not reproduced |
| Update schedule | Batched floating-point adaptation | Batched floating-point adaptation; no hardware-update variant planned |
| Operational online memory | Not implemented | Not implemented despite reduced eligibility-state algebra |

The appropriate near-term validation claim is:

> The repository implements causally aligned Bellec and Frenkel gradient
> factorizations for its supported one-state exponential LIF model. Remaining
> differences concern optional model features and the software-versus-hardware
> training schedule, not the repaired core timing/reset/refractory recursions.
