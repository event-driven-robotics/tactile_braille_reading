# E-prop computations: Bellec (2020) and Frenkel–Indiveri (2022)

This note is an implementation-oriented reconstruction of the learning rules in:

1. G. Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11, 3625 (2020). Repository PDF: `docs/Bellec et al. - A solution to the learning dilemma for recurrent networks of spiking neurons.pdf`.
2. C. Frenkel and G. Indiveri, *ReckOn: A 28nm Sub-mm² Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales*, ISSCC (2022). Repository PDF: `docs/Frenkel, Indiveri - 2022 - ReckOn A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over.pdf`.

The most important result for code review is:

> Frenkel does **not** merely rewrite Bellec's LIF eligibility as a postsynaptic surrogate derivative times a presynaptic trace; Bellec already has that factorization before output filtering. Frenkel's consequential simplification is that input/recurrent updates use the instantaneous product of learning signal, STE, and presynaptic trace, instead of Bellec's additional output-leak filtering of the synapse-specific STE × presynaptic-trace product. Frenkel retains a low-pass trace for the output-weight update.

This note covers the supervised LIF case used by ReckOn and by this repository. Bellec's ALIF and reinforcement-learning extensions are summarized only where they clarify the LIF derivation.

## 1. Notation and conventions

Indices:

- $t$: discrete time step
- $i$: presynaptic input or recurrent neuron
- $j$: postsynaptic recurrent LIF neuron
- $k$: output neuron

Variables:

- $x_i^t$: input activity
- $z_j^t \in \{0,1\}$: recurrent-neuron spike
- $v_j^t$: recurrent-neuron membrane potential
- $y_k^t$: leaky real-valued output
- $y_k^{*,t}$: target
- $W_{ji}^{\mathrm{in}}, W_{ji}^{\mathrm{rec}}, W_{kj}^{\mathrm{out}}$: input, recurrent, and output weights
- $v_{\mathrm{th}}$: firing threshold
- $\alpha = \exp(-\Delta t/\tau_m)$: recurrent LIF membrane leak
- $\kappa = \exp(-\Delta t/\tau_{\mathrm{out}})$: output-neuron leak
- $\psi_j^t$: Bellec pseudo-derivative of the spike function
- $h_j^t$: Frenkel's STE value in Figure 29.4.3; this is **not** Bellec's generic hidden-state vector $h_j^t$

For a time series $a^t$, define the causal low-pass operator

$$
\mathcal F_\lambda(a^t)
= \lambda\,\mathcal F_\lambda(a^{t-1}) + a^t.
$$

Two sign conventions occur in implementations:

- A loss gradient normally uses prediction minus target, $\delta_k^t = y_k^t-y_k^{*,t}$, followed by $W \leftarrow W-\eta\nabla_W E$.
- Frenkel Figure 29.4.3 writes a weight-change direction using target minus prediction, $d_k^t=y_k^{*,t}-y_k^t=-\delta_k^t$.

They are equivalent if the optimizer/update sign is handled consistently.

## 2. Bellec et al.: original e-prop

### 2.1 General recurrent-system factorization

Bellec describes a recurrent neuron by a hidden state and an observable state:

$$
h_j^t=M(h_j^{t-1},z^{t-1},x^t;W_j),
\qquad
z_j^t=f(h_j^t).
$$

The eligibility **vector** for synapse $i\to j$ is

$$
\epsilon_{ji}^t
=
\frac{\partial h_j^t}{\partial h_j^{t-1}}\epsilon_{ji}^{t-1}
+
\frac{\partial h_j^t}{\partial W_{ji}},
$$

and the scalar eligibility trace is

$$
e_{ji}^t
=
\frac{\partial z_j^t}{\partial h_j^t}\cdot\epsilon_{ji}^t.
$$

The recurrent loss gradient can then be factorized as

$$
\frac{dE}{dW_{ji}}
=
\sum_t L_j^t e_{ji}^t,
\qquad
L_j^t=\frac{dE}{dz_j^t}.
$$

For a differentiable recurrent system, this factorization is exact when the ideal total learning signal $dE/dz_j^t$ is used. In a spiking network, this statement is conditional on the chosen pseudo-derivative: the resulting eligibility recursion is exact for the surrogate-gradient computational graph, but it is not the classical derivative of the discontinuous Heaviside forward model. Computing the ideal learning signal requires backward propagation through the rest of the recurrent computation. Online e-prop keeps the local eligibility recursion exact under that convention but replaces the ideal signal with an online approximation, commonly the direct partial derivative $\partial E/\partial z_j^t$. Thus “e-prop” contains two distinct ideas that should not be conflated:

1. an exact, forward-time local eligibility recursion;
2. an approximate, online learning signal.

Relevant Bellec equations: main-text Eqs. (1)–(3), Methods Eqs. (13), (14), and (20), physical PDF pages 3 and 12.

### 2.2 Bellec LIF dynamics

Bellec's LIF neuron is

$$
v_j^{t+1}
=
\alpha v_j^t
+\sum_{i\ne j}W_{ji}^{\mathrm{rec}}z_i^t
+\sum_iW_{ji}^{\mathrm{in}}x_i^{t+1}
-z_j^t v_{\mathrm{th}},
$$

$$
z_j^t=H(v_j^t-v_{\mathrm{th}}).
$$

The paper uses a subtractive reset. It also excludes self-connections in the displayed recurrent sum. Its simulations additionally clamp $z_j^t$ to zero for a refractory period of 2–5 ms after a spike, depending on the experiment; this forward refractory behavior is separate from masking the pseudo-derivative during refractoriness.

Bellec notes that this state equation omits a factor $1-\alpha$ that appeared in an earlier formulation. This is a normalization convention rather than a different neuron model when weights, threshold, and related voltage parameters are scaled consistently. An implementation that multiplies its input current by $1-\alpha$ therefore cannot be compared parameter-for-parameter without accounting for that scaling.

Because the Heaviside derivative is undefined at threshold and zero elsewhere, Bellec substitutes the triangular pseudo-derivative

$$
\psi_j^t
=
\frac{\gamma_{\mathrm{pd}}}{v_{\mathrm{th}}}
\max\!\left(
0,
1-\left|\frac{v_j^t-v_{\mathrm{th}}}{v_{\mathrm{th}}}\right|
\right)
$$

for a LIF neuron outside its refractory period, and sets it to zero during refractoriness. The Methods state $\gamma_{\mathrm{pd}}=0.3$ for the reported LIF/ALIF pseudo-derivative convention. The scale convention must be checked rather than matched by parameter name alone; this repository's `gamma` is not automatically the same numerical convention.

Relevant Bellec equations: Methods Eqs. (6), (7), and the pseudo-derivative paragraph on physical PDF pages 10–11.

### 2.3 Default LIF eligibility: reset derivative omitted

Bellec's main derivation stops the derivative through the emitted spike in the reset path. It therefore uses

$$
\frac{\partial v_j^{t+1}}{\partial v_j^t}=\alpha.
$$

For a recurrent synapse, aligned to the state update above,

$$
\epsilon_{ji}^{\mathrm{rec},t+1}
=
\alpha\epsilon_{ji}^{\mathrm{rec},t}+z_i^t
=
\mathcal F_\alpha(z_i^t),
$$

$$
e_{ji}^{\mathrm{rec},t+1}
=
\psi_j^{t+1}\epsilon_{ji}^{\mathrm{rec},t+1}.
$$

For an input synapse,

$$
\epsilon_{ji}^{\mathrm{in},t+1}
=
\alpha\epsilon_{ji}^{\mathrm{in},t}+x_i^{t+1},
$$

$$
e_{ji}^{\mathrm{in},t+1}
=
\psi_j^{t+1}\epsilon_{ji}^{\mathrm{in},t+1}.
$$

Equivalently, at the time at which (z_j^t) is evaluated,

$$
e_{ji}^{\mathrm{rec},t}=\psi_j^t\,\mathcal F_\alpha(z_i^{t-1}),
\qquad
e_{ji}^{\mathrm{in},t}=\psi_j^t\,\mathcal F_\alpha(x_i^t).
$$

The one-step difference between recurrent and input activity follows from Bellec's state equation: (h^t) depends on (z^{t-1}) but on (x^t). A codebase with a different forward-update order may use different-looking indices while implementing the same causal dependency.

Relevant Bellec equations: Methods Eqs. (22), (23), and the paragraph immediately following them on physical PDF page 12.

### 2.4 Optional reset-aware eligibility

The supplementary material explicitly says the main derivation does not differentiate through the reset. If the subtractive reset is included in the local derivative,

$$
\frac{\partial v_j^{t+1}}{\partial v_j^t}
=
\alpha-v_{\mathrm{th}}\psi_j^t,
$$

and the recurrent eligibility vector becomes

$$
\epsilon_{ji}^{t+1}
=
(\alpha-v_{\mathrm{th}}\psi_j^t)\epsilon_{ji}^{t}+z_i^t.
$$

This equation follows the supplement's preceding derivative and correct synaptic dependency, but differs from apparent typographical errors in its displayed recurrence: the published display uses $\alpha-\beta\psi_j^t$ even though $\beta$ is the ALIF adaptation strength, and it shows the postsynaptic $z_j^t$ where the presynaptic $z_i^t$ is required. The main-text LIF derivation, the supplement's own prose, and direct differentiation of the state equation support the corrected form above.

The resulting eligibility is no longer separable into a postsynaptic factor and a presynaptic-only trace. Bellec reports no improvement from this more complex reset-aware version on the evaluated phoneme-recognition and difficult temporal-credit-assignment tasks, and therefore uses the stop-reset simplification in the main derivation.

This distinction matters when auditing code:

- `detach`/stop-gradient on reset is consistent with Bellec's default LIF rule.
- Differentiating the reset requires the $-v_{\mathrm{th}}\psi_j^t$ term in the eligibility recursion.
- A hard multiplicative reset, $v\leftarrow \tilde v(1-z)$, has a different reset derivative and is not literally Bellec's displayed subtractive-reset neuron.

Source: Bellec Supplementary Note 1, “Eligibility traces for LSNNs with membrane potential reset,” supplementary pages 13–14 (physical PDF pages 28–29).

### 2.5 Leaky readout and supervised loss

Bellec's readout is

$$
y_k^t
=
\kappa y_k^{t-1}
+\sum_jW_{kj}^{\mathrm{out}}z_j^t+b_k^{\mathrm{out}}.
$$

For per-time-step mean squared error,

$$
E=\frac12\sum_{t,k}(y_k^t-y_k^{*,t})^2,
\qquad
\delta_k^t=y_k^t-y_k^{*,t},
$$

the direct derivative of total loss with respect to a spike contains future output errors:

$$
\frac{\partial E}{\partial z_j^t}
=
\sum_k\sum_{t'\ge t}
W_{kj}^{\mathrm{out}}\delta_k^{t'}\kappa^{t'-t}.
$$

Bellec exchanges the order of the time sums so that this can be accumulated online. Define the output-filtered synaptic eligibility

$$
\bar e_{ji}^t
=
\kappa\bar e_{ji}^{t-1}+e_{ji}^t
=
\mathcal F_\kappa(e_{ji}^t),
$$

and the instantaneous symmetric learning signal

$$
L_j^t
=
\sum_kW_{kj}^{\mathrm{out}}\delta_k^t.
$$

The online e-prop gradient estimate is then

$$
\widehat{\nabla}_{W_{ji}}E
=
\sum_t L_j^t\bar e_{ji}^t.
$$

For the three sets of weights, the explicit recursions are

$$
p_i^{\mathrm{rec},t}
=\alpha p_i^{\mathrm{rec},t-1}+z_i^{t-1},
\qquad
e_{ji}^{\mathrm{rec},t}=\psi_j^t p_i^{\mathrm{rec},t},
$$

$$
\bar e_{ji}^{\mathrm{rec},t}
=\kappa\bar e_{ji}^{\mathrm{rec},t-1}
+\psi_j^t p_i^{\mathrm{rec},t},
$$

$$
p_i^{\mathrm{in},t}
=\alpha p_i^{\mathrm{in},t-1}+x_i^t,
\qquad
\bar e_{ji}^{\mathrm{in},t}
=\kappa\bar e_{ji}^{\mathrm{in},t-1}
+\psi_j^t p_i^{\mathrm{in},t},
$$

$$
\bar z_j^t=\kappa\bar z_j^{t-1}+z_j^t.
$$

Therefore

$$
\widehat{\nabla}_{W_{ji}^{\mathrm{rec}}}E
=\sum_tL_j^t\bar e_{ji}^{\mathrm{rec},t},
$$

$$
\widehat{\nabla}_{W_{ji}^{\mathrm{in}}}E
=\sum_tL_j^t\bar e_{ji}^{\mathrm{in},t},
$$

$$
\nabla_{W_{kj}^{\mathrm{out}}}E
=\sum_t\delta_k^t\bar z_j^t.
$$

The crucial storage item is $\bar e_{ji}$: after filtering $e_{ji}=\psi_jp_i$ by $\kappa$, it cannot in general be reconstructed from one current postsynaptic value and one presynaptic trace. It is therefore a per-synapse quantity.

For classification with $\pi^t=\operatorname{softmax}(y^t)$ and cross entropy, replace $\delta_k^t$ by

$$
\delta_k^t=\pi_k^t-\pi_k^{*,t}.
$$

The remaining recursions are unchanged.

Sources: Bellec Methods Eqs. (11), (28), and (29); Supplementary Note 3 Eqs. (13)–(20), supplementary pages 17–18 (physical PDF pages 32–33).

### 2.6 Bellec LIF implementation recipe

For each time step $t$, under the state timing used above:

```text
1. Run the forward LIF and leaky-output updates.
2. psi[j]       = pseudo_derivative(v[j] - threshold)
3. p_in[i]      = alpha * p_in[i]  + x[t, i]
4. p_rec[i]     = alpha * p_rec[i] + z[t-1, i]
5. e_in[j,i]    = psi[j] * p_in[i]
6. e_rec[j,i]   = psi[j] * p_rec[i]
7. bar_e_in     = kappa * bar_e_in  + e_in
8. bar_e_rec    = kappa * bar_e_rec + e_rec
9. bar_z[j]     = kappa * bar_z[j]  + z[t, j]
10. delta[k]    = prediction[k] - target[k]
11. L[j]        = sum_k W_out[k,j] * delta[k]
12. grad_W_in  += outer(L, bar_e_in by postsynaptic row)
13. grad_W_rec += outer(L, bar_e_rec by postsynaptic row)
14. grad_W_out += outer(delta, bar_z)
```

More explicitly, steps 12–13 are elementwise $L_j\bar e_{ji}$, not an additional matrix multiplication over $j$.

## 3. Frenkel–Indiveri: modified e-prop in ReckOn

### 3.1 Equations printed in Figure 29.4.3

Frenkel Figure 29.4.3 writes target-minus-prediction as the update direction. Let

$$
d_k^t=y_k^{*,t}-y_k^t.
$$

The three updates are

$$
\Delta W_{kj}^{\mathrm{out},t}
\propto
d_k^t
\sum_{t'\le t}\kappa^{t-t'}z_j^{t'},
$$

$$
\Delta W_{ji}^{\mathrm{rec},t}
\propto
\left(\sum_kW_{kj}^{\mathrm{out}}d_k^t\right)
h_j^t
\sum_{t'\le t}\alpha^{t-t'}z_i^{t'},
$$

$$
\Delta W_{ji}^{\mathrm{in},t}
\propto
\left(\sum_kW_{kj}^{\mathrm{out}}d_k^t\right)
h_j^t
\sum_{t'\le t}\alpha^{t-t'}x_i^{t'}.
$$

Here Frenkel labels the three factors as:

1. learning signal (LS): $\sum_kW_{kj}^{\mathrm{out}}d_k^t$;
2. straight-through estimator (STE): $h_j^t$;
3. eligibility trace (ET): a low-pass-filtered presynaptic activity stream.

This three-factor description applies literally to the input and recurrent updates. The displayed output-weight update has no spiking STE: its postsynaptic term is the output error $d_k^t$, and its presynaptic term is the $\kappa$-filtered hidden spike trace. The paper's prose groups all three weight types under LS/STE/ET locality, but Figure 29.4.3 gives the more precise distinction.

The paper does not give a closed analytic formula for $h_j^t$. It says the chip implements the STE with a programmable five-segment, 5-bit signed lookup-table function. Substituting a triangular or fast-sigmoid surrogate is an implementation choice, not an equation stated by Frenkel.

The displayed recurrent sum includes $z_i^t$. An implementation whose current membrane state is driven by the previous recurrent spike should use the corresponding causal previous-spike indexing. Compare dependencies in the forward equation, not the symbol $t$ in isolation.

Primary source: Frenkel Figure 29.4.3 (physical PDF page 2, digest page 469), supported by the surrounding text on physical PDF page 1.

### 3.2 Recurrence form suitable for code

Using prediction-minus-target gradients instead of Frenkel's update-direction sign, define

$$
\delta_k^t=y_k^t-y_k^{*,t},
\qquad
L_j^t=\sum_kW_{kj}^{\mathrm{out}}\delta_k^t.
$$

Maintain only presynaptic/neuron traces:

$$
p_i^{\mathrm{in},t}=\alpha p_i^{\mathrm{in},t-1}+x_i^t,
$$

$$
p_i^{\mathrm{rec},t}=\alpha p_i^{\mathrm{rec},t-1}+z_i^{\text{causal}(t)},
$$

$$
q_j^t=\kappa q_j^{t-1}+z_j^t.
$$

Then the per-time-step gradient contributions are

$$
g_{ji}^{\mathrm{in},t}=L_j^t h_j^t p_i^{\mathrm{in},t},
$$

$$
g_{ji}^{\mathrm{rec},t}=L_j^t h_j^t p_i^{\mathrm{rec},t},
$$

$$
g_{kj}^{\mathrm{out},t}=\delta_k^t q_j^t.
$$

Accumulate these contributions over the desired update interval, or apply the corresponding updates per time step. ReckOn applies updates per time step and uses stochastic updates of physical 8-bit weights. This stochastic integer-weight update/rounding scheme is a hardware choice separate from the modified e-prop algebra; it should not be conflated with deterministic forward-pass weight quantization through a floating-point master weight.

### 3.3 Frenkel implementation recipe

```text
1. Run the forward LIF and leaky-output updates.
2. h[j]          = STE_LUT(v[j], threshold, ...)
3. p_in[i]       = alpha * p_in[i]  + x[t, i]
4. p_rec[i]      = alpha * p_rec[i] + causal_recurrent_spike[i]
5. q_out[j]      = kappa * q_out[j] + z[t, j]
6. delta[k]      = prediction[k] - target[k]
7. L[j]          = sum_k W_out[k,j] * delta[k]
8. post[j]       = L[j] * h[j]
9. grad_W_in    += outer(post, p_in)
10. grad_W_rec  += outer(post, p_rec)
11. grad_W_out  += outer(delta, q_out)
```

No synapse-indexed hidden eligibility state is needed in this rule.

The neuron-scaling claim applies to the **eligibility state**, not to total network storage or all update computation. The weights remain synapse-indexed, and a dense outer-product update still touches a number of weights proportional to the number of synapses. ReckOn additionally exploits zero ET and STE values to skip many of those updates.

## 4. Exact difference between Bellec LIF e-prop and Frenkel modified e-prop

For the same learning signal, surrogate/STE, and presynaptic trace $p_i^t$, the hidden-weight factor is:

### Bellec

$$
\bar e_{ji}^t
=
\kappa\bar e_{ji}^{t-1}
+h_j^t p_i^t,
$$

$$
g_{ji}^t=L_j^t\bar e_{ji}^t.
$$

Expanded,

$$
\bar e_{ji}^t
=
\sum_{s\le t}\kappa^{t-s}h_j^s p_i^s.
$$

Bellec preserves the historical pairing: the postsynaptic surrogate at time $s$ remains paired with the presynaptic trace as it was at the same time $s$.

### Frenkel

$$
g_{ji}^t=L_j^t h_j^t p_i^t.
$$

There is no $\kappa$-recursion over the hidden synaptic product. Only current $h_j^t$ and the current presynaptic trace are used.

Consequences:

| Property | Bellec supervised LIF | Frenkel modified e-prop |
|---|---|---|
| Presynaptic LIF trace | $p_i^t=\mathcal F_\alpha(\text{activity})$ | Same form |
| Postsynaptic spike derivative | $\psi_j^t$ | $h_j^t$, programmable STE LUT |
| Additional output-leak filter for input/recurrent weights | $\bar e_{ji}^t=\mathcal F_\kappa(\psi_j^tp_i^t)$ | Omitted |
| Hidden eligibility storage | Per synapse $j,i$ | Per presynaptic neuron $i$, plus current postsynaptic factors |
| Output-weight presynaptic trace | $\mathcal F_\kappa(z_j^t)$ | Retained: $\mathcal F_\kappa(z_j^t)$ |
| Exact equivalence when $\kappa=0$ | $\bar e_{ji}^t=\psi_j^tp_i^t$ | Same, up to timing/STE choices |
| Reset derivative in printed rule | Bellec default omits it; supplement gives optional correction | No reset-derivative recursion is shown |
| Typical memory extension highlighted by authors | ALIF slow adaptation | Frenkel lengthens the LIF leak time constant to task timescale |

The output-weight formula is not the differentiator: both approaches use a $\kappa$-filtered hidden spike trace for a leaky readout.

### 4.1 Separate neuron-model change in ReckOn

Frenkel also changes the model choice. Bellec notes that an ordinary LIF with a biologically typical $\tau_m\approx20\,\mathrm{ms}$ cannot handle long temporal dependencies as well as ALIF neurons. ReckOn avoids ALIF's multi-timescale synaptic eligibility state by using LIF neurons whose leakage time constant is lengthened to match the task; Figure 29.4.3 compares a $2\,\mathrm{s}$-leak LIF model on its delayed-navigation example.

This is distinct from the algebraic simplification:

- larger $\tau_m$ means $\alpha$ is closer to one, so the presynaptic trace lasts longer;
- dropping $\mathcal F_\kappa(h_j^tp_i^t)$ removes a per-synapse state.

A code implementation can make one change without the other.

In ReckOn's modified rule, the hidden presynaptic trace uses the LIF membrane
decay $\alpha$, rather than an independently selected trace decay.
Correspondingly, this repository's `frenkel` branch uses `beta_mem_rec`; the
inactive independent `tau_trace` and `tau_trace_out` CLI parameters have been
removed. The output trace uses the readout decay $\kappa$ through `beta_mem`.

## 5. Counter-check checklist for this repository

The current implementation lives primarily in `utils/train_snn.py::grads_batch`, with forward timing/reset behavior in `utils/neuron_models.py`.

### 5.1 Checks common to both modes

- Confirm tensor orientation: repository weights are documented as `[postsynaptic, presynaptic]` for input/recurrent weights and `[output, hidden]` for output weights.
- Confirm the sign: the repository forms prediction-minus-target gradients and lets the optimizer subtract them; do not compare these directly with Frenkel's target-minus-prediction $\Delta W$ sign.
- Confirm recurrent timing against the forward pass. The repository forward recurrence consumes the previously stored `out`; therefore a `z_prev` eligibility update is causally aligned even though Frenkel's compact figure writes a sum through $z_i^t$.
- The forward pass continues membrane/input integration during refractory steps
  while suppressing spikes, and the e-prop surrogate/STE is zeroed using the
  recorded refractory mask.
- Bellec mode uses the named paper-specific factor $0.3/v_{\mathrm{th}}$;
  `gamma=15` remains the BPTT fast-sigmoid scale and Frenkel software-STE
  amplitude.
- Distinguish the membrane leak $\alpha$, output leak $\kappa$, and any separate synaptic-current decay. Bellec's displayed basic LIF derivation has no additional synaptic-current state.

### 5.2 Expected `bellec` mode structure

The implementation should have all of the following:

- presynaptic input and recurrent traces filtered by the recurrent membrane leak;
- current pseudo-derivative multiplied by each presynaptic trace;
- a **second**, output-leak recursion on the resulting synapse-indexed eligibility;
- $W_{\mathrm{out}}^T\delta^t$ as the symmetric learning signal;
- a hidden-spike output trace filtered by the output leak;
- synapse-sized `bar_e_in` and `bar_e_rec` state.
- for multiplicative hard reset, an additional synapse-indexed eligibility
  recursion containing the exact $1-z_j^{t-1}$ stopped-reset factor.

This matches the structure currently visible in the `bellec` branch of `grads_batch`.

### 5.3 Expected `frenkel` mode structure

The implementation should have all of the following:

- only presynaptic input/recurrent traces for hidden-weight updates;
- `post_factor = learning_signal * STE`;
- input/recurrent gradients as outer products of `post_factor` with the corresponding presynaptic traces;
- no output-leak recursion around `STE * presynaptic_trace` for hidden weights;
- a separate output-leak-filtered hidden spike trace for output weights.

This matches the central factorization currently visible in the `frenkel`
branch of `grads_batch` when subtractive reset is used. With multiplicative
hard reset, the implementation deliberately replaces neuron-scaled ETs with
synapse-indexed reset-aware traces and emits a warning.

The implementation evaluates these contributions in a time loop, but accumulates them over the sequence and batch before one Adamax optimizer step. It is therefore ReckOn-inspired algebra, not ReckOn's physical online update at every timestep.

### 5.4 Weight-update and quantization differences

ReckOn updates the stored 8-bit weights stochastically. Small calculated updates can thereby produce probabilistic one-LSB changes without requiring a higher-precision shadow copy of every weight. This property is the basis of the paper's claim that learning adds no weight-memory footprint relative to inference.

The repository's `quantize_weights` path is different:

- optimizer parameters remain floating-point master weights;
- the forward pass deterministically selects the nearest configured weight level;
- continuous gradients are applied to the master weights with Adamax;
- the optimizer step occurs once per batch, not once per simulation timestep.

Thus enabling `quantize_weights` does not reproduce ReckOn's stochastic update scheme or its weight-memory result. There are also two distinct uses of “STE” in this code and paper: Frenkel's $h_j^t$ is an estimator of the **spiking activation derivative**, whereas `STEFunction` in `utils/neuron_models.py` passes gradients through deterministic **weight quantization**.

### 5.5 Model differences that are not resolved by choosing an e-prop mode

When comparing results with either paper, independently audit:

- subtractive versus multiplicative hard reset; both are supported, but only
  subtractive reset retains the canonical simple trace;
- optional synaptic-current filtering in the repository; current e-prop modes
  reject it until a two-state eligibility recursion is implemented;
- exponential versus linear membrane decay; both implemented e-prop modes require multiplicative exponential decay and now reject `linear_decay`;
- refractory masking; the current implementation records and applies it to the
  spike derivative while continuing membrane integration;
- the exact loss and whether it is applied at every time step or only a delayed/final window;
- batch accumulation and optimizer choice;
- deterministic floating-point/Adamax updates versus ReckOn's stochastic 8-bit updates;
- whether feedback uses the live output weights (symmetric e-prop), fixed random broadcast weights, or another feedback matrix.

## 6. Minimal numerical equivalence tests

These tests are implemented in `scripts/tests/test_eprop_algorithm.py` and pass
before comparing full training curves.

### Test A: Frenkel equals Bellec when output leak is disabled

Set $\kappa=0$, use identical $h=\psi$, identical traces, no reset derivative, and identical timing. Then

$$
\bar e_{ji}^t=h_j^tp_i^t,
$$

so Bellec and Frenkel hidden gradients must agree exactly.

### Test B: demonstrate the intended difference when $\kappa\ne0$

Use two time steps with a nonzero (h_j^1p_i^1), zero (h_j^2p_i^2), and nonzero (L_j^2). At time 2:

$$
g_{ji,\mathrm{Bellec}}^2
=L_j^2\kappa h_j^1p_i^1,
\qquad
g_{ji,\mathrm{Frenkel}}^2=0.
$$

This directly tests the filter Frenkel removed.

### Test C: output gradients should still agree

With identical $\delta^t,z^t,\kappa$, both modes should compute

$$
\sum_t\delta_k^t\mathcal F_\kappa(z_j^t)
$$

for the output-weight gradient.

### Test D: recurrent one-step causality

Inject a single recurrent spike at time $t$, perturb one recurrent weight, and verify that eligibility first becomes active at the time that weight can affect the postsynaptic membrane according to the actual forward loop. This catches off-by-one errors without relying on notation differences between the papers.

## 7. Scope cautions

- Bellec's “symmetric e-prop” uses the transpose of the live output weights in the learning signal. Random e-prop replaces this with fixed random broadcast weights; that is a separate approximation.
- Bellec's ALIF eligibility is two-dimensional and includes threshold-adaptation dynamics. It must not be reduced to the LIF equations above if ALIF neurons are introduced.
- Bellec's exact mathematical factorization does not make the online learning signal exact. For spiking neurons, “exact eligibility” is additionally conditional on the substituted pseudo-derivative and does not mean an exact classical derivative of the Heaviside forward model. The eligibility trace and learning-signal approximation should be tested separately.
- Frenkel's paper is a short hardware paper. Figure 29.4.3 is the primary source for the modified equations, and it does not specify every software-level detail (notably the analytic STE shape and all time-index conventions).
