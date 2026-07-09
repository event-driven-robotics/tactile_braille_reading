# Braille Letter Reading with RSNNs

You can find the Frontiers publication "Braille letter reading: A benchmark for spatio-temporal pattern recognition on neuromorphic hardware" [here](https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full).

Primary entry point:

```bash
python scripts/braille_reading_rsnn.py
```

Original publication:
- Frontiers article: https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full

Additional maintainer notes:
- [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)

## Braille Context

Braille is a tactile writing system based on dot-cell patterns. This project
uses event-based tactile sensor recordings to classify temporal braille patterns
with spiking neural networks.

## Data

Dataset records:
- Original dataset: [10.5281/zenodo.7050094](https://zenodo.org/records/7050094)
- Source dataset: [10.5281/zenodo.13841759](https://zenodo.org/records/13841759)

The training script expects encoded data under `./data/100Hz/` by default:
- `mechanoreceptor_encoded.pkl` for `--encoding_type mechanoreceptor`
- `data_braille_letters_100Hz_th<threshold>.pkl` for `--encoding_type sigma-delta`

Examples of sigma-delta files are:
- `data_braille_letters_100Hz_th1.pkl`
- `data_braille_letters_100Hz_th2.pkl`
- `data_braille_letters_100Hz_th5.pkl`
- `data_braille_letters_100Hz_th10.pkl`

Use `--input_data_path` if your data lives somewhere else. The path should end
with a trailing slash, for example `--input_data_path ./data/100Hz/`.

## Encoding Generation

Use `scripts/event_transform.py` to generate encoded datasets from raw tactile
trials before training or standalone analysis.

Supported encoding modes:
- `mechanoreceptor`: FA-I and SA-II spike streams.
- `sigma_delta`: ON and OFF spike streams.
- `neuron_model`: spike streams from one selected neuron model.

Show all options:

```bash
python scripts/event_transform.py --help
```

Run a guided prompt that asks only the follow-up options relevant to the
selected encoding:

```bash
python scripts/event_transform.py --interactive
```

Generate mechanoreceptor-encoded data:

```bash
python scripts/event_transform.py \
  --encoding-type mechanoreceptor
```

Generate sigma-delta-encoded data:

```bash
python scripts/event_transform.py \
  --encoding-type sigma_delta
```

Generate neuron-model-encoded data (with configurable upsampling):

```bash
python scripts/event_transform.py \
  --encoding-type neuron_model \
  --neuron-model MN_neuron \
  --upsample-strategy linear \
  --upsample-dt-s 0.001
```

For neuron-model encoding, `--upsample-dt-s` is also passed to the neuron model
as its simulation `dt` in seconds, so data availability and neuron dynamics use
the same timestep. `IZ_neuron` converts that value to milliseconds internally.

Common encoding CLI options:
- `--data-path`: input folder for source tactile files (default `data/100Hz`).
- `--data-files`: one or more input files relative to `--data-path`.
- `--output-path`: explicit output pickle path override.

Default output filenames (inside `--data-path`):
- `mechanoreceptor_encoded.pkl`
- `sigma_delta_encoded.pkl`
- `<NEURON_MODEL>_encoded.pkl` for neuron-model mode, for example
  `MN_neuron_encoded.pkl`

## Encoded Data Analysis

Use `scripts/analyse_encoding.py` to visualize all encoded files created by
`event_transform.py`.

Run a guided prompt to select data path, file pattern, and plotting options:

```bash
python scripts/analyse_encoding.py --interactive
```

Analyze all encoded files in `data/100Hz`:

```bash
python scripts/analyse_encoding.py
```

Customize input/output and sampling:

```bash
python scripts/analyse_encoding.py \
  --data-path data/100Hz \
  --pattern "*_encoded*.pkl" \
  --samples-per-letter 5 \
  --output-root figures/encoding_analysis
```

What the analysis script does:
- Scans encoded pickle files by glob pattern.
- Infers schema from keys (`fa/sa`, `ON/OFF`, or `spikes`).
- Samples trials per letter and generates per-sample plots.
- Writes figures to `figures/encoding_analysis/<encoded_file_stem>/`.

When using the project virtual environment directly, run:

```bash
/home/username/.virtualenvs/pytorch/bin/python scripts/event_transform.py --help
/home/username/.virtualenvs/pytorch/bin/python scripts/analyse_encoding.py --help
/home/username/.virtualenvs/pytorch/bin/python scripts/event_transform.py --interactive
/home/username/.virtualenvs/pytorch/bin/python scripts/analyse_encoding.py --interactive
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your system does not provide a `python` executable, use `python3` in the
commands below.

For CUDA acceleration, install the PyTorch build that matches your CUDA runtime
before or after installing the remaining requirements.

## Quick Start

Run the default experiment with all letters, mechanoreceptor encoding, BPTT, one
repetition, and 450 recurrent hidden neurons:

```bash
python scripts/braille_reading_rsnn.py
```

Run a short binary classification experiment:

```bash
python scripts/braille_reading_rsnn.py \
  --letters A B \
  --epochs 10 \
  --nb_hidden 50
```

Run e-prop with a validation split:

```bash
python scripts/braille_reading_rsnn.py \
  --eprop \
  --eprop_mode frenkel \
  --validation \
  --letters A B C \
  --epochs 50
```

Run sigma-delta encoding at threshold 5:

```bash
python scripts/braille_reading_rsnn.py \
  --encoding_type sigma-delta \
  --threshold 5
```

Use selected taxels only:

```bash
python scripts/braille_reading_rsnn.py \
  --selected_channels 0 1 2 5 8
```

Disable CUDA explicitly:

```bash
python scripts/braille_reading_rsnn.py --no-cuda
```

See the complete parser help at any time:

```bash
python scripts/braille_reading_rsnn.py --help
```

## Main CLI Reference

The script uses `argparse` and keeps most experiment settings available from the
command line. Defaults below reflect `scripts/braille_reading_rsnn.py`.

### Paths

| Argument | Default | Description |
| --- | --- | --- |
| `--input_data_path` | `./data/100Hz/` | Folder containing encoded input pickle files. |
| `--fig_path` | `./figures` | Root folder for generated figures. |
| `--model_path` | `./model` | Root folder for saved model files. |
| `--results_path` | `./results` | Root folder for metrics, traces, and parameter JSON. |
| `--log_path` | `./logs` | Root folder for run logs. |

### Run Control

| Argument | Default | Description |
| --- | --- | --- |
| `--cuda`, `--no-cuda` | CUDA enabled | Use GPU if available, or force CPU with `--no-cuda`. |
| `--debug` | off | Enables detailed diagnostics and sets `--log_level DEBUG`. |
| `--log_level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Console and file logging verbosity. |
| `--seed <int>` | unset | Fix NumPy/PyTorch/random hash seeds for reproducibility. |
| `--dtype {float16,float32,float64}` | `float64` | Torch dtype used for computations. |
| `--epochs <int>` | `50` | Number of training epochs. |
| `--repetitions <int>` | `1` | Number of independent repeated training runs. |
| `--batch_size <int>` | `128` | Training and evaluation batch size. |
| `--learning_rate <float>` | `0.0001` | Optimizer learning rate. |
| `--validation` | off | Create a validation split from training data. |
| `--early_stop_epochs <int>` | `0` | Initial epoch window for adaptive early stopping; `0` disables it. |
| `--early_stop_threshold <float>` | `5.0` | Percentage points above chance required during the early-stop window. |

Without `--validation`, data is split 80% train and 20% test. With
`--validation`, the split is approximately 70% train, 20% test, and 10%
validation.

### Data and Encoding

| Argument | Default | Description |
| --- | --- | --- |
| `--letters <labels...>` | all 26 letters plus `Space` | Restrict classification to a subset, for example `--letters A B Space`. |
| `--encoding_type {mechanoreceptor,sigma-delta}` | `mechanoreceptor` | Choose FA-I/SA-II mechanoreceptor spikes or sigma-delta ON/OFF events. |
| `--threshold {1,2,5,10}` | `2` | Sigma-delta event threshold. Ignored for mechanoreceptor encoding. |
| `--time_bin_size <int>` | `1` | Time-bin size in milliseconds. |
| `--selected_channels <ints...>` | `0` through `11` | Taxel indices to keep. Each taxel contributes two input channels. |

Mechanoreceptor encoding uses FA and SA channels. Sigma-delta encoding uses ON
and OFF channels. In both cases, `params["nb_inputs"]` is derived from selected
taxels as `2 * len(selected_channels)`.

### Network Architecture

| Argument | Default | Description |
| --- | --- | --- |
| `--nb_hidden <int>` | `450` | Number of recurrent hidden neurons. |
| `--nb_input_copies <int>` | `1` | Number of copies for each input channel. |

### Learning Algorithm

| Argument | Default | Description |
| --- | --- | --- |
| `--eprop` | off | Use e-prop instead of BPTT. |
| `--eprop_mode {frenkel,bellec,experimental,traditional}` | `frenkel` | E-prop variant. Legacy aliases are `experimental=frenkel` and `traditional=bellec`. |
| `--gamma <float>` | `15.0` | Surrogate gradient scale factor. |

Default training uses BPTT. `--eprop` switches to the online e-prop path.

### Neuron Dynamics

| Argument | Default | Description |
| --- | --- | --- |
| `--tau_mem <float>` | `0.06` | Membrane time constant in seconds. |
| `--tau_mem_rec <float>` | `0.06` | Recurrent membrane time constant in seconds. |
| `--tau_trace <float>` | `0.14` | Hidden eligibility trace time constant in seconds. |
| `--tau_trace_out <float>` | `0.14` | Output trace time constant in seconds. |
| `--tau_ratio <float>` | `10` | Ratio used for synaptic time constant calculation. |
| `--ref_per_ms <float>` | `3.0` | Refractory period in milliseconds. |
| `--ref_per_timesteps <int>` | unset | Deprecated timestep-based refractory override. |
| `--lower_bound <float|None>` | `None` | Optional membrane-potential lower clamp. |
| `--threshold_noise_std <float>` | `0.0` | Random spike-threshold noise standard deviation. |
| `--spike_threshold <float>` | `1.0` | Neuron spike threshold. |
| `--soft_reset` | off | Subtract threshold after a spike instead of hard reset to zero. |
| `--synapse` | off | Enable synaptic current dynamics. |
| `--linear_decay` | off | Use linear rather than exponential decay. |

`--ref_per_ms` is the preferred refractory-period interface. If
`--ref_per_timesteps` is explicitly set, it takes priority and the effective
millisecond value is derived from `--time_bin_size`.

### Weights, Trainability, and Regularization

| Argument | Default | Description |
| --- | --- | --- |
| `--fwd_weight_scale <float>` | `1.0` | Forward weight initialization scale. |
| `--weight_scale_factor <float>` | `0.02` | Recurrent weight scale factor. |
| `--train_rec_ff <bool>` | `true` | Train recurrent-layer input weights. |
| `--train_rec_rec <bool>` | `true` | Train recurrent-layer recurrent weights. |
| `--train_out_ff <bool>` | `true` | Train readout-layer feed-forward weights. |
| `--reg_spikes <float>` | `0.0015` | L1 spike regularization coefficient. |
| `--reg_neurons <float>` | `0.001` | L2 neuron regularization coefficient. |
| `--quantize_weights` | off | Use the quantized-weight forward path. |

Boolean trainability arguments accept values like `true`, `false`, `1`, `0`,
`yes`, and `no`:

```bash
python scripts/braille_reading_rsnn.py \
  --train_rec_rec false \
  --train_out_ff true
```

### Prediction and Artifacts

| Argument | Default | Description |
| --- | --- | --- |
| `--random_tie_breaking` | off | Randomize class selection when predictions tie. |
| `--save_artifacts_for {all,best,none}` | `best` | Control saving of model-weight and trace artifacts. |

Artifact modes:
- `all`: save model weights and traces for every repetition.
- `best`: save model weights and traces only for the globally best repetition.
- `none`: skip model-weight and trace artifacts; metrics are still saved.

## Resume and Inference

Resume from the newest `best_model_*.pt` checkpoint in a previous run:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id>
```

Resume from a specific checkpoint:

```bash
python scripts/braille_reading_rsnn.py \
  --resume-run-id <run_id> \
  --resume-model-name best_model_50_neurons_A_B_rep_1.pt
```

Run evaluation and plotting without further training:

```bash
python scripts/braille_reading_rsnn.py \
  --resume-run-id <run_id> \
  --inference-only
```

Resume behavior:
- Parameters are loaded from `results/<run_id>/experiment_parameters.json`.
- Checkpoints are loaded from `model/<run_id>/`.
- Explicit CLI arguments override loaded parameters for the current invocation.
- `--resume-model-name` selects the checkpoint file only; it does not change the
  experiment hyperparameters.

Accepted aliases:
- `--resume-training` and `--resume_training` for `--resume-run-id`
- `--inference_only`, including forms like `--inference_only=true`

## Outputs

Each invocation creates a timestamped run folder under the configured roots:

```text
results/<run_id>/
model/<run_id>/
figures/<run_id>/
logs/<run_id>/
```

Important files include:
- `results/<run_id>/experiment_parameters.json`
- `logs/<run_id>/training_log_<run_id>.txt`
- `results/<run_id>/braille_reading_rsnn_*_rep_*.npz`
- `model/<run_id>/best_model_*_rep_*.pt`, when artifact saving is enabled
- `results/<run_id>/best_model_weights_*_rep_*.npz`, when artifact saving is enabled
- `results/<run_id>/best_model_traces_*_rep_*.npz`, when trace saving is enabled
- `figures/<run_id>/*confusion_matrix*.pdf`
- `figures/<run_id>/*network_activity*.pdf`

The metrics `.npz` files contain training/test or validation performance arrays.
The trace `.npz` files include `record_mode` metadata and spike/membrane traces
for later analysis.

## Analysis Utilities

Summarize the best experiments by hidden-neuron count:

```bash
python scripts/find_best_experiments.py \
  --results-root ./results \
  --metric best_test \
  --top-k 3
```

Plot performance scaling from an exploration directory:

```bash
python scripts/analyze_exploration_results.py \
  --exploration-dir ./results/<exploration_run> \
  --output-dir ./figures \
  --title "Hidden-neuron sweep"
```

Analyze circuit-level spike exports:

```bash
python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id <run_id> \
  --rep 1 \
  --write-report
```

Visualize mechanoreceptor-encoded samples:

```bash
python scripts/analyse_mechanoreceptor_encoding.py
```

That script currently has no CLI arguments; edit `data_path`, `file_name`, or
`nb_of_leters_to_visualize` in the script if needed.

## Testing

Run the full test suite:

```bash
python -m pytest scripts/tests
```

Run the smoke subset:

```bash
python -m pytest scripts/tests -m smoke -q
```

Or use the wrapper:

```bash
python scripts/tests/run_all_tests.py
```

The GitHub Actions health check lives at `.github/workflows/health-check.yml`
and runs smoke and full test jobs.

## Citation

### Core project publication

Müller-Cleve, S. F., Fra, V., Khacef, L., Pequeño-Zurro, A., Klepatsch, D., Forno, E., Ivanovich, D. G., Rastogi, S., Urgese, G., Zenke, F., & Bartolozzi, C. (2022).
*Braille letter reading: A benchmark for spatio-temporal pattern recognition on neuromorphic hardware*.
Frontiers in Neuroscience, 16, 951164.
DOI: [10.3389/fnins.2022.951164](https://doi.org/10.3389/fnins.2022.951164)

Optional BibTeX:

```bibtex
@article{mullercleve2022braille,
    author = {M{\"{u}}ller-Cleve, Simon F. and Fra, Vittorio and Khacef, Lyes and Peque{\~{n}}o-Zurro, Alejandro and Klepatsch, Daniel and Forno, Evelina and Ivanovich, Diego G. and Rastogi, Shavika and Urgese, Gianvito and Zenke, Friedemann and Bartolozzi, Chiara},
    doi = {10.3389/fnins.2022.951164},
    eprint = {2205.15864},
    issn = {1662-453X},
    journal = {Frontiers in Neuroscience},
    keywords = {Braille reading,benchmarking,event-based encoding,neuromorphic hardware,spatio-temporal pattern recognition,spiking neural networks,tactile sensing},
    month = {nov},
    title = {{Braille letter reading: A benchmark for spatio-temporal pattern recognition on neuromorphic hardware}},
    url = {https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full},
    volume = {16},
    year = {2022}
}
```

### E-prop-related references

The current implementation supports two e-prop modes, `frenkel` and `bellec`.

- Bellec, Scherr, Subramoney, et al., *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications (2020).
  DOI: 10.1038/s41467-020-17236-y
- Frenkel and Indiveri, *ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales*, ISSCC (2022).
  DOI: 10.1109/ISSCC42614.2022.9731734
