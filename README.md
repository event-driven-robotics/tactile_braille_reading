# Braille Letter Reading (RSNN)

This repository contains spiking neural network pipelines for tactile braille letter classification.

Primary training entry point:
- `scripts/braille_reading_rsnn.py`

Original publication context:
- Frontiers article: https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full

## Documentation

- User-facing overview: this README
- Developer/maintenance guide: [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)

## Braille Context

Braille is a tactile writing system based on dot-cell patterns. This project focuses on recognizing these temporal tactile patterns from sensor recordings.

## Dataset

- Original dataset: [10.5281/zenodo.7050094](https://zenodo.org/records/7050094)
- Source dataset: [10.5281/zenodo.13841759](https://zenodo.org/records/13841759)
- Typical input path expected by the training script:
	- `./data/100Hz/`
- Script chooses encoded files based on `--mechanoreceptor_encoding` and threshold/time-bin settings.

## Installation

Create/activate a Python environment, then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional for running tests directly via pytest:

```bash
pip install pytest
```

## Training (Current CLI)

Run default training:

```bash
python scripts/braille_reading_rsnn.py
```

Quick binary-class run:

```bash
python scripts/braille_reading_rsnn.py --letters A B --epochs 10 --nb_hidden 50
```

Useful options (see script help for full list):
- `--eprop` enable e-prop training (otherwise BPTT)
- `--eprop_mode {frenkel,bellec}`
- `--eprop_mode {experimental,traditional}` legacy aliases supported
- `--validation` evaluate on validation split (creates also a validation set, useful for HPO)
- `--early_stop_epochs`, `--early_stop_threshold` adaptive early-stop controls
- `--save_artifacts_for {all,best,none}` controls saved model/traces
- `--train_rec_ff`, `--train_rec_rec`, `--train_out_ff` weight-trainability controls
- `--resume-run-id`, `--resume-model-name` resume from previous run
- `--inference-only` evaluate resumed model without further training
- `--threshold`, `--time_bin_size`, `--mechanoreceptor_encoding` input encoding controls
- `--quantize_weights` run quantized-weight forward path

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

The current implementation supports two e-prop modes (`frenkel`, `bellec`).

- Bellec, Scherr, Subramoney, et al., *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications (2020).
	- DOI: 10.1038/s41467-020-17236-y
- Frenkel and Indiveri, *ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales*, ISSCC (2022).
	- DOI: 10.1109/ISSCC42614.2022.9731734

## Output Structure

Each run creates a timestamped run-id folder in:
- `results/<run_id>/`
- `model/<run_id>/`
- `figures/<run_id>/`
- `logs/<run_id>/`

Behavior of `--save_artifacts_for`:
- `all`: save model/traces for every repetition
- `best`: save model/traces only for globally best repetition
- `none`: save no model/traces (metrics files still saved)

Per-repetition metrics files:
- `results/<run_id>/braille_reading_rsnn_*_rep_*.npz`

Trace files (when enabled):
- `results/<run_id>/best_model_traces_*_rep_*.npz`
- Includes `record_mode` metadata and train/test spike traces

## Resume and Inference-Only

Resume training from newest model in a run:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id>
```

Resume from a specific checkpoint:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id> --resume-model-name <best_model_file.pt>
```

Equivalent resume/inference flag aliases are also accepted:
- `--resume-training`,
- `--inference_only`

Inference-only evaluation from resumed model:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id> --inference-only
```

## Testing

### One-command test runner

```bash
python scripts/tests/run_all_tests.py
```

### Pytest directly

```bash
python -m pytest scripts/tests
```

Quick smoke subset:

```bash
python -m pytest scripts/tests -m smoke -q
```

Detailed run:

```bash
python -m pytest scripts/tests -ra -vv
```

### Test map

- `scripts/tests/test_neuron_computation.py`
	- deterministic neuron/layer dynamics and recurrence masking
- `scripts/tests/test_parser_logic.py`
	- CLI parse logic, alias normalization, resume path resolution
- `scripts/tests/test_network_creation.py`
	- end-to-end layer construction and forward-pass shape/probability invariants

## CI Health Check (GitHub Actions)

Workflow file:
- `.github/workflows/health-check.yml`

Jobs:
- `smoke`: fast `-m smoke` subset
- `full`: complete test suite

By default this is informational unless enforced in branch protection.

## Practical Tip

If `pytest` is not on your shell `PATH`, always use:

```bash
python -m pytest ...
```

This guarantees tests run in the currently selected Python environment.
