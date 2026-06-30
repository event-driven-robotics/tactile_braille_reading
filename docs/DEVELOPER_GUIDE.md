# Developer Guide

This guide captures operational details for contributors maintaining and extending the RSNN training pipeline.

## Primary Entry Points

- Training script: `scripts/braille_reading_rsnn.py`
- Core model/runtime logic:
  - `utils/neuron_models.py`
  - `utils/snn.py`
  - `utils/train_snn.py`
  - `utils/validate_snn.py`
- Tests:
  - `scripts/tests/test_neuron_computation.py`
  - `scripts/tests/test_parser_logic.py`
  - `scripts/tests/test_network_creation.py`

## Environment and Dependencies

Install project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

## Training and Evaluation Workflow

Default training:

```bash
python scripts/braille_reading_rsnn.py
```

Common quick run:

```bash
python scripts/braille_reading_rsnn.py --letters A B --epochs 10 --nb_hidden 50
```

Resume training from latest checkpoint in a run folder:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id>
```

Inference-only evaluation from a resumed run:

```bash
python scripts/braille_reading_rsnn.py --resume-run-id <run_id> --inference-only
```

## Key CLI Parameters

Learning mode:
- `--eprop` (otherwise BPTT)
- `--eprop_mode {frenkel,bellec}`
- Legacy aliases still accepted: `experimental`, `traditional`

Evaluation/stop behavior:
- `--validation` switches evaluation target from test set to validation split
- `--early_stop_epochs`, `--early_stop_threshold` adaptive early-stop controls

Artifact policy:
- `--save_artifacts_for {all,best,none}`
  - `all`: save per-repetition model/traces
  - `best`: save only globally best repetition model/traces
  - `none`: skip model/traces (metrics still saved)

Trainability controls:
- `--train_rec_ff`
- `--train_rec_rec`
- `--train_out_ff`

Resume/inference:
- `--resume-run-id`
- `--resume-model-name`
- `--inference-only`
- Alias forms also accepted:
  - `--resume-training`, `--resume_training`
  - `--inference_only`

Input/encoding controls:
- `--mechanoreceptor_encoding`
- `--threshold`
- `--time_bin_size`
- `--selected_channels`

## Output Layout

Each run creates timestamped folders:
- `results/<run_id>/`
- `model/<run_id>/`
- `figures/<run_id>/`
- `logs/<run_id>/`

Important artifacts:
- Per-repetition metrics:
  - `results/<run_id>/braille_reading_rsnn_*_rep_*.npz`
- Trace artifacts (when enabled):
  - `results/<run_id>/best_model_traces_*_rep_*.npz`
  - Includes `record_mode` metadata and train/test spike traces.

## Testing

Run full test suite:

```bash
python -m pytest scripts/tests -ra -vv
```

Run smoke subset only:

```bash
python -m pytest scripts/tests -m smoke -q
```

One-command wrapper:

```bash
python scripts/tests/run_all_tests.py
```

## CI Health Check

Workflow: `.github/workflows/health-check.yml`

Jobs:
- `smoke`: fast marker-based subset
- `full`: complete test suite

Both run on push and pull requests. Enforce strictness via branch protection if desired.
