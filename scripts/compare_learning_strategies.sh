#!/usr/bin/env bash

# Compare BPTT, Bellec e-prop, and Frenkel e-prop on Braille A vs B.
#
# Fixed experiment settings:
#   - letters: A B
#   - hidden neurons: 50
#   - maximum epochs: 25
#   - early stopping: require 55% training accuracy by epoch 5
#   - seed: 42
#   - debug mode: enabled
#   - validation split: enabled (used to rank grid configurations)
#   - batch size: 128
#
# Performance-sensitive grid:
#   - learning rate: 1e-5, 5e-5, 1e-4
#   - surrogate-gradient gamma: 5, 15, 30 (BPTT and Frenkel only)
#
# Learning rate is tuned separately because BPTT uses a batch-mean loss,
# whereas the e-prop implementations sum their manual gradients over the batch
# and selected error timesteps.
#
# Usage:
#   ./scripts/compare_learning_strategies.sh
#   ./scripts/compare_learning_strategies.sh --dry-run
#   ./scripts/compare_learning_strategies.sh --no-cuda
#   ./scripts/compare_learning_strategies.sh --python /path/to/python
#
# Optional environment overrides for the search grid:
#   LEARNING_RATES="0.00001 0.00005 0.0001"
#   GAMMAS="5 15 30"

set -Euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly TRAINING_SCRIPT="${SCRIPT_DIR}/braille_reading_rsnn.py"

readonly EPOCHS=25
readonly EARLY_STOP_EPOCHS=5
readonly EARLY_STOP_THRESHOLD=5
readonly NB_HIDDEN=50
readonly SEED=42
readonly BATCH_SIZE=128
readonly LETTERS=(A B C D)

read -r -a LEARNING_RATE_GRID <<< "${LEARNING_RATES:-0.00001 0.00005 0.0001}"
read -r -a GAMMA_GRID <<< "${GAMMAS:-5 15 30}"

DRY_RUN=false
CUDA_ARG=""
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
    sed -n '2,/^$/s/^# \{0,1\}//p' "$0"
}

while (($# > 0)); do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-cuda)
            CUDA_ARG="--no-cuda"
            shift
            ;;
        --python)
            if (($# < 2)); then
                echo "Error: --python requires a path." >&2
                exit 2
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    elif [[ -x "/home/smullercleve/.virtualenvs/pytorch/bin/python" ]]; then
        PYTHON_BIN="/home/smullercleve/.virtualenvs/pytorch/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Error: no executable Python interpreter found." >&2
    echo "Use --python /path/to/python or set PYTHON_BIN." >&2
    exit 1
fi

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "Error: training script not found: $TRAINING_SCRIPT" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import numpy, torch' 2>/dev/null; then
    echo "Error: $PYTHON_BIN cannot import both NumPy and PyTorch." >&2
    echo "Use --python /path/to/project/python or activate the project environment." >&2
    exit 1
fi

if ((${#LEARNING_RATE_GRID[@]} == 0 || ${#GAMMA_GRID[@]} == 0)); then
    echo "Error: learning-rate and gamma grids must not be empty." >&2
    exit 1
fi

readonly COMPARISON_ID="$(date +%Y%m%d_%H%M%S)_learning_strategy_comparison"
readonly RESULTS_ROOT="${REPO_ROOT}/results/${COMPARISON_ID}"
readonly MODELS_ROOT="${REPO_ROOT}/model/${COMPARISON_ID}"
readonly FIGURES_ROOT="${REPO_ROOT}/figures/${COMPARISON_ID}"
readonly LOGS_ROOT="${REPO_ROOT}/logs/${COMPARISON_ID}"
readonly MANIFEST_FILE="${RESULTS_ROOT}/comparison_manifest.tsv"
readonly SUMMARY_FILE="${RESULTS_ROOT}/comparison_summary.tsv"

readonly STRATEGIES=(bptt bellec frenkel)
readonly TOTAL_CONFIGS=$((${#LEARNING_RATE_GRID[@]} * (2 * ${#GAMMA_GRID[@]} + 1)))

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$RESULTS_ROOT" "$MODELS_ROOT" "$FIGURES_ROOT" "$LOGS_ROOT"
    printf 'strategy\tlearning_rate\tbatch_size\tgamma\tstatus\tconfig_results_dir\n' > "$MANIFEST_FILE"
fi

echo "Learning-strategy comparison: $COMPARISON_ID"
echo "Training script: $TRAINING_SCRIPT"
echo "Python: $PYTHON_BIN"
echo "Fixed: letters=${LETTERS[*]}, hidden=$NB_HIDDEN, max_epochs=$EPOCHS, seed=$SEED, batch_size=$BATCH_SIZE, debug=true"
echo "Early stopping: epoch=$EARLY_STOP_EPOCHS, threshold=${EARLY_STOP_THRESHOLD}pp above chance (55% for A/B)"
echo "Selection: validation split enabled; configurations ranked by val_acc"
echo "Fixed for fairness: reg_spikes=0, reg_neurons=0"
echo "Learning-rate grid: ${LEARNING_RATE_GRID[*]}"
echo "Gamma grid (BPTT/Frenkel only): ${GAMMA_GRID[*]}"
echo "Configurations: $TOTAL_CONFIGS"
echo "Results root: $RESULTS_ROOT"
if [[ "$DRY_RUN" == true ]]; then
    echo "Mode: dry run (commands only)"
fi
echo

run_number=0
failures=0

for strategy in "${STRATEGIES[@]}"; do
    if [[ "$strategy" == "bellec" ]]; then
        # Bellec uses a fixed pseudo-derivative normalization; CLI gamma is ignored.
        strategy_gamma_grid=(15)
    else
        strategy_gamma_grid=("${GAMMA_GRID[@]}")
    fi

    for learning_rate in "${LEARNING_RATE_GRID[@]}"; do
        for gamma in "${strategy_gamma_grid[@]}"; do
            ((run_number += 1))

            config_name="${strategy}_lr_${learning_rate}_batch_${BATCH_SIZE}_gamma_${gamma}"
            config_results_dir="${RESULTS_ROOT}/${config_name}"
            config_models_dir="${MODELS_ROOT}/${config_name}"
            config_figures_dir="${FIGURES_ROOT}/${config_name}"
            config_logs_dir="${LOGS_ROOT}/${config_name}"

            command=(
                "$PYTHON_BIN"
                "$TRAINING_SCRIPT"
                --letters "${LETTERS[@]}"
                --nb_hidden "$NB_HIDDEN"
                --epochs "$EPOCHS"
                --seed "$SEED"
                --debug
                --validation
                --early_stop_epochs "$EARLY_STOP_EPOCHS"
                --early_stop_threshold "$EARLY_STOP_THRESHOLD"
                --repetitions 1
                --learning_rate "$learning_rate"
                --batch_size "$BATCH_SIZE"
                --gamma "$gamma"
                --reg_spikes 0
                --reg_neurons 0
                --results_path "$config_results_dir"
                --model_path "$config_models_dir"
                --fig_path "$config_figures_dir"
                --log_path "$config_logs_dir"
                --save_artifacts_for best
            )

            if [[ -n "$CUDA_ARG" ]]; then
                command+=("$CUDA_ARG")
            fi

            case "$strategy" in
                bptt)
                    # BPTT is the default path: do not pass --eprop.
                    ;;
                bellec)
                    command+=(--eprop --eprop_mode bellec)
                    ;;
                frenkel)
                    command+=(--eprop --eprop_mode frenkel)
                    ;;
            esac

            printf '[%d/%d] %s | lr=%s | batch=%s | gamma=%s\n' \
                "$run_number" "$TOTAL_CONFIGS" "$strategy" "$learning_rate" "$BATCH_SIZE" "$gamma"

            if [[ "$DRY_RUN" == true ]]; then
                printf '  '
                printf '%q ' "${command[@]}"
                printf '\n\n'
                continue
            fi

            if "${command[@]}"; then
                status="completed"
                echo "Completed: $config_name"
            else
                status="failed"
                ((failures += 1))
                echo "Failed: $config_name" >&2
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$strategy" "$learning_rate" "$BATCH_SIZE" "$gamma" "$status" "$config_results_dir" \
                >> "$MANIFEST_FILE"
            echo
        done
    done
done

if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

# Build a sortable summary from each completed run. val_acc is the accuracy of
# the checkpoint returned by braille_reading_rsnn.py; best_test and final_test
# are retained so that checkpoint and last-epoch behavior can both be compared.
"$PYTHON_BIN" - "$RESULTS_ROOT" "$SUMMARY_FILE" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

results_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []

for params_path in results_root.rglob("experiment_parameters.json"):
    with params_path.open(encoding="utf-8") as handle:
        params = json.load(handle)

    result_files = sorted(params_path.parent.glob("braille_reading_rsnn_*.npz"))
    for result_path in result_files:
        with np.load(result_path, allow_pickle=False) as result:
            acc_test = np.asarray(result["acc_test"], dtype=float)
            strategy = "bptt"
            if bool(params.get("eprop", False)):
                strategy = str(params.get("eprop_mode", "frenkel"))
            rows.append({
                "strategy": strategy,
                "learning_rate": float(params["learning_rate"]),
                "batch_size": int(params["batch_size"]),
                "gamma": float(params["gamma"]),
                "val_acc": float(result["val_acc"]),
                "best_test": float(np.max(acc_test)),
                "final_test": float(acc_test[-1]),
                "run_id": str(params.get("run_id", params_path.parent.name)),
                "result_file": str(result_path),
            })

rows.sort(key=lambda row: (row["strategy"], -row["val_acc"]))
fieldnames = [
    "strategy", "learning_rate", "batch_size", "gamma", "val_acc", "best_test",
    "final_test", "run_id", "result_file",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print("Best configuration per strategy (ranked by val_acc):")
for strategy in ("bptt", "bellec", "frenkel"):
    candidates = [row for row in rows if row["strategy"] == strategy]
    if not candidates:
        print(f"  {strategy:7s}: no completed result")
        continue
    best = max(candidates, key=lambda row: row["val_acc"])
    print(
        f"  {strategy:7s}: val_acc={best['val_acc']:.4f}, "
        f"best_test={best['best_test']:.4f}, final_test={best['final_test']:.4f}, "
        f"lr={best['learning_rate']:g}, batch={best['batch_size']}, gamma={best['gamma']:g}, "
        f"run_id={best['run_id']}"
    )
PY

echo
echo "Comparison complete."
echo "Manifest: $MANIFEST_FILE"
echo "Ranked summary: $SUMMARY_FILE"
echo "Failed configurations: $failures"

if ((failures > 0)); then
    exit 1
fi
