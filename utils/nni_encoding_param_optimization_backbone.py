"""
utils/nni_encoding_param_optimization_backbone.py

One complete NNI trial:

1. Receive encoding parameters from NNI.
2. Encode the tactile input data into a trial-specific file.
3. Train and evaluate the SNN using that encoded file.
4. Report intermediate and final validation performance to NNI.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from typing import Any

import nni

# Support both ``python -m utils.nni_encoding_param_optimization_backbone``
# and direct execution of this file.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.nni_braille_reading_rsnn import AccuracyMetrics, run_training
from utils.nni_event_transform import (EncodingConfig, count_trial_spikes,
                                       run_encoding)

DEFAULT_PARAMS: dict[str, int] = {
    "threshold": 1,
    "max_frequency": 10,
    "refractory_period_ms": 0,
}

QUICK_TEST_LETTERS = ("A", "B", "C")
MIN_TRIALS_PER_CLASS_FOR_THREE_WAY_SPLIT = 10

# These parameters remain fixed during the encoding HPO.
FIXED_TRAINING_PARAMS: dict[str, Any] = {
    "epochs": 30,
    "nb_hidden": 25,
    # Optional: list of letters to work with.
    "letters": list(QUICK_TEST_LETTERS),

    "repetitions": 1,
    "batch_size": 128,
    "learning_rate": 0.005,
    "seed": 42,
    "validation": True,
    "encoding_type": "mechanoreceptor",
    "save_artifacts_for": "none",
    "log_level": "INFO",
    "cuda": True,
}


def build_encoding_config(
    params: dict[str, Any],
    output_path: Path,
) -> EncodingConfig:
    """
    Convert NNI parameters into one encoder configuration.

    NNI samples refractory_period_ms in milliseconds, while the
    mechanoreceptor encoder expects seconds.
    """

    refractory_period_seconds = (
        float(params["refractory_period_ms"])
        / 1000.0
    )

    return EncodingConfig(
        encoding_type="mechanoreceptor",

        mechanoreceptor_threshold=float(params["threshold"]),
        mechanoreceptor_max_frequency=float(params["max_frequency"]),
        mechanoreceptor_refractory_period=refractory_period_seconds,

        data_path="data/100Hz",
        data_files=(
            "data_braille_letters_0.0.pkl",
            "data_braille_letters_0.000125.pkl",
        ),

        letters=QUICK_TEST_LETTERS,

        # A stratified 70/20/10 train/test/validation split needs at least
        # 10 samples per class so every class can occur in every subset.
        nb_trials=MIN_TRIALS_PER_CLASS_FOR_THREE_WAY_SPLIT,

        output_path=output_path,
        save_output=True,
    )


def train_and_evaluate(
    encoded_data_path: Path,
    trial_id: str,
) -> AccuracyMetrics:
    """
    Train the SNN using a trial-specific encoded pickle file.

    Intermediate results are currently reported after each successful
    training repetition. Per-epoch reporting would need a callback
    inside build_and_train().
    """

    reported_metrics: list[AccuracyMetrics] = []

    def report_intermediate(metrics: AccuracyMetrics) -> None:
        reported_metrics.append(metrics)
        nni.report_intermediate_result(metrics)

    final_validation_accuracy = run_training(
        encoded_data_path=encoded_data_path,
        parameter_overrides=FIXED_TRAINING_PARAMS,
        intermediate_result_callback=report_intermediate,
        run_id=f"nni_{trial_id}",
    )

    if not reported_metrics:
        raise RuntimeError("Training returned no accuracy metrics.")

    best_metrics = max(
        reported_metrics,
        key=lambda metrics: metrics["default"],
    )
    if not math.isclose(
        float(final_validation_accuracy),
        best_metrics["default"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Training returned a final optimization metric that does not match "
            "the best reported validation accuracy."
        )
    return best_metrics


def main() -> None:
    # This must be called exactly once in each NNI trial.
    optimized_params = nni.get_next_parameter()

    params = DEFAULT_PARAMS.copy()
    params.update(optimized_params)

    trial_id = nni.get_trial_id()

    print(f"NNI trial ID: {trial_id}")
    print(f"NNI encoding parameters: {params}")

    # Every parallel trial gets its own directory.
    trial_directory = (
        Path("nni_trial_outputs")
        / trial_id
    )

    trial_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded_data_path = (
        trial_directory
        / "mechanoreceptor_encoded.pkl"
    )

    encoding_config = build_encoding_config(
        params=params,
        output_path=encoded_data_path,
    )

    try:
        encoded_data, saved_path = run_encoding(
            config=encoding_config,
            show_progress=False,
        )

        spike_counts = [
            count_trial_spikes(encoded_data, trial_index)
            for trial_index in range(len(encoded_data["letter"]))
        ]

        encoding_stats = {
            "number_of_trials": len(spike_counts),
            "mean_spikes": sum(spike_counts) / len(spike_counts),
            "min_spikes": min(spike_counts),
            "max_spikes": max(spike_counts),
        }

        print(
            f"[encoding done] path={saved_path}, "
            f"statistics={encoding_stats}",
            flush=True,
        )

        if saved_path is None:
            raise RuntimeError(
                "The encoder did not save an output file."
            )

        saved_path = Path(saved_path)

        if not saved_path.is_file():
            raise FileNotFoundError(
                f"Encoded data file was not created: "
                f"{saved_path}"
            )

        accuracy_metrics = train_and_evaluate(
            encoded_data_path=saved_path,
            trial_id=trial_id,
        )

        print(
            f"Final validation accuracy: {accuracy_metrics['default']:.6f}; "
            f"training accuracy: {accuracy_metrics['training_accuracy']:.6f}; "
            f"test accuracy (reporting only): {accuracy_metrics['test_accuracy']:.6f}"
        )

        # Report the final result exactly once.
        nni.report_final_result({
            **accuracy_metrics,
            **encoding_stats,
        })

    finally:
        # Remove the temporary encoded file after training.
        #
        # Comment this out while debugging if you want to inspect
        # each trial's encoded pickle.
        shutil.rmtree(
            trial_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
