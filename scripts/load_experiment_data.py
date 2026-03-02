import numpy as np
import torch
from typing import Any, Dict, List
import importlib
import sys
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_local_utils_for_unpickling() -> None:
    """Ensure torch unpickling resolves `utils.*` from this workspace, not another project."""
    # If a different project already loaded `utils`, remove it so importlib re-resolves.
    existing_utils = sys.modules.get("utils")
    if existing_utils is not None:
        module_file = getattr(existing_utils, "__file__", "") or ""
        if str(PROJECT_ROOT) not in module_file:
            for module_name in [name for name in list(sys.modules.keys()) if name == "utils" or name.startswith("utils.")]:
                del sys.modules[module_name]

    importlib.import_module("utils.neuron_models")

experiment_id = "20260115_0833_exploration/20260225_092341"

data_path = f"./results/{experiment_id}"
data_file_name = "braille_reading_rsnn_5_neurons_A_B_rep_1.npz"

model_path = f"./model/{experiment_id}"
model_file_name = "best_model_5_neurons_A_B_rep_1.pt"
initial_weights_file_name = "initial_weights_5_neurons_A_B_rep_1.npz"
best_model_weights_file_name = "best_model_weights_5_neurons_A_B_rep_1.npz"

def load_experiment_npz(npz_path: str) -> Dict[str, Any]:
    """Load a saved experiment .npz file and auto-convert scalar fields.

    Notes
    -----
    - Scalar metadata fields are cast to native Python types.
    - Label-like vectors are normalized to 1D to handle both legacy and new file formats.
    """
    scalar_casts: Dict[str, Any] = {
        "val_acc": float,
        "repetition": int,
        "nb_hidden": int,
        "nb_epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "letters": str,
        "eprop": bool,
        "run_id": str,
    }

    unpacked: Dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]

            if key in {"trues", "preds", "network_input_labels"}:
                if isinstance(value, np.ndarray) and value.dtype == object:
                    try:
                        value = np.concatenate(list(value), axis=0)
                    except Exception:
                        value = np.asarray(value).reshape(-1)
                else:
                    value = np.asarray(value).reshape(-1)

            if key in scalar_casts:
                unpacked[key] = scalar_casts[key](value)
            else:
                unpacked[key] = value

    return unpacked


def load_model_pt(model_pt_path: str, map_location: str = "cpu") -> List[Any]:
    """Load a saved model (.pt) containing the layer objects."""
    _ensure_local_utils_for_unpickling()
    return torch.load(model_pt_path, map_location=map_location)


def load_weights_npz(weights_npz_path: str) -> Dict[str, np.ndarray]:
    """Load saved weight arrays from an .npz file into a standard dict."""
    with np.load(weights_npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _weight_stats(array: np.ndarray) -> Dict[str, Any]:
    """Compute distribution statistics for one weight tensor.

    Why these metrics:
    - mean: detects global bias/drift in weights.
    - std: measures spread/dispersion (capacity usage and regularization effects).
    - min/max: catches clipping/saturation and outliers.
    - abs_mean: magnitude summary independent of sign cancellation.
    - l2_norm: global energy/size of the parameter tensor.
    """
    arr = np.asarray(array)
    return {
        "shape": arr.shape,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "l2_norm": float(np.linalg.norm(arr)),
    }


def summarize_weight_changes(
    initial_weights: Dict[str, np.ndarray],
    final_weights: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """Compare initial and final weights and return per-matrix change summaries.

    Delta metrics show *how much* a weight tensor moved during training.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    common_keys = sorted(set(initial_weights.keys()) & set(final_weights.keys()))
    for key in common_keys:
        initial = np.asarray(initial_weights[key])
        final = np.asarray(final_weights[key])

        if initial.shape != final.shape:
            summary[key] = {
                "initial": _weight_stats(initial),
                "final": _weight_stats(final),
                "shape_mismatch": True,
            }
            continue

        delta = final - initial
        summary[key] = {
            "initial": _weight_stats(initial),
            "final": _weight_stats(final),
            "delta": {
                # mean: direction of average update (positive/negative drift)
                "mean": float(np.mean(delta)),
                # std: heterogeneity of updates across weights
                "std": float(np.std(delta)),
                # min/max: strongest negative/positive single-weight updates
                "min": float(np.min(delta)),
                "max": float(np.max(delta)),
                # abs_mean: average update magnitude regardless of sign
                "abs_mean": float(np.mean(np.abs(delta))),
                # l2_norm: total update energy for the whole tensor
                "l2_norm": float(np.linalg.norm(delta)),
            },
            "shape_mismatch": False,
        }
    return summary


def plot_weight_histograms(
    initial_weights: Dict[str, np.ndarray],
    final_weights: Dict[str, np.ndarray],
    output_dir: str,
    bins: int = 60
) -> List[str]:
    """Plot and save initial vs final weight histograms for each common weight matrix."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    common_keys = sorted(set(initial_weights.keys()) & set(final_weights.keys()))

    for key in common_keys:
        initial = np.asarray(initial_weights[key]).reshape(-1)
        final = np.asarray(final_weights[key]).reshape(-1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(initial, bins=bins, alpha=0.55, label="Initial", density=True)
        ax.hist(final, bins=bins, alpha=0.55, label="Final", density=True)
        ax.set_title(f"Weight Distribution: {key}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        save_path = output_path / f"{key}_initial_vs_final_hist.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        saved_files.append(str(save_path))

    return saved_files


def run_health_checks(experiment: Dict[str, Any]) -> List[str]:
    """Run lightweight integrity checks on loaded experiment content.

    Returns a list of warning strings. Empty list means all checks passed.
    """
    warnings: List[str] = []

    def _shape_or_na(value: Any) -> Any:
        return getattr(value, "shape", "n/a")

    val_acc = experiment.get("val_acc")
    if val_acc is not None and not (0.0 <= float(val_acc) <= 1.0):
        warnings.append(f"val_acc out of [0, 1] range: {val_acc}")

    trues = experiment.get("trues")
    preds = experiment.get("preds")
    if trues is not None and preds is not None and len(trues) != len(preds):
        warnings.append(f"trues/preds length mismatch: {len(trues)} vs {len(preds)}")

    network_input = experiment.get("network_input")
    network_input_labels = experiment.get("network_input_labels")
    spk_rec_hidden = experiment.get("spk_rec_hidden")
    spk_rec_readout = experiment.get("spk_rec_readout")

    if network_input is not None and getattr(network_input, "ndim", 0) != 3:
        warnings.append(f"network_input expected 3D [samples, time, channels], got shape {_shape_or_na(network_input)}")

    if network_input_labels is not None and getattr(network_input_labels, "ndim", 0) != 1:
        warnings.append(f"network_input_labels expected 1D [samples], got shape {_shape_or_na(network_input_labels)}")

    if spk_rec_hidden is not None and getattr(spk_rec_hidden, "ndim", 0) != 3:
        warnings.append(f"spk_rec_hidden expected 3D [samples, time, neurons], got shape {_shape_or_na(spk_rec_hidden)}")

    if spk_rec_readout is not None and getattr(spk_rec_readout, "ndim", 0) != 3:
        warnings.append(f"spk_rec_readout expected 3D [samples, time, neurons], got shape {_shape_or_na(spk_rec_readout)}")

    if network_input is not None and network_input_labels is not None:
        if network_input.shape[0] != network_input_labels.shape[0]:
            warnings.append(
                "network_input/network_input_labels sample mismatch: "
                f"{network_input.shape[0]} vs {network_input_labels.shape[0]}"
            )

    if network_input is not None and spk_rec_hidden is not None:
        if network_input.shape[0] != spk_rec_hidden.shape[0]:
            warnings.append(
                "network_input/spk_rec_hidden sample mismatch: "
                f"{network_input.shape[0]} vs {spk_rec_hidden.shape[0]}"
            )
        if network_input.shape[1] != spk_rec_hidden.shape[1]:
            warnings.append(
                "network_input/spk_rec_hidden timestep mismatch: "
                f"{network_input.shape[1]} vs {spk_rec_hidden.shape[1]}"
            )

    if network_input is not None and spk_rec_readout is not None:
        if network_input.shape[0] != spk_rec_readout.shape[0]:
            warnings.append(
                "network_input/spk_rec_readout sample mismatch: "
                f"{network_input.shape[0]} vs {spk_rec_readout.shape[0]}"
            )
        if network_input.shape[1] != spk_rec_readout.shape[1]:
            warnings.append(
                "network_input/spk_rec_readout timestep mismatch: "
                f"{network_input.shape[1]} vs {spk_rec_readout.shape[1]}"
            )

    if trues is not None and network_input_labels is not None and len(trues) != len(network_input_labels):
        warnings.append(f"trues/network_input_labels length mismatch: {len(trues)} vs {len(network_input_labels)}")

    nb_epochs = experiment.get("nb_epochs")
    acc_train = experiment.get("acc_train")
    acc_test = experiment.get("acc_test")
    loss_train = experiment.get("loss_train")

    if nb_epochs is not None:
        if acc_train is not None and len(acc_train) not in (0, int(nb_epochs)):
            warnings.append(f"acc_train length ({len(acc_train)}) differs from nb_epochs ({nb_epochs})")
        if acc_test is not None and len(acc_test) not in (0, int(nb_epochs)):
            warnings.append(f"acc_test length ({len(acc_test)}) differs from nb_epochs ({nb_epochs})")
        if loss_train is not None and len(loss_train) not in (0, int(nb_epochs)):
            warnings.append(f"loss_train length ({len(loss_train)}) differs from nb_epochs ({nb_epochs})")

    return warnings

if __name__ == "__main__":
    npz_path: str = f"{data_path}/{data_file_name}"
    model_pt_path: str = f"{model_path}/{model_file_name}"
    initial_weights_path: str = f"{model_path}/{initial_weights_file_name}"
    best_model_weights_path: str = f"{model_path}/{best_model_weights_file_name}"
    legacy_final_weights_path: str = f"{model_path}/{best_model_weights_file_name.replace('best_model_weights_', 'final_weights_')}"
    histogram_output_dir: str = f"./figures/{experiment_id}/weight_analysis"

    experiment: Dict[str, Any] = load_experiment_npz(npz_path)
    model_layers: List[Any] = load_model_pt(model_pt_path, map_location="cpu")
    initial_weights: Dict[str, np.ndarray] = load_weights_npz(initial_weights_path)
    if Path(best_model_weights_path).exists():
        selected_best_weights_path = best_model_weights_path
    elif Path(legacy_final_weights_path).exists():
        selected_best_weights_path = legacy_final_weights_path
        print(f"Using legacy weights filename: {legacy_final_weights_path}")
    else:
        raise FileNotFoundError(
            "Neither best_model_weights nor legacy final_weights file found in model path."
        )

    final_weights: Dict[str, np.ndarray] = load_weights_npz(selected_best_weights_path)
    weight_change_summary: Dict[str, Dict[str, Any]] = summarize_weight_changes(initial_weights, final_weights)
    histogram_files: List[str] = plot_weight_histograms(
        initial_weights=initial_weights,
        final_weights=final_weights,
        output_dir=histogram_output_dir,
        bins=60,
    )

    print("Available keys:", list(experiment.keys()))

    expected_keys = {
        "acc_train", "acc_test", "loss_train", "val_acc", "trues", "preds",
        "repetition", "nb_hidden", "nb_epochs", "learning_rate", "batch_size",
        "letters", "eprop", "run_id", "network_input", "network_input_labels",
        "spk_rec_hidden", "spk_rec_readout"
    }
    missing_expected: set[str] = expected_keys - set(experiment.keys())
    unexpected_keys: set[str] = set(experiment.keys()) - expected_keys

    if missing_expected:
        print("Missing expected keys:", sorted(missing_expected))
    if unexpected_keys:
        print("Additional keys in file:", sorted(unexpected_keys))
    if not missing_expected and not unexpected_keys:
        print("Key coverage check: all expected keys are present.")

    # Optional: assign to variables in one line
    acc_train: np.ndarray = experiment["acc_train"]
    acc_test: np.ndarray = experiment["acc_test"]
    loss_train: np.ndarray = experiment["loss_train"]
    val_acc: float = experiment["val_acc"]
    trues: np.ndarray = experiment["trues"]
    preds: np.ndarray = experiment["preds"]

    repetition: int = experiment["repetition"]
    nb_hidden: int = experiment["nb_hidden"]
    nb_epochs: int = experiment["nb_epochs"]
    learning_rate: float = experiment["learning_rate"]
    batch_size: int = experiment["batch_size"]
    letters: str = experiment["letters"]
    eprop: bool = experiment["eprop"]
    run_id: str = experiment["run_id"]

    network_input: np.ndarray = experiment["network_input"]
    network_input_labels: np.ndarray = experiment["network_input_labels"]
    spk_rec_hidden: np.ndarray = experiment["spk_rec_hidden"]
    spk_rec_readout: np.ndarray = experiment["spk_rec_readout"]

    accessed_keys = {
        "acc_train", "acc_test", "loss_train", "val_acc", "trues", "preds",
        "repetition", "nb_hidden", "nb_epochs", "learning_rate", "batch_size",
        "letters", "eprop", "run_id", "network_input", "network_input_labels",
        "spk_rec_hidden", "spk_rec_readout"
    }
    not_accessed: set[str] = set(experiment.keys()) - accessed_keys
    if not_accessed:
        print("Keys available but not assigned to variables:", sorted(not_accessed))
    else:
        print("Access check: all available keys are assigned to variables.")

    health_warnings: List[str] = run_health_checks(experiment)
    if health_warnings:
        print("Health checks: warnings found")
        for warning in health_warnings:
            print(f"  - {warning}")
    else:
        print("Health checks: all passed.")

    print(f"run_id={run_id}, val_acc={val_acc:.4f}, eprop={eprop}")
    print(f"input shape={network_input.shape}, hidden spikes shape={spk_rec_hidden.shape}, readout spikes shape={spk_rec_readout.shape}")
    print(f"Loaded model layers: {len(model_layers)}")
    print(f"Initial weight keys: {sorted(initial_weights.keys())}")
    print(f"Final weight keys: {sorted(final_weights.keys())}")
    print(f"Saved weight histograms to: {histogram_output_dir}")
    for hist_file in histogram_files:
        print(f"  - {hist_file}")

    print("\nWeight distribution summary (initial -> final, with delta):")
    for key in sorted(weight_change_summary.keys()):
        stats = weight_change_summary[key]
        if stats["shape_mismatch"]:
            print(f"- {key}: shape mismatch, cannot compute delta")
            print(f"    initial shape={stats['initial']['shape']}, final shape={stats['final']['shape']}")
            continue

        print(f"- {key}:")
        print(f"    initial mean/std = {stats['initial']['mean']:.6f} / {stats['initial']['std']:.6f}")
        print(f"    final   mean/std = {stats['final']['mean']:.6f} / {stats['final']['std']:.6f}")
        print(f"    delta   abs_mean = {stats['delta']['abs_mean']:.6f}, l2_norm = {stats['delta']['l2_norm']:.6f}")