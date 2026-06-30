"""find_best_experiments.py

Scan results directories and report best-performing trials per hidden neuron count.

This script inspects .npz result files created by braille_reading_rsnn_mod_eprop_reduce_label.py
and summarizes the top experiments for each hidden layer size.
"""

import argparse
import json
import numbers
from pathlib import Path

import numpy as np


DEFAULT_RESULTS_ROOT = "./results"
DEFAULT_METRIC = "val_acc"
METRIC_CHOICES = ["val_acc", "best_test", "final_test", "best_train"]


def _to_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if np.isscalar(value):
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, numbers.Real):
            return float(scalar)
        return scalar
    return value


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_nb_hidden(npz_path):
    parts = npz_path.stem.split("_")
    if "neurons" in parts:
        try:
            return int(parts[parts.index("neurons") - 1])
        except (IndexError, ValueError):
            return None
    return None


def _load_result(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    acc_train = data["acc_train"] if "acc_train" in data else None
    acc_test = data["acc_test"] if "acc_test" in data else None

    entry = {
        "path": str(npz_path),
        "run_dir": str(npz_path.parent),
        "run_id": _to_scalar(data["run_id"]) if "run_id" in data else None,
        "nb_hidden": _to_scalar(data["nb_hidden"]) if "nb_hidden" in data else _parse_nb_hidden(npz_path),
        "letters": _to_scalar(data["letters"]) if "letters" in data else None,
        "eprop": bool(_to_scalar(data["eprop"])) if "eprop" in data else None,
        "learning_rate": _to_scalar(data["learning_rate"]) if "learning_rate" in data else None,
        "batch_size": _to_scalar(data["batch_size"]) if "batch_size" in data else None,
        "nb_epochs": _to_scalar(data["nb_epochs"]) if "nb_epochs" in data else None,
        "repetition": _to_scalar(data["repetition"]) if "repetition" in data else None,
        "val_acc": _to_scalar(data["val_acc"]) if "val_acc" in data else None,
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    if acc_train is not None and len(acc_train) > 0:
        entry["final_train"] = float(acc_train[-1])
        entry["best_train"] = float(np.max(acc_train))
    else:
        entry["final_train"] = None
        entry["best_train"] = None

    if acc_test is not None and len(acc_test) > 0:
        entry["final_test"] = float(acc_test[-1])
        entry["best_test"] = float(np.max(acc_test))
    else:
        entry["final_test"] = None
        entry["best_test"] = None

    return entry


def _compute_metric(entry, metric):
    if metric == "val_acc":
        return entry["val_acc"] if entry["val_acc"] is not None else entry["best_test"]
    if metric == "best_test":
        return entry["best_test"]
    if metric == "final_test":
        return entry["final_test"]
    if metric == "best_train":
        return entry["best_train"]
    return None


def _collect_results(results_root):
    results = []
    for npz_path in Path(results_root).rglob("*.npz"):
        try:
            entry = _load_result(npz_path)
        except Exception as exc:
            print(f"Warning: Failed to load {npz_path}: {exc}")
            continue

        if entry["nb_hidden"] is None:
            continue

        if entry["acc_test"] is None and entry["val_acc"] is None:
            continue

        results.append(entry)

    return results


def _select_top_by_hidden(results, metric, top_k):
    grouped = {}
    for entry in results:
        metric_value = _compute_metric(entry, metric)
        if metric_value is None:
            continue
        entry = dict(entry)
        entry["metric"] = metric_value
        grouped.setdefault(entry["nb_hidden"], []).append(entry)

    best_by_hidden = {}
    for nb_hidden, entries in grouped.items():
        ranked = sorted(entries, key=lambda item: item["metric"], reverse=True)
        best_by_hidden[nb_hidden] = ranked[:top_k]

    return dict(sorted(best_by_hidden.items()))


def _filter_by_best_test(results, min_test_acc):
    matches = []
    for entry in results:
        best_test = entry.get("best_test")
        if best_test is None:
            continue
        if best_test >= min_test_acc:
            matches.append(entry)
    return matches


def _print_summary(best_by_hidden, metric):
    if not best_by_hidden:
        print("No usable results found.")
        return

    print("\nBest experiments per hidden neuron count")
    print("=" * 72)
    for nb_hidden, entries in best_by_hidden.items():
        print(f"\nHidden neurons: {nb_hidden} (metric: {metric})")
        for idx, entry in enumerate(entries, start=1):
            val_acc = entry.get("val_acc")
            best_test = entry.get("best_test")
            final_test = entry.get("final_test")
            letters = entry.get("letters")
            run_id = entry.get("run_id")
            eprop = entry.get("eprop")
            print(
                f"  {idx}. metric={entry['metric']:.4f} | val_acc={val_acc} | "
                f"best_test={best_test} | final_test={final_test} | "
                f"letters={letters} | eprop={eprop} | run_id={run_id}"
            )
            print(f"     file: {entry['path']}")


def _write_json(best_by_hidden, output_path):
    output = {}
    for nb_hidden, entries in best_by_hidden.items():
        output[str(nb_hidden)] = []
        for entry in entries:
            cleaned = dict(entry)
            cleaned.pop("acc_train", None)
            cleaned.pop("acc_test", None)
            output[str(nb_hidden)].append(cleaned)

    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)


def _print_detailed(results, min_test_acc):
    if not results:
        print(f"\nNo experiments with best_test >= {min_test_acc:.2f}.")
        return

    print(f"\nAll experiment details with best_test >= {min_test_acc:.2f}")
    print(f"Count: {len(results)}")
    print("=" * 72)
    for entry in results:
        print(json.dumps(_to_jsonable(entry), indent=2, sort_keys=True))
        print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Find best-performing experiments per hidden neuron count"
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing results subfolders",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=METRIC_CHOICES,
        help="Metric used to rank experiments",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top experiments to list per hidden neuron count (summary only)",
    )
    parser.add_argument(
        "--output-json",
        default="./results/summary.json",
        help="Path to save JSON summary (top-k per hidden)",
    )
    parser.add_argument(
        "--min-test-acc",
        type=float,
        default=0.60,
        help="Minimum best_test accuracy to print full experiment details",
    )
    parser.add_argument(
        "--output-json-over-threshold",
        default="./results/over_threshold.json",
        help="Path to save full JSON for experiments over threshold",
    )

    args = parser.parse_args()

    results = _collect_results(args.results_root)
    best_by_hidden = _select_top_by_hidden(results, args.metric, args.top_k)
    _print_summary(best_by_hidden, args.metric)

    over_threshold = _filter_by_best_test(results, args.min_test_acc)
    _print_detailed(over_threshold, args.min_test_acc)

    if args.output_json_over_threshold:
        with open(args.output_json_over_threshold, "w") as handle:
            json.dump(_to_jsonable(over_threshold), handle, indent=2, sort_keys=True)
        print(f"\nSaved full JSON to: {args.output_json_over_threshold}")

    if args.output_json:
        _write_json(best_by_hidden, args.output_json)
        print(f"\nSaved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
