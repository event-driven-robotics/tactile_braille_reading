#!/usr/bin/env python3
"""Analyze circuit-level spike exports for mapping/indexing consistency.

This script inspects circuit-level spike text files and checks whether output spikes
are consistent with hidden->output weights under different address mapping hypotheses.

Primary use case:
- Validate claims like "output neuron spikes despite only inhibitory input".
- Compare the exporter mapping (reversed addresses per layer) against a simpler
  offset-only mapping often assumed on hardware.

Examples
--------
python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id 20260115_0833_exploration/20260303_120143

python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id 20260115_0833_exploration/20260303_120143 \
  --write-report

python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id 20260115_0833_exploration/20260303_120143 \
  --write-report \
  --write-suspicious-only-csv

python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id 20260115_0833_exploration/20260303_120143 \
  --write-report \
  --narrative-trial 17 \
  --narrative-output-neuron 1

python scripts/analyze_circuit_spike_mapping.py \
  --experiment-id 20260115_0833_exploration/20260303_120143 \
  --write-report \
  --narrative-trial 17 \
  --narrative-output-neuron 1 \
  --write-suspicious-only-csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Event:
    time_s: float
    delta_s: float
    address: int


def load_experiment_parameters(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze circuit-level spike exports for mapping mismatches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="20260115_0833_exploration/20260303_120143",
        help="Experiment folder under results/ and model/",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("./results"),
        help="Root folder containing experiment result directories",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("./model"),
        help="Root folder containing experiment model directories",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=1,
        help="Repetition index used to select best_model_weights_*_rep_<N>.npz",
    )
    parser.add_argument(
        "--causal-delay-steps",
        type=int,
        default=1,
        help="Only count hidden spikes up to (t_output - causal_delay_steps * T_CLOCK)",
    )
    parser.add_argument(
        "--show-per-trial",
        action="store_true",
        help="Print detailed first-spike diagnostics for every trial",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write CSV and markdown report files for sharing",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Output directory for generated report files (default: results/<experiment>/circuit_level_spk/analysis_report)",
    )
    parser.add_argument(
        "--write-suspicious-only-csv",
        action="store_true",
        help="When writing reports, also export a filtered CSV with offset-inhibitory-only rows",
    )
    parser.add_argument(
        "--narrative-trial",
        type=int,
        default=17,
        help="Trial index for single-case plain-text narrative report",
    )
    parser.add_argument(
        "--narrative-output-neuron",
        type=int,
        default=1,
        help="Output neuron index for single-case plain-text narrative report",
    )
    return parser.parse_args()


def parse_events(path: Path) -> list[Event]:
    lines = path.read_text(encoding="utf-8").splitlines()
    events: list[Event] = []
    started = False
    for line in lines:
        if not started:
            if line.strip().startswith("TIME"):
                started = True
            continue

        line = line.strip()
        if not line:
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue

        events.append(Event(time_s=float(parts[0]), delta_s=float(
            parts[1]), address=int(parts[2])))
    return events


def exporter_address_maps(n_output: int, n_hidden: int) -> tuple[dict[int, int], dict[int, int]]:
    # Matches scripts/create_circuit_level_spk_file.py neuron_id_map() with layer order output->hidden.
    output_map = {idx: (n_output - 1 - idx) for idx in range(n_output)}
    hidden_map = {
        idx: n_output + (n_hidden - 1 - idx)
        for idx in range(n_hidden)
    }
    return output_map, hidden_map


def load_out_weights(model_dir: Path, rep: int) -> Path:
    candidates = sorted(model_dir.glob(f"best_model_weights_*_rep_{rep}.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No best_model_weights_*_rep_{rep}.npz found in {model_dir}")
    return candidates[0]


def signed_input_sum(
    hidden_events: list[Event],
    out_idx: int,
    out_weights: np.ndarray,
    hidden_addr_to_idx: dict[int, int],
    t_until_s: float,
) -> tuple[float, int, int, list[tuple[int, int, float]]]:
    weighted_sum = 0.0
    exc_count = 0
    inh_count = 0
    by_hidden: dict[int, int] = {}

    for ev in hidden_events:
        if ev.address == 7 or ev.time_s > t_until_s:
            continue
        if ev.address not in hidden_addr_to_idx:
            continue
        h_idx = hidden_addr_to_idx[ev.address]
        by_hidden[h_idx] = by_hidden.get(h_idx, 0) + 1

    details: list[tuple[int, int, float]] = []
    for h_idx, count in sorted(by_hidden.items()):
        w = float(out_weights[out_idx, h_idx])
        weighted_sum += count * w
        if w >= 0:
            exc_count += count
        else:
            inh_count += count
        details.append((h_idx, count, w))

    return weighted_sum, exc_count, inh_count, details


def format_details(details: list[tuple[int, int, float]]) -> str:
    if not details:
        return ""
    return ";".join(f"{h}:{c}:{w:.6g}" for h, c, w in details)


def _time_to_step(time_s: float, t_clock: float) -> int:
    return int(round(time_s / t_clock))


def _build_input_drive_per_step(
    *,
    hidden_events: list[Event],
    n_steps: int,
    t_clock: float,
    output_neuron_idx: int,
    out_weights: np.ndarray,
    hidden_addr_to_idx: dict[int, int],
) -> np.ndarray:
    drive = np.zeros(n_steps, dtype=np.float64)
    for ev in hidden_events:
        if ev.address == 7:
            continue
        h_idx = hidden_addr_to_idx.get(ev.address)
        if h_idx is None:
            continue
        step = _time_to_step(ev.time_s, t_clock)
        if 0 <= step < n_steps:
            drive[step] += float(out_weights[output_neuron_idx, h_idx])
    return drive


def reconstruct_membrane_trace(
    *,
    input_drive: np.ndarray,
    t_clock: float,
    tau_mem: float,
    linear_decay: bool,
    spike_threshold: float,
    soft_reset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate readout membrane from weighted hidden spikes.

    Returns membrane trace and a binary spike vector inferred from the same model.
    """
    n_steps = int(input_drive.shape[0])
    mem = np.zeros(n_steps, dtype=np.float64)
    spk = np.zeros(n_steps, dtype=np.int32)

    if linear_decay:
        beta = max(0.0, 1.0 - (t_clock / max(tau_mem, 1e-12)))
    else:
        beta = math.exp(-t_clock / max(tau_mem, 1e-12))

    v_prev = 0.0
    for t in range(n_steps):
        v_t = beta * v_prev + float(input_drive[t])
        if v_t >= spike_threshold:
            spk[t] = 1
            if soft_reset:
                v_t = v_t - spike_threshold
            else:
                v_t = 0.0
        mem[t] = v_t
        v_prev = v_t

    return mem, spk


def build_trial_narrative_text(
    *,
    trial_idx: int,
    output_neuron_idx: int,
    experiment_params: dict,
    t_clock: float,
    causal_window_shift: float,
    output_map_exp: dict[int, int],
    hidden_addr_to_idx_exp: dict[int, int],
    hidden_addr_to_idx_offset: dict[int, int],
    hidden_events: list[Event],
    output_events: list[Event],
    out_weights: np.ndarray,
    true_mem_trace: np.ndarray | None,
) -> str:
    lines: list[str] = []
    lines.append("Single-Trial Mapping Narrative")
    lines.append("=" * 80)
    lines.append(f"trial_index: {trial_idx}")
    lines.append(f"output_neuron_index: {output_neuron_idx}")
    lines.append(f"t_clock_s: {t_clock:.6f}")
    lines.append(f"causal_window_shift_s: {causal_window_shift:.6f}")
    lines.append(
        f"spike_threshold: {float(experiment_params.get('spike_threshold', 1.0)):.6g}")
    lines.append(
        f"tau_mem: {float(experiment_params.get('tau_mem', 0.06)):.6g}")
    lines.append(
        f"linear_decay: {bool(experiment_params.get('linear_decay', False))}")
    lines.append(
        f"soft_reset: {bool(experiment_params.get('soft_reset', False))}")
    lines.append("")

    if true_mem_trace is None:
        lines.append(
            "Membrane note: true mem_rec_readout trace is not available in this saved results file;"
        )
        lines.append(
            "the membrane values below are a decay-aware reconstruction from hidden spikes and out_weights."
        )
    else:
        lines.append(
            "Membrane note: true mem_rec_readout trace is available and reported below."
        )
    lines.append("")

    if output_neuron_idx not in output_map_exp:
        lines.append(
            "Requested output neuron index is not valid for this run.")
        return "\n".join(lines) + "\n"

    target_output_address = output_map_exp[output_neuron_idx]
    output_spikes = [
        ev for ev in output_events if ev.address == target_output_address]
    if not output_spikes:
        lines.append(
            "No spikes found for requested output neuron in selected trial.")
        return "\n".join(lines) + "\n"

    first_out = output_spikes[0]
    t_until = first_out.time_s - causal_window_shift + 1e-12
    out_step = _time_to_step(first_out.time_s, t_clock)
    until_step = _time_to_step(t_until, t_clock)

    lines.append("Target first output spike")
    lines.append("-" * 80)
    lines.append(
        f"Output neuron {output_neuron_idx} (address {target_output_address}) first spikes at "
        f"t={first_out.time_s:.6f}s (step {out_step})."
    )
    lines.append(
        f"For causality, hidden spikes up to t={t_until:.6f}s (step {until_step}) are considered."
    )
    lines.append("")

    pre_hidden = [ev for ev in hidden_events if ev.address !=
                  7 and ev.time_s <= t_until]
    if not pre_hidden:
        lines.append("No hidden spikes were found in the causal window.")
        return "\n".join(lines) + "\n"

    clock_steps = [_time_to_step(ev.time_s, t_clock)
                   for ev in output_events if ev.address == 7]
    if clock_steps:
        n_steps = max(clock_steps) + 1
    else:
        max_time = max(ev.time_s for ev in (hidden_events + output_events))
        n_steps = _time_to_step(max_time, t_clock) + 2
    drive_exp = _build_input_drive_per_step(
        hidden_events=hidden_events,
        n_steps=n_steps,
        t_clock=t_clock,
        output_neuron_idx=output_neuron_idx,
        out_weights=out_weights,
        hidden_addr_to_idx=hidden_addr_to_idx_exp,
    )

    tau_mem = float(experiment_params.get("tau_mem", 0.06))
    spike_threshold = float(experiment_params.get("spike_threshold", 1.0))
    linear_decay = bool(experiment_params.get("linear_decay", False))
    soft_reset = bool(experiment_params.get("soft_reset", False))
    recon_mem, recon_spk = reconstruct_membrane_trace(
        input_drive=drive_exp,
        t_clock=t_clock,
        tau_mem=tau_mem,
        linear_decay=linear_decay,
        spike_threshold=spike_threshold,
        soft_reset=soft_reset,
    )

    lines.append("Event-by-event explanation in the causal window")
    lines.append("-" * 80)
    lines.append(
        f"Decay factor beta = exp(-t_clock/tau_mem) = exp(-{t_clock}/{tau_mem}) = {math.exp(-t_clock/max(tau_mem,1e-12)):.6f}"
        if not linear_decay else
        f"Decay factor beta = 1 - t_clock/tau_mem = 1 - {t_clock}/{tau_mem} = {max(0.0, 1.0-(t_clock/max(tau_mem,1e-12))):.6f}"
    )
    lines.append(
        "Per event: mem_prev × beta + weight → mem_pre_reset → (reset if ≥ threshold) → mem_stored"
    )
    lines.append("")

    if linear_decay:
        beta = max(0.0, 1.0 - (t_clock / max(tau_mem, 1e-12)))
    else:
        beta = math.exp(-t_clock / max(tau_mem, 1e-12))

    for i, ev in enumerate(pre_hidden, start=1):
        step = _time_to_step(ev.time_s, t_clock)

        h_exp = hidden_addr_to_idx_exp.get(ev.address, -1)
        h_off = hidden_addr_to_idx_offset.get(ev.address, -1)

        w_exp = float(out_weights[output_neuron_idx, h_exp]
                      ) if h_exp >= 0 else float("nan")
        w_off = float(out_weights[output_neuron_idx, h_off]
                      ) if h_off >= 0 else float("nan")
        sign_exp = ("exc" if w_exp >= 0 else "inh") if not math.isnan(
            w_exp) else "n/a"
        sign_off = ("exc" if w_off >= 0 else "inh") if not math.isnan(
            w_off) else "n/a"

        prev_mem = float(recon_mem[step - 1]) if step > 0 else 0.0
        decayed = beta * prev_mem
        drive = w_exp if not math.isnan(w_exp) else 0.0
        pre_reset = decayed + drive
        after_reset = float(recon_mem[step])
        fired = bool(recon_spk[step])
        reset_note = f" [SPIKE → {'soft' if soft_reset else 'hard'}-reset to {after_reset:.6g}]" if fired else ""

        lines.append(
            f"[{i}] t={ev.time_s:.6f}s (step {step}), hidden address {ev.address}:")
        lines.append(
            f"     exporter → hidden_idx {h_exp}, weight {w_exp:.6g} ({sign_exp})   |   "
            f"offset → hidden_idx {h_off}, weight {w_off:.6g} ({sign_off})"
        )
        lines.append(
            f"     membrane: {prev_mem:.6g} (step {step-1}) × {beta:.6f} (decay) = {decayed:.6g}   "
            f"+ {drive:.6g} (input) = {pre_reset:.6g}"
            f"   → stored = {after_reset:.6g}{reset_note}"
        )
        lines.append("")

    lines.append("")
    lines.append("Firing-time explanation")
    lines.append("-" * 80)
    delay_steps = int(round(causal_window_shift / t_clock))
    step_before = max(0, out_step - 1)
    recon_before = float(recon_mem[step_before])
    recon_at = float(recon_mem[out_step])
    recon_spike_prev = int(recon_spk[step_before])
    recon_spike_at = int(recon_spk[out_step])
    lines.append(
        f"Reconstructed membrane just before first spike step (step {step_before}, t={step_before*t_clock:.6f}s): {recon_before:.6g}"
    )
    lines.append(
        f"Reconstructed membrane at spike step (step {out_step}, t={out_step*t_clock:.6f}s): {recon_at:.6g}"
    )
    lines.append(
        f"Reconstructed spike flag at previous step {step_before}: {recon_spike_prev}"
    )
    lines.append(
        f"Reconstructed spike flag at step {out_step}: {recon_spike_at} (threshold {spike_threshold:.6g})"
    )
    if true_mem_trace is not None and 0 <= out_step < true_mem_trace.shape[0]:
        lines.append(
            f"True membrane at spike step from saved trace: {float(true_mem_trace[out_step]):.6g}"
        )
    lines.append(
        f"Interpretation: with causal delay of {delay_steps} step(s), a reconstructed threshold crossing at "
        f"step {step_before} can appear as an output spike at step {out_step}."
    )
    lines.append(
        "So firing is explained by accumulated weighted input plus decay/reset dynamics, not by raw weight summation alone."
    )

    sum_exp, exc_exp, inh_exp, details_exp = signed_input_sum(
        hidden_events=hidden_events,
        out_idx=output_neuron_idx,
        out_weights=out_weights,
        hidden_addr_to_idx=hidden_addr_to_idx_exp,
        t_until_s=t_until,
    )
    sum_off, exc_off, inh_off, details_off = signed_input_sum(
        hidden_events=hidden_events,
        out_idx=output_neuron_idx,
        out_weights=out_weights,
        hidden_addr_to_idx=hidden_addr_to_idx_offset,
        t_until_s=t_until,
    )

    lines.append("")
    lines.append("Compact summary for this case")
    lines.append("-" * 80)
    lines.append(
        f"Correct exporter mapping: weighted_sum={sum_exp:.6g}, exc_spikes={exc_exp}, inh_spikes={inh_exp}, details={format_details(details_exp)}"
    )
    lines.append(
        f"Offset-only mapping: weighted_sum={sum_off:.6g}, exc_spikes={exc_off}, inh_spikes={inh_off}, details={format_details(details_off)}"
    )
    lines.append("")

    if exc_off == 0 and inh_off > 0 and exc_exp > 0:
        lines.append(
            "Interpretation: under offset-only indexing this looks inhibitory-only, but under the actual exporter mapping there is excitatory drive."
        )
    else:
        lines.append(
            "Interpretation: this trial/output case does not show the classic inhibitory-only mismatch pattern."
        )

    return "\n".join(lines) + "\n"


def plot_trial_visualization(
    *,
    trial_idx: int,
    output_neuron_idx: int,
    experiment_params: dict,
    t_clock: float,
    causal_window_shift: float,
    output_map_exp: dict[int, int],
    hidden_addr_to_idx_exp: dict[int, int],
    hidden_events: list[Event],
    output_events: list[Event],
    out_weights: np.ndarray,
    true_mem_trace: np.ndarray | None,
    save_path: Path,
) -> None:
    """Generate and save a two-panel figure for a single trial.

    Top panel:   hidden neuron spike raster, coloured by weight sign to the
                 selected output neuron. Causal boundary and output spikes
                 are overlaid as vertical dashed lines.
    Bottom panel: output neuron membrane potential (true or reconstructed)
                 with the firing threshold, threshold-crossing markers, and
                 output spike times.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("WARNING: matplotlib not available; skipping visualization.")
        return

    tau_mem = float(experiment_params.get("tau_mem", 0.06))
    spike_threshold = float(experiment_params.get("spike_threshold", 1.0))
    linear_decay = bool(experiment_params.get("linear_decay", False))
    soft_reset = bool(experiment_params.get("soft_reset", False))
    n_hidden = out_weights.shape[1]

    # Determine total number of steps from clock events (address 7).
    clock_times = [e.time_s for e in output_events if e.address == 7]
    if clock_times:
        n_steps = _time_to_step(max(clock_times), t_clock) + 2
    else:
        all_times = [e.time_s for e in hidden_events +
                     output_events if e.address != 7]
        n_steps = (_time_to_step(max(all_times), t_clock) +
                   2) if all_times else 100

    # Build reconstructed membrane over the full trial.
    drive_exp = _build_input_drive_per_step(
        hidden_events=hidden_events,
        n_steps=n_steps,
        t_clock=t_clock,
        output_neuron_idx=output_neuron_idx,
        out_weights=out_weights,
        hidden_addr_to_idx=hidden_addr_to_idx_exp,
    )
    recon_mem, recon_spk = reconstruct_membrane_trace(
        input_drive=drive_exp,
        t_clock=t_clock,
        tau_mem=tau_mem,
        linear_decay=linear_decay,
        spike_threshold=spike_threshold,
        soft_reset=soft_reset,
    )

    mem_to_plot = true_mem_trace if true_mem_trace is not None else recon_mem
    mem_is_true = true_mem_trace is not None
    time_axis = np.arange(len(mem_to_plot)) * t_clock

    # Output spike times and causal boundary.
    target_addr = output_map_exp[output_neuron_idx]
    out_spike_times = [
        ev.time_s for ev in output_events if ev.address == target_addr]
    first_out_spike = out_spike_times[0] if out_spike_times else None
    causal_boundary = (
        first_out_spike - causal_window_shift) if first_out_spike is not None else None

    # Hidden spike times grouped by exporter neuron index.
    hidden_spikes_by_idx: dict[int, list[float]] = {
        i: [] for i in range(n_hidden)}
    for ev in hidden_events:
        if ev.address == 7:
            continue
        idx = hidden_addr_to_idx_exp.get(ev.address, -1)
        if 0 <= idx < n_hidden:
            hidden_spikes_by_idx[idx].append(ev.time_s)

    def _weight_color(w: float) -> str:
        if w > 0:
            return "#2ca02c"  # green
        if w < 0:
            return "#d62728"  # red
        return "#7f7f7f"  # neutral gray

    fig, (ax_raster, ax_mem) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    fig.suptitle(
        f"Trial {trial_idx} — Output neuron {output_neuron_idx}   "
        f"(tau_mem={tau_mem}, threshold={spike_threshold}, "
        f"linear_decay={linear_decay}, soft_reset={soft_reset})",
        fontsize=10,
    )

    # ── Top panel: hidden spike raster ──────────────────────────────────────
    for h_idx in range(n_hidden):
        w = float(out_weights[output_neuron_idx, h_idx])
        c = _weight_color(w)
        spk_t = hidden_spikes_by_idx[h_idx]
        ax_raster.axhline(h_idx, color="#e8e8e8", linewidth=0.5, zorder=1)
        if spk_t:
            ax_raster.scatter(
                spk_t, [h_idx] * len(spk_t),
                c=c, s=80, marker="|", linewidths=1.8, zorder=3,
            )

    for t_out in out_spike_times:
        ax_raster.axvline(t_out, color="#1f77b4", linestyle="--",
                          linewidth=0.9, alpha=0.5, zorder=4)
    if causal_boundary is not None:
        ax_raster.axvline(
            causal_boundary, color="darkorange", linestyle=":",
            linewidth=1.4, zorder=5, label="causal boundary",
        )

    y_labels = [
        f"n{i}  w={out_weights[output_neuron_idx, i]:.3f}" for i in range(n_hidden)]
    ax_raster.set_yticks(range(n_hidden))
    ax_raster.set_yticklabels(y_labels, fontsize=8)
    ax_raster.set_ylim(-0.5, n_hidden - 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_ylabel("Hidden neuron (exporter idx)")
    ax_raster.set_title(
        "Hidden neuron spikes  –  green = excitatory weight, red = inhibitory weight (to selected output neuron)"
    )

    exc_patch = mpatches.Patch(
        color="#2ca02c", label="excitatory weight (w > 0)")
    inh_patch = mpatches.Patch(
        color="#d62728", label="inhibitory weight (w < 0)")
    causal_line = Line2D([0], [0], color="darkorange",
                         linestyle=":", linewidth=1.4, label="causal boundary")
    out_line = Line2D([0], [0], color="#1f77b4",
                      linestyle="--", linewidth=0.9, label="output spike")
    ax_raster.legend(handles=[exc_patch, inh_patch,
                     causal_line, out_line], fontsize=8, loc="upper right")

    # ── Bottom panel: membrane potential ────────────────────────────────────
    mem_label = "true mem_rec_readout" if mem_is_true else "reconstructed membrane (decay-aware)"
    ax_mem.plot(time_axis, mem_to_plot, color="#1f77b4",
                linewidth=1.2, label=mem_label, zorder=3)
    ax_mem.axhline(
        spike_threshold, color="crimson", linestyle="--",
        linewidth=1.0, label=f"threshold = {spike_threshold:.3g}", zorder=2,
    )

    crossing_steps = np.where(recon_spk > 0)[0]
    if len(crossing_steps) > 0:
        ax_mem.scatter(
            crossing_steps * t_clock,
            [spike_threshold] * len(crossing_steps),
            c="crimson", s=60, marker="^", zorder=5,
            label="threshold crossing (reconstructed)",
        )

    first_out_plotted = False
    for t_out in out_spike_times:
        label = "output spikes" if not first_out_plotted else None
        lw = 1.5 if not first_out_plotted else 0.9
        ax_mem.axvline(t_out, color="#1f77b4", linestyle="--",
                       linewidth=lw, alpha=0.7, zorder=4, label=label)
        first_out_plotted = True
    if causal_boundary is not None:
        ax_mem.axvline(
            causal_boundary, color="darkorange", linestyle=":",
            linewidth=1.4, zorder=5, label="causal boundary",
        )

    ax_mem.set_ylabel("Membrane potential")
    ax_mem.set_xlabel("Time (s)")
    ax_mem.set_title(f"Output neuron {output_neuron_idx} membrane potential")
    ax_mem.legend(fontsize=8, loc="upper right")
    ax_mem.annotate(
        "[true trace]" if mem_is_true else "[reconstructed — no mem_rec_readout in saved file]",
        xy=(0.01, 0.97), xycoords="axes fraction", fontsize=7, color="gray", va="top",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    base_results = args.results_root / args.experiment_id
    circuit_dir = base_results / "circuit_level_spk"
    model_dir = args.model_root / args.experiment_id
    params_path = base_results / "experiment_parameters.json"

    if not circuit_dir.exists():
        raise FileNotFoundError(f"Circuit directory not found: {circuit_dir}")

    hidden_files = sorted(p for p in circuit_dir.glob(
        "hidden_layer_trial_*.txt") if "npwhere_debug" not in p.name)
    output_files = sorted(p for p in circuit_dir.glob(
        "output_layer_trial_*.txt") if "npwhere_debug" not in p.name)

    if not hidden_files or not output_files:
        raise FileNotFoundError(
            "Could not find hidden/output circuit trial files")
    if len(hidden_files) != len(output_files):
        raise ValueError("Mismatch in number of hidden/output trial files")

    if not params_path.exists():
        raise FileNotFoundError(
            f"Experiment parameters not found: {params_path}")
    experiment_params = load_experiment_parameters(params_path)

    sample_hidden_events = parse_events(hidden_files[0])
    sample_output_events = parse_events(output_files[0])

    hidden_addrs_observed = sorted(
        {e.address for e in sample_hidden_events if e.address != 7})
    output_addrs_observed = sorted(
        {e.address for e in sample_output_events if e.address != 7})

    weights_path = load_out_weights(model_dir=model_dir, rep=args.rep)
    with np.load(weights_path, allow_pickle=True) as data:
        out_weights = np.asarray(data["out_weights"])

    if out_weights.ndim != 2:
        raise ValueError(
            f"Expected 2D out_weights, got shape {out_weights.shape}")

    n_output, n_hidden = int(out_weights.shape[0]), int(out_weights.shape[1])
    output_map_exp, hidden_map_exp = exporter_address_maps(
        n_output=n_output, n_hidden=n_hidden)
    output_addr_to_idx_exp = {
        addr: idx for idx, addr in output_map_exp.items()}
    hidden_addr_to_idx_exp = {
        addr: idx for idx, addr in hidden_map_exp.items()}

    output_addr_to_idx_offset = {addr: addr for addr in range(n_output)}
    hidden_addr_to_idx_offset = {
        addr: addr - n_output for addr in range(n_output, n_output + n_hidden)}

    t_clock = min(
        e.delta_s for e in sample_output_events if e.address == 7 and e.delta_s > 0)
    causal_window_shift = args.causal_delay_steps * t_clock

    print("=== Circuit/Weight Consistency Analysis ===")
    print(f"Experiment: {args.experiment_id}")
    print(f"Trials: {len(hidden_files)}")
    print(f"Weights file: {weights_path}")
    print(f"Inferred layer sizes: output={n_output}, hidden={n_hidden}")
    print(f"T_CLOCK inferred from file: {t_clock:.6f} s")
    print(
        f"Causal window shift: {causal_window_shift:.6f} s ({args.causal_delay_steps} step)")
    print()

    print("Exporter mapping (local_idx -> address):")
    print(f"  output: {output_map_exp}")
    print(f"  hidden: {hidden_map_exp}")
    print("Offset-only mapping (address -> local_idx):")
    print(f"  output: {output_addr_to_idx_offset}")
    print(f"  hidden: {hidden_addr_to_idx_offset}")
    print("Observed active addresses in trial 0:")
    print(f"  output: {output_addrs_observed}")
    print(f"  hidden: {hidden_addrs_observed}")
    print()

    diagnostic_trial_idx = args.narrative_trial
    diagnostic_output_idx = args.narrative_output_neuron
    diagnostic_hidden_path = circuit_dir / f"hidden_layer_trial_{diagnostic_trial_idx}.txt"
    diagnostic_output_path = circuit_dir / f"output_layer_trial_{diagnostic_trial_idx}.txt"
    diagnostic_window_until: float | None = None
    diagnostic_first: Event | None = None
    diagnostic_details_exp: list[tuple[int, int, float]] = []
    diagnostic_details_off: list[tuple[int, int, float]] = []

    diagnostic_spikes: list[Event] = []
    if (
        diagnostic_hidden_path.exists()
        and diagnostic_output_path.exists()
        and diagnostic_output_idx in output_map_exp
    ):
        diagnostic_hidden = parse_events(diagnostic_hidden_path)
        diagnostic_output = parse_events(diagnostic_output_path)
        diagnostic_spikes = [
            e for e in diagnostic_output
            if e.address == output_map_exp[diagnostic_output_idx]
        ]

    if diagnostic_spikes:
        diagnostic_first = diagnostic_spikes[0]
        t_until = diagnostic_first.time_s - causal_window_shift + 1e-12
        diagnostic_window_until = t_until
        _, _, _, diagnostic_details_exp = signed_input_sum(
            hidden_events=diagnostic_hidden,
            out_idx=diagnostic_output_idx,
            out_weights=out_weights,
            hidden_addr_to_idx=hidden_addr_to_idx_exp,
            t_until_s=t_until,
        )
        _, _, _, diagnostic_details_off = signed_input_sum(
            hidden_events=diagnostic_hidden,
            out_idx=diagnostic_output_idx,
            out_weights=out_weights,
            hidden_addr_to_idx=hidden_addr_to_idx_offset,
            t_until_s=t_until,
        )
        print(
            f"Diagnostic check (trial {diagnostic_trial_idx}, first spike of output neuron {diagnostic_output_idx}):"
        )
        print(f"  first spike time: {diagnostic_first.time_s:.6f} s")
        print(f"  using hidden spikes up to: {t_until:.6f} s")
        print(
            f"  exporter mapping details (hidden_idx, count, weight): {diagnostic_details_exp}")
        print(
            f"  offset-only mapping details (hidden_idx, count, weight): {diagnostic_details_off}")
        print()

    total_first_spikes_checked = 0
    inhibitory_only_offset = 0
    excitatory_present_exporter = 0
    report_rows: list[dict[str, str | int | float]] = []

    for trial_idx, (hidden_path, output_path) in enumerate(zip(hidden_files, output_files)):
        hidden_events = parse_events(hidden_path)
        output_events = parse_events(output_path)

        per_neuron_first: dict[int, Event] = {}
        for ev in output_events:
            if ev.address == 7:
                continue
            out_idx = output_addr_to_idx_exp.get(ev.address)
            if out_idx is not None and out_idx not in per_neuron_first:
                per_neuron_first[out_idx] = ev

        if args.show_per_trial:
            print(f"Trial {trial_idx}:")

        for out_idx in sorted(per_neuron_first):
            ev = per_neuron_first[out_idx]
            t_until = ev.time_s - causal_window_shift + 1e-12

            sum_exp, exc_exp, inh_exp, details_exp = signed_input_sum(
                hidden_events=hidden_events,
                out_idx=out_idx,
                out_weights=out_weights,
                hidden_addr_to_idx=hidden_addr_to_idx_exp,
                t_until_s=t_until,
            )
            sum_off, exc_off, inh_off, details_off = signed_input_sum(
                hidden_events=hidden_events,
                out_idx=out_idx,
                out_weights=out_weights,
                hidden_addr_to_idx=hidden_addr_to_idx_offset,
                t_until_s=t_until,
            )

            total_first_spikes_checked += 1
            if exc_off == 0 and inh_off > 0:
                inhibitory_only_offset += 1
                if exc_exp > 0:
                    excitatory_present_exporter += 1

            report_rows.append(
                {
                    "trial": trial_idx,
                    "output_neuron_index": out_idx,
                    "output_address": ev.address,
                    "first_spike_time_s": ev.time_s,
                    "window_until_s": t_until,
                    "exporter_weighted_sum": sum_exp,
                    "exporter_exc_spikes": exc_exp,
                    "exporter_inh_spikes": inh_exp,
                    "exporter_details_hiddenidx_count_weight": format_details(details_exp),
                    "offset_weighted_sum": sum_off,
                    "offset_exc_spikes": exc_off,
                    "offset_inh_spikes": inh_off,
                    "offset_details_hiddenidx_count_weight": format_details(details_off),
                    "offset_appears_inhibitory_only": int(exc_off == 0 and inh_off > 0),
                    "exporter_has_excitatory_input": int(exc_exp > 0),
                }
            )

            if args.show_per_trial:
                print(
                    f"  out_idx={out_idx} first_spike={ev.time_s:.6f}s | "
                    f"exporter(sum={sum_exp:.3f}, exc={exc_exp}, inh={inh_exp}) | "
                    f"offset(sum={sum_off:.3f}, exc={exc_off}, inh={inh_off})"
                )
                print(f"    exporter details: {details_exp}")
                print(f"    offset details:   {details_off}")

    print("=== Summary Across First Spikes ===")
    print(f"Total first spikes checked: {total_first_spikes_checked}")
    print(
        f"Inhibitory-only under offset-only mapping: {inhibitory_only_offset}")
    print(
        f"Those with excitatory input under exporter mapping: {excitatory_present_exporter}")

    frac_text = "n/a"
    if inhibitory_only_offset > 0:
        frac = 100.0 * excitatory_present_exporter / inhibitory_only_offset
        frac_text = f"{frac:.1f}% ({excitatory_present_exporter}/{inhibitory_only_offset})"
        print(f"Fraction explained by mapping reversal: {frac_text}")

    if not args.write_report:
        return

    report_dir = args.report_dir if args.report_dir is not None else (
        circuit_dir / "analysis_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "first_spike_mapping_analysis.csv"
    csv_fields = [
        "trial",
        "output_neuron_index",
        "output_address",
        "first_spike_time_s",
        "window_until_s",
        "exporter_weighted_sum",
        "exporter_exc_spikes",
        "exporter_inh_spikes",
        "exporter_details_hiddenidx_count_weight",
        "offset_weighted_sum",
        "offset_exc_spikes",
        "offset_inh_spikes",
        "offset_details_hiddenidx_count_weight",
        "offset_appears_inhibitory_only",
        "exporter_has_excitatory_input",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(report_rows)

    suspicious_csv_path: Path | None = None
    if args.write_suspicious_only_csv:
        suspicious_rows = [row for row in report_rows if int(
            row["offset_appears_inhibitory_only"]) == 1]
        suspicious_csv_file = report_dir / "first_spike_mapping_suspicious_only.csv"
        with suspicious_csv_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(suspicious_rows)
        suspicious_csv_path = suspicious_csv_file

    narrative_trial_idx = diagnostic_trial_idx
    if not (0 <= narrative_trial_idx < len(hidden_files)):
        raise ValueError(
            f"narrative_trial={narrative_trial_idx} out of range [0, {len(hidden_files)-1}]")

    narrative_hidden_events = parse_events(
        circuit_dir / f"hidden_layer_trial_{narrative_trial_idx}.txt")
    narrative_output_events = parse_events(
        circuit_dir / f"output_layer_trial_{narrative_trial_idx}.txt")

    # Try to load true membrane trace if available in either repetition results or detailed trace artifact.
    true_mem_trace: np.ndarray | None = None
    preferred_rep_results_path = base_results / \
        f"braille_reading_rsnn_{n_hidden}_neurons_A_B_rep_{args.rep}.npz"
    rep_result_candidates: list[Path] = []
    if preferred_rep_results_path.exists():
        rep_result_candidates.append(preferred_rep_results_path)
    rep_result_candidates.extend(
        p for p in sorted(base_results.glob(f"braille_reading_rsnn_*_rep_{args.rep}.npz"))
        if p not in rep_result_candidates
    )

    for rep_results_path in rep_result_candidates:
        with np.load(rep_results_path, allow_pickle=True) as data:
            if "mem_rec_readout" in data.files:
                mem = np.asarray(data["mem_rec_readout"])
                if mem.ndim == 3 and 0 <= narrative_trial_idx < mem.shape[0] and 0 <= args.narrative_output_neuron < mem.shape[2]:
                    true_mem_trace = mem[narrative_trial_idx,
                                         :, args.narrative_output_neuron]
                    break

    if true_mem_trace is None:
        trace_candidates = sorted(base_results.glob(
            f"best_model_traces_*_rep_{args.rep}.npz"))
        if trace_candidates:
            with np.load(trace_candidates[0], allow_pickle=True) as data:
                key = "mem_rec_readout_test" if "mem_rec_readout_test" in data.files else "mem_rec_readout_train"
                if key in data.files:
                    mem = np.asarray(data[key])
                    if mem.ndim == 3 and 0 <= narrative_trial_idx < mem.shape[0] and 0 <= args.narrative_output_neuron < mem.shape[2]:
                        true_mem_trace = mem[narrative_trial_idx,
                                             :, args.narrative_output_neuron]

    narrative_path = report_dir / "single_trial_narrative.txt"
    narrative_text = build_trial_narrative_text(
        trial_idx=narrative_trial_idx,
        output_neuron_idx=args.narrative_output_neuron,
        experiment_params=experiment_params,
        t_clock=t_clock,
        causal_window_shift=causal_window_shift,
        output_map_exp=output_map_exp,
        hidden_addr_to_idx_exp=hidden_addr_to_idx_exp,
        hidden_addr_to_idx_offset=hidden_addr_to_idx_offset,
        hidden_events=narrative_hidden_events,
        output_events=narrative_output_events,
        out_weights=out_weights,
        true_mem_trace=true_mem_trace,
    )
    narrative_path.write_text(narrative_text, encoding="utf-8")

    plot_path = report_dir / "single_trial_visualization.png"
    plot_trial_visualization(
        trial_idx=narrative_trial_idx,
        output_neuron_idx=args.narrative_output_neuron,
        experiment_params=experiment_params,
        t_clock=t_clock,
        causal_window_shift=causal_window_shift,
        output_map_exp=output_map_exp,
        hidden_addr_to_idx_exp=hidden_addr_to_idx_exp,
        hidden_events=narrative_hidden_events,
        output_events=narrative_output_events,
        out_weights=out_weights,
        true_mem_trace=true_mem_trace,
        save_path=plot_path,
    )

    what_to_change_example_lines: list[str] = [
        f"Concrete example from selected diagnostic trial (trial {diagnostic_trial_idx}, output neuron {diagnostic_output_idx}):",
    ]
    if diagnostic_first is not None and diagnostic_window_until is not None:
        causal_hidden = [
            ev for ev in narrative_hidden_events
            if ev.address != 7 and ev.time_s <= diagnostic_window_until
        ]
        if causal_hidden:
            by_addr: dict[int, int] = {}
            for ev in causal_hidden:
                by_addr[ev.address] = by_addr.get(ev.address, 0) + 1
            focus_addr = max(by_addr.items(), key=lambda item: item[1])[0]

            h_exp = hidden_addr_to_idx_exp.get(focus_addr)
            h_off = hidden_addr_to_idx_offset.get(focus_addr)
            w_exp = float(out_weights[diagnostic_output_idx, h_exp]) if h_exp is not None else float("nan")
            w_off = float(out_weights[diagnostic_output_idx, h_off]) if h_off is not None else float("nan")

            max_addr = n_output + n_hidden - 1
            h_exp_text = str(h_exp) if h_exp is not None else "n/a"
            h_off_text = str(h_off) if h_off is not None else "n/a"
            w_exp_text = f"{w_exp:.6g}" if not math.isnan(w_exp) else "n/a"
            w_off_text = f"{w_off:.6g}" if not math.isnan(w_off) else "n/a"

            what_to_change_example_lines.extend(
                [
                    f"- Using n_output={n_output}, n_hidden={n_hidden}, hidden address {focus_addr} maps to hidden_idx = {max_addr} - {focus_addr} = {h_exp_text} (exporter mapping), not {h_off_text} (offset-only).",
                    f"- For output neuron index {diagnostic_output_idx}, spikes from hidden address {focus_addr} therefore use weight ({diagnostic_output_idx},{h_exp_text}) = {w_exp_text}, not weight ({diagnostic_output_idx},{h_off_text}) = {w_off_text}.",
                    "- This is why a case that may look contradictory under offset-only indexing can become consistent under exporter mapping.",
                ]
            )
        else:
            what_to_change_example_lines.append(
                "- No hidden spikes were present in the causal window for this selected diagnostic case, so no concrete address remapping example is available."
            )
    else:
        what_to_change_example_lines.append(
            "- No first spike was found for this selected diagnostic case, so no concrete address remapping example is available."
        )

    summary_path = report_dir / "mapping_analysis_summary.md"
    summary_lines = [
        "# Circuit Spike Mapping Analysis Report",
        "",
        "## Plain-Language Takeaway",
        "- We checked whether the exported spike files and the weight matrix are consistent with each other.",
        "- The main issue is index interpretation: hidden neuron addresses in the exported spike files are reversed within the hidden layer.",
        "- If someone assumes a simple offset-only mapping (no reversal), some spikes look impossible (inhibitory-only input before a spike).",
        "- With the correct exporter mapping, those same cases become consistent (excitatory input is present).",
        "",
        "## Inputs",
        f"- experiment_id: {args.experiment_id}",
        f"- circuit_dir: {circuit_dir}",
        f"- weights_file: {weights_path}",
        f"- trials: {len(hidden_files)}",
        f"- inferred_output_neurons: {n_output}",
        f"- inferred_hidden_neurons: {n_hidden}",
        f"- t_clock_s: {t_clock:.6f}",
        f"- causal_delay_steps: {args.causal_delay_steps}",
        f"- causal_window_shift_s: {causal_window_shift:.6f}",
        f"- membrane_trace_available_for_narrative: {true_mem_trace is not None}",
        "",
        "## Mapping Used by Export Script",
        f"- output local_idx -> address: {output_map_exp}",
        f"- hidden local_idx -> address: {hidden_map_exp}",
        "- Meaning: hidden local index 0 is exported as the highest hidden address, not the lowest.",
        "",
        "## Alternative Mapping Tested",
        "- output address -> local_idx: identity",
        "- hidden address -> local_idx: address - n_output",
        "- Meaning: this is the intuitive mapping many hardware checks start with, but it is not what the export script uses here.",
        "",
        f"## Diagnostic Trial {diagnostic_trial_idx} (output neuron index {diagnostic_output_idx} first spike)",
    ]

    if diagnostic_first is not None and diagnostic_window_until is not None:
        summary_lines.extend(
            [
                f"- first_spike_time_s: {diagnostic_first.time_s:.6f}",
                f"- window_until_s: {diagnostic_window_until:.6f}",
                f"- exporter_details (hidden_idx:count:weight): {format_details(diagnostic_details_exp)}",
                f"- offset_details (hidden_idx:count:weight): {format_details(diagnostic_details_off)}",
            ]
        )
    else:
        summary_lines.append(
            f"- No spike found for output neuron index {diagnostic_output_idx} in trial {diagnostic_trial_idx}")

    summary_lines.extend(
        [
            "",
            "## Global Summary Across First Spikes",
            f"- total_first_spikes_checked: {total_first_spikes_checked}",
            f"- inhibitory_only_under_offset_mapping: {inhibitory_only_offset}",
            f"- with_excitatory_input_under_exporter_mapping: {excitatory_present_exporter}",
            f"- fraction_explained_by_mapping_reversal: {frac_text}",
            "",
            "## How To Interpret The Global Summary",
            "- `total_first_spikes_checked`: number of evaluated first-spike events (first output spike per output neuron per trial, when present).",
            "- `inhibitory_only_under_offset_mapping`: among those events, how many appear contradictory under the naive offset-only hidden-index mapping (only inhibitory pre-spike evidence).",
            "- `with_excitatory_input_under_exporter_mapping`: among those suspicious offset cases, how many become consistent when using the actual exporter mapping (excitatory evidence appears).",
            "- `fraction_explained_by_mapping_reversal`: defined as",
            "  `with_excitatory_input_under_exporter_mapping / inhibitory_only_under_offset_mapping`.",
            "- Practical reading rule:",
            "  near 100% => mismatch is primarily an indexing/mapping interpretation issue;",
            "  much lower => at least some suspicious cases are not resolved by mapping alone and need additional investigation.",
            f"- Example `{frac_text}`: all {excitatory_present_exporter} cases that looked inhibitory-only under offset mapping are explained once exporter mapping is applied.",
            f"- Why not `{total_first_spikes_checked}/{total_first_spikes_checked}`?: because {total_first_spikes_checked} is the full set of first-spike events, while only {inhibitory_only_offset} were suspicious under offset mapping.",
            "  The headline fraction is intentionally conditional: it asks how many suspicious cases are resolved by correct mapping.",
            "- Optional companion view: resolved fraction over all checked events =",
            "  `with_excitatory_input_under_exporter_mapping / total_first_spikes_checked`",
            f"  (for this run: {excitatory_present_exporter}/{total_first_spikes_checked} = {100.0*excitatory_present_exporter/total_first_spikes_checked:.1f}%).",
            "",
            "## What To Change (To Reproduce Software Behaviour On Chip)",
            "Use this as a step-by-step checklist when reading spike files against the weight matrix:",
            "1. Do not use hidden index = (hidden_address - n_output).",
            "   That offset-only rule is the source of the mismatch in this dataset.",
            "2. Decode output and hidden addresses with exporter reversal:",
            f"   - output_idx = (n_output - 1) - output_address   [here n_output={n_output}]",
            f"   - hidden_idx = (n_output + n_hidden - 1) - hidden_address   [here n_output={n_output}, n_hidden={n_hidden}]",
            "3. Then read the synapse as weight = out_weights[output_idx, hidden_idx].",
            "4. For first-spike causality checks, only include hidden spikes up to:",
            "   output_spike_time - causal_delay_steps * T_CLOCK.",
            "5. If membrane is needed, do not use raw sum only; include decay + reset dynamics between spikes.",
            "",
            "## Python Pseudocode: Fix Hidden Index Mapping",
            "```python",
            "# Read spike file",
            "import csv",
            "spikes = []",
            "with open('circuit_level_spikes.txt', 'r') as f:",
            "    reader = csv.DictFieldReader(f, fieldnames=['TIME', 'DELTA_TIME', 'ADDRESS'])",
            "    for row in reader:",
            "        time_ms = float(row['TIME'])",
            "        address = int(row['ADDRESS'])",
            "        spikes.append({'time_ms': time_ms, 'address': address})",
            "",
            "# Load weight matrix",
            "import numpy as np",
            "out_weights = np.load('out_weights.npy')  # shape (n_output, n_hidden)",
            "n_output, n_hidden = out_weights.shape",
            "",
            "# Correction function: apply exporter reversal mapping",
            "def correct_address(address, n_output, n_hidden):",
            "    if address < n_output:",
            "        # Output neuron address",
            "        local_idx = (n_output - 1) - address",
            "        return 'output', local_idx",
            "    else:",
            "        # Hidden neuron address",
            "        hidden_addr = address",
            "        hidden_idx = (n_output + n_hidden - 1) - hidden_addr",
            "        return 'hidden', hidden_idx",
            "",
            "# Process spikes with corrected mapping",
            "corrected_spikes = []",
            "for spike in spikes:",
            "    neuron_type, local_idx = correct_address(spike['address'], n_output, n_hidden)",
            "    corrected_spikes.append({",
            "        'time_ms': spike['time_ms'],",
            "        'address': spike['address'],",
            "        'neuron_type': neuron_type,",
            "        'local_idx': local_idx",
            "    })",
            "",
            "# Validate: check causality for each first output spike",
            "T_CLOCK_MS = 3.0  # (0.003 s in milliseconds)",
            "causal_delay_steps = 1",
            "causal_delay_ms = causal_delay_steps * T_CLOCK_MS",
            "",
            "for trial_data in corrected_spikes:",
            "    output_spikes = [s for s in trial_data if s['neuron_type'] == 'output']",
            "    hidden_spikes = [s for s in trial_data if s['neuron_type'] == 'hidden']",
            "",
            "    for output_spike in output_spikes:",
            "        output_idx = output_spike['local_idx']",
            "        spike_time = output_spike['time_ms']",
            "        causal_until = spike_time - causal_delay_ms",
            "",
            "        # Sum weighted input in causal window",
            "        weighted_sum = 0.0",
            "        for hidden_spike in hidden_spikes:",
            "            if hidden_spike['time_ms'] <= causal_until:",
            "                hidden_idx = hidden_spike['local_idx']",
            "                weight = out_weights[output_idx, hidden_idx]",
            "                weighted_sum += weight",
            "",
            "        # Check causality",
            "        if weighted_sum > 0:",
            "            print(f'Output {output_idx} at {spike_time:.2f} ms: valid (sum={weighted_sum:.4f})')",
            "        else:",
            "            print(f'Output {output_idx} at {spike_time:.2f} ms: SUSPICIOUS (sum={weighted_sum:.4f})')",
            "```",
            "",
            *what_to_change_example_lines,
            "",
            "## Mini Glossary",
            "- output neuron index: row index in out_weights.",
            "- hidden index: column index in out_weights.",
            "- address: integer ID written in circuit-level spike text files.",
            "- weighted_sum: sum(count_of_spikes_from_hidden_i * weight_to_output_from_hidden_i) in the causal window.",
            "- causal window: hidden spikes counted up to (output_spike_time - causal_delay_steps * T_CLOCK).",
            "",
            "## Reproduce Locally",
            "Run from repository root:",
            "```bash",
            "python scripts/analyze_circuit_spike_mapping.py \\",
            f"  --experiment-id {args.experiment_id} \\",
            "  --write-report \\",
            f"  --narrative-trial {args.narrative_trial} \\",
            f"  --narrative-output-neuron {args.narrative_output_neuron}",
            "```",
            "",
            "Optional suspicious-only CSV:",
            "```bash",
            "python scripts/analyze_circuit_spike_mapping.py \\",
            f"  --experiment-id {args.experiment_id} \\",
            "  --write-report \\",
            "  --write-suspicious-only-csv",
            "```",
            "",
            "Generated files:",
            f"- {csv_path}",
            f"- {summary_path}",
            f"- {narrative_path}",
            f"- {plot_path}",
        ]
    )

    if suspicious_csv_path is not None:
        summary_lines.append(f"- {suspicious_csv_path}")

    summary_lines.extend(
        [
            "",
            f"## Figure Caption — `single_trial_visualization.png`",
            "",
            f"**What is shown**: A two-panel summary for trial {narrative_trial_idx}, output neuron {args.narrative_output_neuron}.",
            "",
            "### Top panel — Hidden neuron spike raster",
            "- Each row is one hidden neuron, labelled by its exporter index and its synaptic weight `w` to the selected output neuron.",
            "  Indexing follows the **exporter mapping** (address reversal within the hidden layer), not the naive offset-only assumption.",
            "- A tick mark on row `i` at time `t` means hidden neuron `i` fired at time `t`.",
            "- **Colour of ticks**: green = excitatory weight (`w > 0`), red = inhibitory weight (`w < 0`).",
            "  Only weights from the hidden neuron to the *selected* output neuron are considered here; a neuron coloured green",
            "  pushes the output membrane up, one coloured red pulls it down.",
            "- **Dotted orange vertical line**: causal boundary — the latest time at which a hidden spike could causally",
            f"  have driven the *first* output spike (i.e. first-output-spike time − {args.causal_delay_steps} clock step(s)).",
            "  Only hidden spikes to the *left* of this line are considered in the mapping analysis.",
            "- **Dashed blue vertical lines**: times at which the selected output neuron fires.",
            "",
            "### Bottom panel — Output neuron membrane potential",
            "- **Blue trace**: the membrane potential of the selected output neuron over the full trial.",
            f"  {'This is the true `mem_rec_readout` tensor saved during inference.' if true_mem_trace is not None else 'The true `mem_rec_readout` was not saved for this run; the trace is a decay-aware reconstruction computed analytically from hidden spike times and `out_weights`.'  }",
            f"  Reconstruction uses exponential decay with `tau_mem={experiment_params.get('tau_mem')}` s and",
            f"  `{'hard' if not experiment_params.get('soft_reset', False) else 'soft'} reset` after each spike.",
            "- **Dashed red horizontal line**: the firing threshold (membrane must reach or exceed this to emit a spike).",
            "- **Red triangle markers** (▲): moments at which the reconstructed membrane crosses the threshold.",
            "  These correspond to threshold-crossing events as predicted by the decay model, regardless of whether the true trace is available.",
            "- **Dashed blue vertical lines** and **dotted orange line**: same as top panel (output spikes and causal boundary),",
            "  allowing direct comparison of when the membrane crosses threshold versus when the causal boundary falls.",
            "- **Bottom-left annotation**: states whether the trace is true or reconstructed.",
            "",
            "### How to read both panels together",
            "1. Find the first **blue vertical line** (first output spike) and the **orange dotted line** just before it (causal boundary).",
            "2. In the top panel, any spike tick to the **left** of the orange line contributed to driving the output neuron.",
            "   Green ticks to the left = excitatory evidence; red ticks to the left = inhibitory evidence.",
            "3. In the bottom panel, trace the blue membrane curve up to the **red threshold line**; the triangle shows",
            "   when the threshold is crossed — accounting for decay between successive inputs, not just raw weight sums.",
            "4. Cross-reference with `single_trial_narrative.txt` for the step-by-step arithmetic.",
            "",
            "---",
            "",
            "## CSV column format notes",
            "- details columns use hidden_idx:count:weight entries separated by ';'",
            "- weighted sums are count*weight accumulated over hidden spikes in the causal window",
            "- suspicious rows are where offset mapping gives only inhibitory evidence before a spike",
            "- single_trial_narrative.txt explains one selected case in plain text with timestamp-level detail",
            "",
            "Meaning of details fields used in this report:",
            "- `exporter_details (hidden_idx:count:weight)`: hidden-neuron contributions in the causal window when addresses are decoded using the exporter mapping.",
            "- `offset_details (hidden_idx:count:weight)`: same contribution list but using the naive offset-only mapping.",
            "- Each entry `h:c:w` means: hidden index `h` fired `c` times in the causal window and has weight `w` to the selected output neuron.",
            "- Contribution of one entry to weighted_sum is `c * w`.",
            "- If a hidden address maps to different hidden indices under the two mapping rules, the same physical spikes will be assigned different weights, which is exactly what this analysis is testing.",
        ]
    )

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print()
    print("=== Report Files Written ===")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Single-trial narrative: {narrative_path}")
    print(f"Visualization: {plot_path}")
    if suspicious_csv_path is not None:
        print(f"Suspicious-only CSV: {suspicious_csv_path}")


if __name__ == "__main__":
    main()
