import numpy as np
import json
from pathlib import Path
from datetime import date


experiment_id = "20260115_0833_exploration/20260225_092341"
results_file_name = "braille_reading_rsnn_5_neurons_A_B_rep_1.npz"

header_name = "Tactile Braille Reading"
header_data = date.today().isoformat()

results_file = Path(f"./results/{experiment_id}/{results_file_name}")
params_file = results_file.parent / "experiment_parameters.json"
output_dir = Path(f"./results/{experiment_id}/circuit_level_spk")


def load_spike_records(npz_path: Path) -> list[tuple[str, np.ndarray]]:
    """Load spike recordings and return named layers in output->hidden order.

    Expects arrays with shape [samples, time, neurons].
    """
    with np.load(npz_path, allow_pickle=True) as data:
        required = ["spk_rec_readout", "spk_rec_hidden"]
        for key in required:
            if key not in data.files:
                raise KeyError(f"Missing '{key}' in {npz_path}")

        readout = np.asarray(data["spk_rec_readout"])
        hidden = np.asarray(data["spk_rec_hidden"])

    if readout.ndim != 3 or hidden.ndim != 3:
        raise ValueError("Spike arrays must be 3D with shape [samples, time, neurons].")

    return [("output_layer", readout), ("hidden_layer", hidden)]


def neuron_id_map(layer_sizes: list[int]) -> list[dict[int, int]]:
    """Build neuron index -> address map with reverse order per layer.

    Layer order must be output->hidden->... .
    """
    address_maps: list[dict[int, int]] = []
    next_addr = 0
    for n_neurons in layer_sizes:
        mapping = {idx: next_addr + (n_neurons - 1 - idx) for idx in range(n_neurons)}
        address_maps.append(mapping)
        next_addr += n_neurons
    return address_maps


def load_experiment_parameters(json_path: Path) -> dict:
    """Load experiment parameter JSON."""
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_timing_header_values(params: dict, n_steps: int) -> dict[str, float]:
    """Infer timing-related header values from experiment parameters.

    Inference rules:
    - T_CLOCK: `time_step` if available, else `time_bin_size * 1e-3`.
    - T_REF_MIN/MAX: `ref_per_timesteps * T_CLOCK`.
    - T_LEAK_MIN/MAX: use `tau_mem` and `tau_mem_rec` (min/max).
    - T_END: derived from prepared spike data as `n_steps * T_CLOCK`.
    """
    t_clock = float(params.get("time_step", params.get("time_bin_size", 1) * 1.0e-3))

    ref_steps_raw = params.get("ref_per_timesteps")
    if ref_steps_raw is None:
        ref_per_timesteps = 0
    else:
        ref_per_timesteps = int(ref_steps_raw)
    t_ref = max(0, ref_per_timesteps) * t_clock

    tau_mem = float(params.get("tau_mem", 0.06))
    tau_mem_rec = float(params.get("tau_mem_rec", tau_mem))
    t_leak_min = min(tau_mem, tau_mem_rec)
    t_leak_max = max(tau_mem, tau_mem_rec)

    t_end = float(n_steps) * t_clock

    return {
        "t_clock": t_clock,
        "t_ref_min": t_ref,
        "t_ref_max": t_ref,
        "t_leak_min": t_leak_min,
        "t_leak_max": t_leak_max,
        "t_end": t_end,
    }


def write_circuit_spike_file(
    layer_spk: np.ndarray,
    layer_address_map: dict[int, int],
    sample_idx: int,
    file_path: Path,
    name_value: str,
    data_value: str,
    header_timing: dict[str, float],
) -> None:
    """Write spike events for one source layer with required header and clock events."""
    n_samples = layer_spk.shape[0]
    n_steps = layer_spk.shape[1]
    t_clock = float(header_timing["t_clock"])

    if sample_idx < 0 or sample_idx >= n_samples:
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {n_samples - 1}]")

    tab2 = "\t\t"
    lines = [
        f"NAME{tab2}{name_value}",
        f"DATA{tab2}{data_value}",
        f"C_RATIO{tab2}20~21",
        f"T_REF_MIN{tab2}{header_timing['t_ref_min']:.1e}",
        f"T_REF_MAX{tab2}{header_timing['t_ref_max']:.1e}",
        f"T_LEAK_MIN{tab2}{header_timing['t_leak_min']:.1e}",
        f"T_LEAK_MAX{tab2}{header_timing['t_leak_max']:.1e}",
        f"T_CLOCK{tab2}{header_timing['t_clock']:.1e}",
        f"T_END{tab2}{header_timing['t_end']:.1f}",
        "",
        f"TIME{tab2}DELTA{tab2}ADDRESS",
    ]

    last_event_time = 0.0
    for step in range(n_steps):
        current_t = step * t_clock
        delta_clock = current_t - last_event_time
        lines.append(f"{current_t:.6f}{tab2}{delta_clock:.6e}{tab2}{7}")
        last_event_time = current_t

        active_neurons = np.where(layer_spk[sample_idx, step] > 0)[0]
        for neuron_idx in active_neurons:
            address = layer_address_map[int(neuron_idx)]
            lines.append(f"{current_t:.6f}{tab2}{0.0:.6e}{tab2}{address}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    layers_spikes = load_spike_records(results_file)

    params = load_experiment_parameters(params_file)
    n_steps = layers_spikes[0][1].shape[1]
    header_timing = infer_timing_header_values(params=params, n_steps=n_steps)

    layer_sizes = [layer.shape[2] for _, layer in layers_spikes]
    address_maps = neuron_id_map(layer_sizes)

    n_trials = layers_spikes[0][1].shape[0]
    for _, layer in layers_spikes[1:]:
        if layer.shape[0] != n_trials:
            raise ValueError("All layer spike arrays must have the same number of trials.")

    output_dir.mkdir(parents=True, exist_ok=True)
    created_count = 0
    for trial_idx in range(n_trials):
        trial_name = f"{header_name} Trial {trial_idx}"
        for layer_idx, (layer_name, layer_spk) in enumerate(layers_spikes):
            out_file = output_dir / f"{layer_name}_trial_{trial_idx}.txt"
            write_circuit_spike_file(
                layer_spk=layer_spk,
                layer_address_map=address_maps[layer_idx],
                sample_idx=trial_idx,
                file_path=out_file,
                name_value=trial_name,
                data_value=header_data,
                header_timing=header_timing,
            )
            created_count += 1

    print(f"Created {created_count} files in: {output_dir}")