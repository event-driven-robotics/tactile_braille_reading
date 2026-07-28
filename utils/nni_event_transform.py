"""
mechanoreceptor_encoding_hpo.py

Parallel-safe tactile encoding pipeline for NNI hyperparameter optimization.

The encoder can be used in two ways:

1. From Python:
       config = EncodingConfig(...)
       encoded_data, output_path = run_encoding(config)

2. As an NNI trial:
       python mechanoreceptor_encoding_hpo.py

When executed as an NNI trial, the script:
    - calls nni.get_next_parameter() once;
    - creates an EncodingConfig;
    - runs the selected encoder;
    - evaluates the encoded output;
    - reports the final metric to NNI.
"""

from __future__ import annotations

import math
import pickle as pkl
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import nni
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.neuron_models import (
    AdExLIF_neuron,
    CuBaLIF_neuron,
    IZ_neuron,
    LIF_neuron,
    MN_neuron,
    RA_I_mechanoreceptor,
    SA_II_mechanoreceptor,
)


# ---------------------------------------------------------------------------
# Constant model registries
#
# These are safe as module-level constants because trials do not modify them.
# ---------------------------------------------------------------------------

NEURON_MODEL_CLASSES = {
    "AdExLIF_neuron": AdExLIF_neuron,
    "CuBaLIF_neuron": CuBaLIF_neuron,
    "IZ_neuron": IZ_neuron,
    "LIF_neuron": LIF_neuron,
    "MN_neuron": MN_neuron,
}

ENCODING_CHOICES = {
    "mechanoreceptor",
    "sigma_delta",
    "neuron_model",
}

UPSAMPLE_STRATEGIES = {
    "linear",
    "hold",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodingConfig:
    """
    Complete configuration for one encoding run.

    The dataclass is frozen so a trial cannot accidentally modify its
    configuration while it is running.
    """

    encoding_type: str = "neuron_model"

    neuron_model: str = "MN_neuron"
    neuron_params: Mapping[str, Any] = field(default_factory=dict)

    upsample_strategy: str = "linear"
    upsample_dt_s: float = 0.001

    mechanoreceptor_threshold: float = 2.0
    mechanoreceptor_max_frequency: float = 150.0
    mechanoreceptor_refractory_period: float = 0.003

    sigma_delta_threshold: float = 2.0
    sigma_delta_refractory_period: float = 0.003

    data_path: str | Path = "data/100Hz"
    data_files: tuple[str, ...] = (
        "data_braille_letters_0.0.pkl",
        "data_braille_letters_0.000125.pkl",
    )

    output_path: str | Path | None = None
    letters: tuple[str, ...] | None = None
    nb_trials: int | None = None

    # Disable this during most HPO trials to avoid unnecessary disk I/O.
    save_output: bool = False


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_config(config: EncodingConfig) -> None:
    """Validate one encoding configuration before processing data."""

    if config.encoding_type not in ENCODING_CHOICES:
        raise ValueError(
            f"Unknown encoding_type {config.encoding_type!r}. "
            f"Expected one of {sorted(ENCODING_CHOICES)}."
        )

    if config.encoding_type == "neuron_model":
        if config.neuron_model not in NEURON_MODEL_CLASSES:
            raise ValueError(
                f"Unknown neuron_model {config.neuron_model!r}. "
                f"Expected one of {sorted(NEURON_MODEL_CLASSES)}."
            )

        if config.upsample_strategy not in UPSAMPLE_STRATEGIES:
            raise ValueError(
                f"Unknown upsample_strategy "
                f"{config.upsample_strategy!r}. "
                f"Expected one of {sorted(UPSAMPLE_STRATEGIES)}."
            )

        if config.upsample_dt_s <= 0:
            raise ValueError("upsample_dt_s must be positive.")

    if config.mechanoreceptor_threshold < 0:
        raise ValueError(
            "mechanoreceptor_threshold cannot be negative."
        )

    if config.mechanoreceptor_max_frequency <= 0:
        raise ValueError(
            "mechanoreceptor_max_frequency must be positive."
        )

    if config.mechanoreceptor_refractory_period < 0:
        raise ValueError(
            "mechanoreceptor_refractory_period cannot be negative."
        )

    if config.sigma_delta_threshold < 0:
        raise ValueError(
            "sigma_delta_threshold cannot be negative."
        )

    if config.sigma_delta_refractory_period < 0:
        raise ValueError(
            "sigma_delta_refractory_period cannot be negative."
        )

    if config.nb_trials is not None and config.nb_trials <= 0:
        raise ValueError(
            "nb_trials must be a positive integer or None."
        )

    if not config.data_files:
        raise ValueError("data_files cannot be empty.")

    data_directory = Path(config.data_path)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_directory}"
        )

    missing_files = [
        filename
        for filename in config.data_files
        if not (data_directory / filename).is_file()
    ]

    if missing_files:
        formatted = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Input files not found in {data_directory}: "
            f"{formatted}"
        )


# ---------------------------------------------------------------------------
# Input utilities
# ---------------------------------------------------------------------------

def normalize_letters(
    letters: Sequence[str] | None,
) -> list[str] | None:
    """
    Normalize optional letter selections.

    Accepts values such as:
        ("A", "B")
        ("A,B", "C")
        ["a", "b"]
    """

    if not letters:
        return None

    normalized: list[str] = []

    for letter in letters:
        for token in str(letter).split(","):
            token = token.strip()

            if token:
                normalized.append(token.upper())

    return normalized or None


def load_trials(
    input_data_path: str | Path,
    input_files: Sequence[str],
    show_progress: bool = True,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load all trials from all requested pickle files."""

    data_directory = Path(input_data_path)
    trials: list[tuple[str, np.ndarray, np.ndarray]] = []

    file_iterator = tqdm(
        input_files,
        desc="Loading files",
        disable=not show_progress,
    )

    for filename in file_iterator:
        input_path = data_directory / filename

        with input_path.open("rb") as file:
            data = pkl.load(file)

        required_keys = {
            "letter",
            "taxel_data",
            "timestamp",
        }

        missing_keys = required_keys.difference(data.keys())

        if missing_keys:
            raise KeyError(
                f"{input_path} is missing required fields: "
                f"{sorted(missing_keys)}"
            )

        letter_list = data["letter"].values
        taxels_list = data["taxel_data"].values
        timestamps_list = data["timestamp"].values

        for letter, taxels, timestamps in zip(
            letter_list,
            taxels_list,
            timestamps_list,
        ):
            taxels_array = np.asarray(taxels)
            timestamps_array = np.asarray(
                timestamps,
                dtype=float,
            ).reshape(-1)

            if taxels_array.ndim != 2:
                raise ValueError(
                    f"Expected taxel_data with shape "
                    f"(time, channels), got "
                    f"{taxels_array.shape} in {input_path}."
                )

            if taxels_array.shape[0] != timestamps_array.shape[0]:
                raise ValueError(
                    f"Taxel and timestamp lengths differ in "
                    f"{input_path}: "
                    f"{taxels_array.shape[0]} versus "
                    f"{timestamps_array.shape[0]}."
                )

            if timestamps_array.size == 0:
                continue

            trials.append(
                (
                    str(letter),
                    taxels_array,
                    timestamps_array,
                )
            )

    return trials


def filter_sorted_trials(
    trials: Sequence[
        tuple[str, np.ndarray, np.ndarray]
    ],
    letters: Sequence[str] | None = None,
    nb_trials: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Filter trials by letter and optionally keep only a fixed number
    of trials per letter.
    """

    selected_letters = set(
        normalize_letters(letters) or []
    )

    filtered: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = []

    per_letter_counts: dict[str, int] = {}

    for letter, taxels, timestamps in trials:
        normalized_letter = str(letter).upper()

        if (
            selected_letters
            and normalized_letter not in selected_letters
        ):
            continue

        if nb_trials is not None:
            current_count = per_letter_counts.get(
                normalized_letter,
                0,
            )

            if current_count >= nb_trials:
                continue

            per_letter_counts[normalized_letter] = (
                current_count + 1
            )

        filtered.append(
            (
                letter,
                taxels,
                timestamps,
            )
        )

    return filtered


def iterate_trials(
    input_data_path: str | Path,
    input_files: Sequence[str],
    letters: Sequence[str] | None = None,
    nb_trials: int | None = None,
    show_progress: bool = True,
):
    """
    Yield sorted and optionally filtered trials.

    Each yielded element is:
        letter, taxels, timestamps
    """

    trials = load_trials(
        input_data_path=input_data_path,
        input_files=input_files,
        show_progress=show_progress,
    )

    trials.sort(key=lambda item: item[0])

    filtered_trials = filter_sorted_trials(
        trials=trials,
        letters=letters,
        nb_trials=nb_trials,
    )

    if letters or nb_trials is not None:
        print(
            f"Selected {len(filtered_trials)}/"
            f"{len(trials)} trials "
            f"(letters={letters or 'all'}, "
            f"nb_trials="
            f"{nb_trials if nb_trials is not None else 'all'})."
        )

    trial_iterator = tqdm(
        filtered_trials,
        desc="Encoding trials",
        disable=not show_progress,
    )

    yield from trial_iterator


# ---------------------------------------------------------------------------
# General output utilities
# ---------------------------------------------------------------------------

def sort_output_dict_by_letter(
    out_dict: dict[str, Any],
    letter_key: str = "letter",
) -> dict[str, Any]:
    """Sort all per-trial list fields by their corresponding letter."""

    if (
        letter_key not in out_dict
        or len(out_dict[letter_key]) == 0
    ):
        return out_dict

    sorted_indices = np.argsort(
        out_dict[letter_key],
        kind="stable",
    )

    expected_length = len(out_dict[letter_key])

    for key, value in out_dict.items():
        if (
            isinstance(value, list)
            and len(value) == expected_length
        ):
            out_dict[key] = [
                value[index]
                for index in sorted_indices
            ]

    return out_dict


def add_subset_suffix(
    output_path: str | Path,
    letters: Sequence[str] | None,
    nb_trials: int | None,
) -> Path:
    """Add selected letters and trial count to a filename."""

    path = Path(output_path)

    if not letters and nb_trials is None:
        return path

    suffix_parts: list[str] = []

    normalized_letters = normalize_letters(letters)

    if normalized_letters:
        suffix_parts.append(
            "letters-" + "-".join(normalized_letters)
        )

    if nb_trials is not None:
        suffix_parts.append(f"trials-{nb_trials}")

    suffix_text = "_".join(suffix_parts)

    return path.with_name(
        f"{path.stem}_{suffix_text}{path.suffix}"
    )


def save_encoded_output(
    out_dict: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save encoded output atomically.

    A temporary UUID-based path is used so parallel trials never
    share the same temporary filename.
    """

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        f".{destination.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            pkl.dump(
                out_dict,
                file,
                protocol=pkl.HIGHEST_PROTOCOL,
            )

        temporary_path.replace(destination)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved encoded data to: {destination}")

    return destination


def update_trial_duration_bounds(
    timestamps: np.ndarray,
    longest_trial: float,
    shortest_trial: float,
) -> tuple[float, float]:
    """Update the minimum and maximum trial durations."""

    if timestamps.size == 0:
        return longest_trial, shortest_trial

    trial_duration = float(
        timestamps[-1] - timestamps[0]
    )

    longest_trial = max(
        longest_trial,
        trial_duration,
    )

    shortest_trial = min(
        shortest_trial,
        trial_duration,
    )

    return longest_trial, shortest_trial


# ---------------------------------------------------------------------------
# Neuron-model utilities
# ---------------------------------------------------------------------------

def build_neuron_model(
    model_name: str,
    nb_inputs: int,
    neuron_params: Mapping[str, Any] | None = None,
):
    """Construct one neuron model instance."""

    try:
        model_class = NEURON_MODEL_CLASSES[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown neuron model: {model_name!r}"
        ) from error

    return model_class(
        nb_inputs=nb_inputs,
        **dict(neuron_params or {}),
    )


def upsample_taxel_trial(
    timestamps: np.ndarray,
    taxels: np.ndarray,
    dt_s: float,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample one trial to a fixed time step.

    Supported strategies:
        linear
        hold
    """

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    ).reshape(-1)

    taxels = np.asarray(
        taxels,
        dtype=float,
    )

    if taxels.ndim != 2:
        raise ValueError(
            "taxels must have shape (time, channels)."
        )

    if timestamps.shape[0] != taxels.shape[0]:
        raise ValueError(
            "timestamps and taxels must have matching "
            "time dimensions."
        )

    if timestamps.shape[0] < 2:
        return timestamps, taxels

    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    if strategy not in UPSAMPLE_STRATEGIES:
        raise ValueError(
            f"Unsupported upsampling strategy "
            f"{strategy!r}."
        )

    order = np.argsort(
        timestamps,
        kind="stable",
    )

    timestamps = timestamps[order]
    taxels = taxels[order]

    # Keep the last sample for duplicate timestamps.
    keep_last_duplicate = np.concatenate(
        (
            timestamps[1:] != timestamps[:-1],
            [True],
        )
    )

    timestamps = timestamps[keep_last_duplicate]
    taxels = taxels[keep_last_duplicate]

    if timestamps.shape[0] < 2:
        return timestamps, taxels

    start_time = float(timestamps[0])
    end_time = float(timestamps[-1])

    number_of_steps = (
        int(
            np.floor(
                (end_time - start_time) / dt_s
            )
        )
        + 1
    )

    upsampled_timestamps = (
        start_time
        + np.arange(
            number_of_steps,
            dtype=float,
        )
        * dt_s
    )

    if strategy == "linear":
        upsampled_taxels = np.empty(
            (
                upsampled_timestamps.shape[0],
                taxels.shape[1],
            ),
            dtype=float,
        )

        for channel_index in range(
            taxels.shape[1]
        ):
            upsampled_taxels[:, channel_index] = (
                np.interp(
                    upsampled_timestamps,
                    timestamps,
                    taxels[:, channel_index],
                )
            )

    else:
        sample_indices = np.searchsorted(
            timestamps,
            upsampled_timestamps,
            side="right",
        ) - 1

        sample_indices = np.clip(
            sample_indices,
            0,
            timestamps.shape[0] - 1,
        )

        upsampled_taxels = taxels[sample_indices]

    upsampled_timestamps = np.round(
        upsampled_timestamps,
        decimals=8,
    )

    return (
        upsampled_timestamps,
        upsampled_taxels,
    )


# ---------------------------------------------------------------------------
# Individual encoding branches
# ---------------------------------------------------------------------------

def encode_mechanoreceptor(
    config: EncodingConfig,
    letters: Sequence[str] | None,
    show_progress: bool,
) -> tuple[dict[str, Any], float, float]:
    """Run FA-I and SA-II mechanoreceptor encoding."""

    print(
        "Encoding tactile data with mechanoreceptor models."
    )

    output: dict[str, Any] = {
        "letter": [],
        "taxel_data": [],
        "timestamps": [],
        "fa_spikes": [],
        "sa_spikes": [],
    }

    longest_trial = -math.inf
    shortest_trial = math.inf

    for letter, taxels, timestamps in iterate_trials(
        input_data_path=config.data_path,
        input_files=config.data_files,
        letters=letters,
        nb_trials=config.nb_trials,
        show_progress=show_progress,
    ):
        longest_trial, shortest_trial = (
            update_trial_duration_bounds(
                timestamps=timestamps,
                longest_trial=longest_trial,
                shortest_trial=shortest_trial,
            )
        )

        fa_spikes: list[Any] = []
        sa_spikes: list[Any] = []

        fa_encoder = RA_I_mechanoreceptor(
            taxel_values=taxels[0],
            fa_threshold=(
                config.mechanoreceptor_threshold
            ),
            ref_period=(
                config.mechanoreceptor_refractory_period
            ),
        )

        sa_encoder = SA_II_mechanoreceptor(
            channels=taxels.shape[1],
            max_frequ=(
                config.mechanoreceptor_max_frequency
            ),
            ref_period=(
                config.mechanoreceptor_refractory_period
            ),
        )

        last_time = timestamps[0]

        for time_index in range(
            1,
            taxels.shape[0],
        ):
            current_time = timestamps[time_index]
            current_taxels = taxels[time_index]

            fa_result = fa_encoder.step(
                taxel_values=current_taxels,
                current_time=current_time,
                last_time=last_time,
            )

            sa_result = sa_encoder.step(
                taxel_values=current_taxels,
                current_time=current_time,
                last_time=last_time,
            )

            if fa_result.shape[0] > 0:
                fa_spikes.extend(fa_result)

            if sa_result.shape[0] > 0:
                sa_spikes.extend(sa_result)

            last_time = current_time

        output["letter"].append(letter)
        output["taxel_data"].append(taxels)
        output["timestamps"].append(timestamps)

        output["fa_spikes"].append(
            np.asarray(
                fa_spikes,
                dtype=float,
            )
        )

        output["sa_spikes"].append(
            np.asarray(
                sa_spikes,
                dtype=float,
            )
        )

    return (
        output,
        longest_trial,
        shortest_trial,
    )


def encode_sigma_delta(
    config: EncodingConfig,
    letters: Sequence[str] | None,
    show_progress: bool,
) -> tuple[dict[str, Any], float, float]:
    """Run ON/OFF sigma-delta encoding."""

    print("Encoding tactile data with sigma-delta.")

    output: dict[str, Any] = {
        "letter": [],
        "taxel_data": [],
        "timestamps": [],
        "ON_spikes": [],
        "OFF_spikes": [],
    }

    longest_trial = -math.inf
    shortest_trial = math.inf

    for letter, taxels, timestamps in iterate_trials(
        input_data_path=config.data_path,
        input_files=config.data_files,
        letters=letters,
        nb_trials=config.nb_trials,
        show_progress=show_progress,
    ):
        longest_trial, shortest_trial = (
            update_trial_duration_bounds(
                timestamps=timestamps,
                longest_trial=longest_trial,
                shortest_trial=shortest_trial,
            )
        )

        on_spikes: list[Any] = []
        off_spikes: list[Any] = []

        sigma_delta_encoder = RA_I_mechanoreceptor(
            taxel_values=taxels[0],
            fa_threshold=config.sigma_delta_threshold,
            ref_period=(
                config.sigma_delta_refractory_period
            ),
        )

        last_time = timestamps[0]
        taxel_differences = np.diff(
            taxels,
            axis=0,
        )

        for time_index in range(
            1,
            taxels.shape[0],
        ):
            current_time = timestamps[time_index]
            current_taxels = taxels[time_index]

            results = sigma_delta_encoder.step(
                taxel_values=current_taxels,
                current_time=current_time,
                last_time=last_time,
            )

            if results.shape[0] > 0:
                positive_channels = np.where(
                    taxel_differences[
                        time_index - 1
                    ] > 0
                )[0]

                negative_channels = np.where(
                    taxel_differences[
                        time_index - 1
                    ] < 0
                )[0]

                event_rows = np.asarray(results)

                if (
                    event_rows.ndim == 2
                    and event_rows.shape[1] >= 2
                ):
                    event_channels = (
                        event_rows[:, 1].astype(int)
                    )

                    if positive_channels.size > 0:
                        matching_on_events = event_rows[
                            np.isin(
                                event_channels,
                                positive_channels,
                            )
                        ]

                        if matching_on_events.shape[0] > 0:
                            on_spikes.extend(
                                matching_on_events
                            )

                    if negative_channels.size > 0:
                        matching_off_events = event_rows[
                            np.isin(
                                event_channels,
                                negative_channels,
                            )
                        ]

                        if matching_off_events.shape[0] > 0:
                            off_spikes.extend(
                                matching_off_events
                            )

            last_time = current_time

        output["letter"].append(letter)
        output["taxel_data"].append(taxels)
        output["timestamps"].append(timestamps)

        output["ON_spikes"].append(
            np.asarray(
                on_spikes,
                dtype=float,
            )
        )

        output["OFF_spikes"].append(
            np.asarray(
                off_spikes,
                dtype=float,
            )
        )

    return (
        output,
        longest_trial,
        shortest_trial,
    )


def encode_neuron_model(
    config: EncodingConfig,
    letters: Sequence[str] | None,
    show_progress: bool,
) -> tuple[dict[str, Any], float, float]:
    """Run encoding with the selected neuron model."""

    print(
        f"Encoding tactile data with "
        f"{config.neuron_model}."
    )

    output: dict[str, Any] = {
        "letter": [],
        "taxel_data": [],
        "timestamps": [],
        "spikes": [],
    }

    longest_trial = -math.inf
    shortest_trial = math.inf

    for letter, taxels, timestamps in iterate_trials(
        input_data_path=config.data_path,
        input_files=config.data_files,
        letters=letters,
        nb_trials=config.nb_trials,
        show_progress=show_progress,
    ):
        # Construct a new model for every trial. This resets
        # stateful neuron variables between tactile trials.
        neuron_model = build_neuron_model(
            model_name=config.neuron_model,
            nb_inputs=taxels.shape[1],
            neuron_params=config.neuron_params,
        )

        timestamps, normalized_taxels = (
            upsample_taxel_trial(
                timestamps=timestamps,
                taxels=taxels,
                dt_s=config.upsample_dt_s,
                strategy=config.upsample_strategy,
            )
        )

        # Normalize tactile values to [0, 1].
        normalized_taxels = (
            normalized_taxels / 255.0
        )

        longest_trial, shortest_trial = (
            update_trial_duration_bounds(
                timestamps=timestamps,
                longest_trial=longest_trial,
                shortest_trial=shortest_trial,
            )
        )

        spikes: list[Any] = []

        for time_index in range(
            1,
            normalized_taxels.shape[0],
        ):
            current_taxels = torch.as_tensor(
                normalized_taxels[time_index],
                dtype=torch.get_default_dtype(),
            )

            with torch.no_grad():
                neuron_response = neuron_model.forward(
                    input=current_taxels
                )

            if neuron_response is None:
                continue

            response_array = (
                neuron_response
                .detach()
                .cpu()
                .numpy()
            )

            # Preserve compatibility with models returning
            # either (1, channels) or (channels,).
            response_array = np.asarray(
                response_array
            ).squeeze()

            if response_array.ndim != 1:
                raise ValueError(
                    f"{config.neuron_model} returned an "
                    f"unexpected shape: "
                    f"{response_array.shape}"
                )

            firing_channels = np.flatnonzero(
                response_array > 0
            )

            if firing_channels.size > 0:
                event_times = np.full(
                    firing_channels.shape,
                    timestamps[time_index],
                    dtype=float,
                )

                aer_spikes = np.column_stack(
                    (
                        event_times,
                        firing_channels.astype(float),
                    )
                )

                spikes.extend(aer_spikes)

        output["letter"].append(letter)
        output["taxel_data"].append(
            normalized_taxels
        )
        output["timestamps"].append(timestamps)
        output["spikes"].append(
            np.asarray(
                spikes,
                dtype=float,
            )
        )

    return (
        output,
        longest_trial,
        shortest_trial,
    )


# ---------------------------------------------------------------------------
# Main callable encoder API
# ---------------------------------------------------------------------------

def make_default_output_path(
    config: EncodingConfig,
    letters: Sequence[str] | None,
) -> Path:
    """Construct a readable default output path."""

    output_directory = Path(config.data_path)

    if config.encoding_type == "mechanoreceptor":
        filename = (
            "mechanoreceptor"
            f"_thr-{config.mechanoreceptor_threshold:g}"
            f"_freq-{config.mechanoreceptor_max_frequency:g}"
            f"_ref-{config.mechanoreceptor_refractory_period:g}"
            ".pkl"
        )

    elif config.encoding_type == "sigma_delta":
        filename = (
            "sigma_delta"
            f"_thr-{config.sigma_delta_threshold:g}"
            f"_ref-{config.sigma_delta_refractory_period:g}"
            ".pkl"
        )

    else:
        filename = (
            f"{config.neuron_model}"
            f"_dt-{config.upsample_dt_s:g}"
            f"_{config.upsample_strategy}"
            ".pkl"
        )

    return add_subset_suffix(
        output_directory / filename,
        letters=letters,
        nb_trials=config.nb_trials,
    )


def run_encoding(
    config: EncodingConfig,
    *,
    show_progress: bool = True,
) -> tuple[dict[str, Any], Path | None]:
    """
    Run one complete encoding job.

    This function does not modify module-level state, making it
    appropriate for concurrent NNI trial processes.

    Returns:
        encoded_data:
            Dictionary containing encoded trials.

        output_path:
            Saved file path, or None when save_output=False.
    """

    validate_config(config)

    letters = normalize_letters(config.letters)

    if config.encoding_type == "mechanoreceptor":
        output, longest_trial, shortest_trial = (
            encode_mechanoreceptor(
                config=config,
                letters=letters,
                show_progress=show_progress,
            )
        )

    elif config.encoding_type == "sigma_delta":
        output, longest_trial, shortest_trial = (
            encode_sigma_delta(
                config=config,
                letters=letters,
                show_progress=show_progress,
            )
        )

    else:
        output, longest_trial, shortest_trial = (
            encode_neuron_model(
                config=config,
                letters=letters,
                show_progress=show_progress,
            )
        )

    if not output["letter"]:
        raise ValueError(
            "No trials were processed. Check data_files, "
            "letters, and nb_trials."
        )

    print(
        f"Longest trial duration: "
        f"{longest_trial:.8g} seconds"
    )

    print(
        f"Shortest trial duration: "
        f"{shortest_trial:.8g} seconds"
    )

    output = sort_output_dict_by_letter(output)

    if not config.save_output:
        return output, None

    output_path = (
        Path(config.output_path)
        if config.output_path is not None
        else make_default_output_path(
            config=config,
            letters=letters,
        )
    )

    saved_path = save_encoded_output(
        out_dict=output,
        output_path=output_path,
    )

    return output, saved_path


# ---------------------------------------------------------------------------
# NNI integration
# ---------------------------------------------------------------------------

def as_bool(
    value: Any,
    *,
    default: bool,
) -> bool:
    """Convert an NNI parameter to bool."""

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"Cannot interpret {value!r} as a boolean."
    )


def as_optional_letters(
    value: Any,
) -> tuple[str, ...] | None:
    """
    Convert an NNI parameter into selected letters.

    Supported examples:
        None
        "A,B,C"
        ["A", "B", "C"]
    """

    if value is None:
        return None

    if isinstance(value, str):
        normalized = normalize_letters([value])
    elif isinstance(value, Sequence):
        normalized = normalize_letters(
            [str(item) for item in value]
        )
    else:
        normalized = normalize_letters([str(value)])

    if not normalized:
        return None

    return tuple(normalized)


def as_optional_positive_int(
    value: Any,
) -> int | None:
    """Convert an optional NNI value to a positive integer."""

    if value is None:
        return None

    integer_value = int(value)

    if integer_value <= 0:
        raise ValueError(
            "Expected a positive integer."
        )

    return integer_value


def extract_neuron_params(
    nni_params: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Extract neuron constructor parameters.

    Any NNI search-space parameter beginning with `neuron__`
    is forwarded to the selected neuron constructor.

    Example:
        neuron__a
        neuron__b
        neuron__threshold

    becomes:
        {
            "a": ...,
            "b": ...,
            "threshold": ...,
        }
    """

    prefix = "neuron__"

    return {
        key[len(prefix):]: value
        for key, value in nni_params.items()
        if key.startswith(prefix)
    }


def config_from_nni_parameters(
    nni_params: Mapping[str, Any],
) -> EncodingConfig:
    """
    Convert parameters received from NNI into EncodingConfig.

    Parameters absent from the NNI search space keep their
    EncodingConfig defaults.
    """

    defaults = EncodingConfig()

    data_files_value = nni_params.get(
        "data_files",
        defaults.data_files,
    )

    if isinstance(data_files_value, str):
        data_files = tuple(
            token.strip()
            for token in data_files_value.split(",")
            if token.strip()
        )
    else:
        data_files = tuple(
            str(item)
            for item in data_files_value
        )

    output_path_value = nni_params.get(
        "output_path",
        defaults.output_path,
    )

    return EncodingConfig(
        encoding_type=str(
            nni_params.get(
                "encoding_type",
                defaults.encoding_type,
            )
        ),
        neuron_model=str(
            nni_params.get(
                "neuron_model",
                defaults.neuron_model,
            )
        ),
        neuron_params=extract_neuron_params(
            nni_params
        ),
        upsample_strategy=str(
            nni_params.get(
                "upsample_strategy",
                defaults.upsample_strategy,
            )
        ),
        upsample_dt_s=float(
            nni_params.get(
                "upsample_dt_s",
                defaults.upsample_dt_s,
            )
        ),
        mechanoreceptor_threshold=float(
            nni_params.get(
                "mechanoreceptor_threshold",
                defaults.mechanoreceptor_threshold,
            )
        ),
        mechanoreceptor_max_frequency=float(
            nni_params.get(
                "mechanoreceptor_max_frequency",
                defaults.mechanoreceptor_max_frequency,
            )
        ),
        mechanoreceptor_refractory_period=float(
            nni_params.get(
                "mechanoreceptor_refractory_period",
                defaults.mechanoreceptor_refractory_period,
            )
        ),
        sigma_delta_threshold=float(
            nni_params.get(
                "sigma_delta_threshold",
                defaults.sigma_delta_threshold,
            )
        ),
        sigma_delta_refractory_period=float(
            nni_params.get(
                "sigma_delta_refractory_period",
                defaults.sigma_delta_refractory_period,
            )
        ),
        data_path=str(
            nni_params.get(
                "data_path",
                defaults.data_path,
            )
        ),
        data_files=data_files,
        output_path=(
            None
            if output_path_value in {
                None,
                "",
            }
            else str(output_path_value)
        ),
        letters=as_optional_letters(
            nni_params.get(
                "letters",
                defaults.letters,
            )
        ),
        nb_trials=as_optional_positive_int(
            nni_params.get(
                "nb_trials",
                defaults.nb_trials,
            )
        ),
        save_output=as_bool(
            nni_params.get(
                "save_output",
                defaults.save_output,
            ),
            default=defaults.save_output,
        ),
    )


# ---------------------------------------------------------------------------
# HPO objective
# ---------------------------------------------------------------------------

def count_trial_spikes(
    encoded_data: Mapping[str, Any],
    trial_index: int,
) -> int:
    """Count all encoded spikes in one trial."""

    if "spikes" in encoded_data:
        return int(
            len(encoded_data["spikes"][trial_index])
        )

    if (
        "fa_spikes" in encoded_data
        and "sa_spikes" in encoded_data
    ):
        return int(
            len(encoded_data["fa_spikes"][trial_index])
            + len(encoded_data["sa_spikes"][trial_index])
        )

    if (
        "ON_spikes" in encoded_data
        and "OFF_spikes" in encoded_data
    ):
        return int(
            len(encoded_data["ON_spikes"][trial_index])
            + len(encoded_data["OFF_spikes"][trial_index])
        )

    raise KeyError(
        "No recognized spike fields were found."
    )


def evaluate_encoded_data(
    encoded_data: Mapping[str, Any],
) -> float:
    """
    Return the metric optimized by NNI.

    CURRENT DEFAULT:
        Mean number of spikes per encoded trial.

    Replace this function with your actual downstream objective,
    for example:
        - validation accuracy;
        - F1 score;
        - reconstruction loss;
        - classification loss;
        - energy/sparsity trade-off;
        - accuracy penalized by spike count.

    Ensure that the NNI experiment's optimize_mode matches the
    metric:
        maximize for accuracy or F1;
        minimize for loss or error.
    """

    number_of_trials = len(encoded_data["letter"])

    if number_of_trials == 0:
        raise ValueError(
            "Cannot evaluate an empty encoding."
        )

    spike_counts = [
        count_trial_spikes(
            encoded_data,
            trial_index,
        )
        for trial_index in range(number_of_trials)
    ]

    return float(np.mean(spike_counts))


def run_event_encoding() -> float:
    """Execute one NNI trial."""

    # Each NNI trial should call this exactly once.
    nni_params = nni.get_next_parameter()

    print(f"NNI parameters: {nni_params}")

    config = config_from_nni_parameters(
        nni_params
    )

    print(f"Encoding configuration: {config}")

    encoded_data, output_path = run_encoding(
        config=config,
        show_progress=True,
    )

    metric = evaluate_encoded_data(
        encoded_data
    )

    if not np.isfinite(metric):
        raise ValueError(
            f"The objective returned a non-finite "
            f"value: {metric}"
        )

    print(f"Final objective: {metric}")

    if output_path is not None:
        print(f"Encoded output: {output_path}")

    nni.report_final_result(metric)

    return metric


if __name__ == "__main__":
    run_event_encoding()