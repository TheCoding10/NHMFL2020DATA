"""Reusable signal-processing tools for condensed matter experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
from scipy import signal

from experiment_loader import Experiment


LOGGER = logging.getLogger(__name__)

STRUCTURAL_PREFIXES = (
    "Timestamp_",
    "RuO_T_",
    "Cx_T_",
    "Cernox_T_",
    "Cernox_R_",
    "Cernox_",
    "DR_Temp_",
    "Field_",
    "Angle_",
    "Steps_",
)
TEMPERATURE_PREFIXES = ("RuO_T_", "Cx_T_", "Cernox_T_", "DR_Temp_", "Cernox_")
SIGNAL_PREFERENCE_PREFIXES = ("Counter_", "FQ1_", "FQ2_")


@dataclass(frozen=True)
class SignalColumns:
    """Detected signal and coordinate columns for an experiment."""

    signal: str
    timestamp: str | None
    magnetic_field: str | None
    temperature: str | None
    angle: str | None
    measurements: tuple[str, ...]


@dataclass(frozen=True)
class SeriesResult:
    """A one-dimensional analysis result tied to a source signal."""

    signal_column: str
    values: np.ndarray
    x_values: np.ndarray | None = None


@dataclass(frozen=True)
class ExtremaResult:
    """Local maxima and minima indices and values."""

    signal_column: str
    maxima_indices: np.ndarray
    maxima_values: np.ndarray
    minima_indices: np.ndarray
    minima_values: np.ndarray
    x_values: np.ndarray | None = None


@dataclass(frozen=True)
class PeakDetectionResult:
    """Peak detection output from scipy.signal.find_peaks."""

    signal_column: str
    peak_indices: np.ndarray
    peak_values: np.ndarray
    properties: dict[str, np.ndarray]
    x_values: np.ndarray | None = None


@dataclass(frozen=True)
class OscillationResult:
    """Summary of oscillatory structure in a signal."""

    signal_column: str
    detected: bool
    zero_crossing_count: int
    peak_count: int
    trough_count: int
    estimated_period: float | None
    estimated_frequency: float | None
    amplitude_estimate: float


@dataclass(frozen=True)
class FFTResult:
    """One-sided FFT amplitudes for a real-valued signal."""

    signal_column: str
    frequencies: np.ndarray
    amplitudes: np.ndarray
    fft_values: np.ndarray
    sample_spacing: float
    signal_length: int


@dataclass(frozen=True)
class PowerSpectrumResult:
    """Power spectrum derived from the real-valued FFT."""

    signal_column: str
    frequencies: np.ndarray
    power: np.ndarray
    sample_spacing: float
    signal_length: int


@dataclass(frozen=True)
class NoiseStatistics:
    """Basic descriptive noise statistics."""

    signal_column: str
    mean: float
    standard_deviation: float
    variance: float
    root_mean_square: float
    median_absolute_deviation: float
    peak_to_peak: float
    signal_to_noise_ratio: float | None


def detect_signal_columns(experiment: Experiment) -> SignalColumns:
    """Detect measurement signal and coordinate columns from an experiment."""

    columns = tuple(str(column) for column in experiment.dataframe.columns)
    measurements = tuple(
        column for column in columns if not column.startswith(STRUCTURAL_PREFIXES)
    )
    signal_column = choose_signal_column(columns, measurements)

    return SignalColumns(
        signal=signal_column,
        timestamp=_find_optional_prefixed_column(columns, "Timestamp_"),
        magnetic_field=_find_optional_prefixed_column(columns, "Field_"),
        temperature=_find_optional_prefixed_column_by_options(
            columns,
            TEMPERATURE_PREFIXES,
        ),
        angle=_find_optional_prefixed_column(columns, "Angle_"),
        measurements=measurements,
    )


def choose_signal_column(
    columns: Iterable[str],
    measurements: Iterable[str] = (),
) -> str:
    """Choose the best measurement signal column, preferring Counter."""

    column_tuple = tuple(columns)
    for prefix in SIGNAL_PREFERENCE_PREFIXES:
        column = _find_optional_prefixed_column(column_tuple, prefix)
        if column is not None:
            return column

    measurement_tuple = tuple(measurements)
    if measurement_tuple:
        return measurement_tuple[0]

    for column in column_tuple:
        if not column.startswith(STRUCTURAL_PREFIXES):
            return column

    raise ValueError("no measurement signal column was found")


def get_signal(
    experiment: Experiment,
    signal_column: str | None = None,
) -> tuple[str, np.ndarray]:
    """Return a finite numeric signal array from an experiment."""

    column = signal_column or detect_signal_columns(experiment).signal
    if column not in experiment.dataframe.columns:
        raise ValueError(f"signal column is missing: {column}")

    values = experiment.dataframe[column].to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        LOGGER.warning(
            "Dropping %d non-finite value(s) from %s",
            int((~finite_mask).sum()),
            column,
        )
        values = values[finite_mask]

    if values.size == 0:
        raise ValueError(f"signal column has no finite values: {column}")

    return column, values


def get_x_values(
    experiment: Experiment,
    x_column: str | None = None,
) -> np.ndarray | None:
    """Return finite x values from a requested or detected timestamp column."""

    column = x_column or detect_signal_columns(experiment).timestamp
    if column is None:
        return None
    if column not in experiment.dataframe.columns:
        raise ValueError(f"x column is missing: {column}")

    values = experiment.dataframe[column].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    return finite_values if finite_values.size else None


def smooth_signal(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    window_length: int = 51,
    polyorder: int = 3,
) -> SeriesResult:
    """Smooth a signal with a Savitzky-Golay filter."""

    column, values = get_signal(experiment, signal_column)
    window = _valid_savgol_window(len(values), window_length, polyorder)
    smoothed = signal.savgol_filter(values, window_length=window, polyorder=polyorder)
    return SeriesResult(signal_column=column, values=smoothed)


def baseline_correction(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    polynomial_order: int = 2,
) -> SeriesResult:
    """Subtract a polynomial baseline from a signal."""

    column, values = get_signal(experiment, signal_column)
    if len(values) <= polynomial_order:
        raise ValueError("signal is too short for requested baseline order")

    x_values = np.arange(len(values), dtype=float)
    coefficients = np.polyfit(x_values, values, deg=polynomial_order)
    baseline = np.polyval(coefficients, x_values)
    corrected = values - baseline
    return SeriesResult(signal_column=column, values=corrected, x_values=x_values)


def numerical_derivative(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    x_column: str | None = None,
) -> SeriesResult:
    """Compute the numerical derivative d(signal)/d(x)."""

    column, values = get_signal(experiment, signal_column)
    x_values = get_x_values(experiment, x_column)
    if x_values is not None and len(x_values) != len(values):
        LOGGER.warning("Ignoring x values with mismatched signal length")
        x_values = None

    if x_values is not None:
        derivative = np.gradient(values, x_values)
    else:
        derivative = np.gradient(values)

    return SeriesResult(signal_column=column, values=derivative, x_values=x_values)


def detect_peaks(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> PeakDetectionResult:
    """Detect peaks using scipy.signal.find_peaks."""

    column, values = get_signal(experiment, signal_column)
    peak_indices, properties = signal.find_peaks(
        values,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    return PeakDetectionResult(
        signal_column=column,
        peak_indices=peak_indices,
        peak_values=values[peak_indices],
        properties={key: np.asarray(value) for key, value in properties.items()},
    )


def local_extrema(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    order: int = 1,
) -> ExtremaResult:
    """Find local maxima and minima with scipy.signal.argrelextrema."""

    column, values = get_signal(experiment, signal_column)
    maxima_indices = signal.argrelextrema(values, np.greater, order=order)[0]
    minima_indices = signal.argrelextrema(values, np.less, order=order)[0]

    return ExtremaResult(
        signal_column=column,
        maxima_indices=maxima_indices,
        maxima_values=values[maxima_indices],
        minima_indices=minima_indices,
        minima_values=values[minima_indices],
    )


def detect_oscillations(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    prominence: float | None = None,
    min_peak_count: int = 3,
) -> OscillationResult:
    """Estimate whether a signal contains repeated oscillations."""

    column, values = get_signal(experiment, signal_column)
    centered = values - np.nanmean(values)
    zero_crossing_count = int(np.count_nonzero(np.diff(np.signbit(centered))))

    peak_indices, _peak_props = signal.find_peaks(centered, prominence=prominence)
    trough_indices, _trough_props = signal.find_peaks(-centered, prominence=prominence)
    extrema_indices = np.sort(np.concatenate([peak_indices, trough_indices]))
    estimated_period = _median_spacing(extrema_indices)
    estimated_frequency = None
    if estimated_period is not None and estimated_period > 0:
        estimated_frequency = 1.0 / (2.0 * estimated_period)

    amplitude_estimate = float(
        (np.nanpercentile(values, 95) - np.nanpercentile(values, 5)) / 2.0
    )
    detected = (
        len(peak_indices) >= min_peak_count
        and len(trough_indices) >= min_peak_count
        and zero_crossing_count >= min_peak_count * 2
    )

    return OscillationResult(
        signal_column=column,
        detected=detected,
        zero_crossing_count=zero_crossing_count,
        peak_count=len(peak_indices),
        trough_count=len(trough_indices),
        estimated_period=estimated_period,
        estimated_frequency=estimated_frequency,
        amplitude_estimate=amplitude_estimate,
    )


def fast_fourier_transform(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    sample_spacing: float | None = None,
    detrend: bool = True,
) -> FFTResult:
    """Compute a one-sided real FFT for the selected signal."""

    column, values = get_signal(experiment, signal_column)
    spacing = sample_spacing or _estimate_sample_spacing(experiment)
    processed = signal.detrend(values) if detrend and len(values) > 1 else values

    fft_values = np.fft.rfft(processed)
    frequencies = np.fft.rfftfreq(len(processed), d=spacing)
    amplitudes = np.abs(fft_values) / len(processed)

    return FFTResult(
        signal_column=column,
        frequencies=frequencies,
        amplitudes=amplitudes,
        fft_values=fft_values,
        sample_spacing=spacing,
        signal_length=len(processed),
    )


def power_spectrum(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    sample_spacing: float | None = None,
    detrend: bool = True,
) -> PowerSpectrumResult:
    """Compute the one-sided power spectrum for the selected signal."""

    fft_result = fast_fourier_transform(
        experiment,
        signal_column=signal_column,
        sample_spacing=sample_spacing,
        detrend=detrend,
    )
    power = np.abs(fft_result.fft_values) ** 2 / fft_result.signal_length
    return PowerSpectrumResult(
        signal_column=fft_result.signal_column,
        frequencies=fft_result.frequencies,
        power=power,
        sample_spacing=fft_result.sample_spacing,
        signal_length=fft_result.signal_length,
    )


def noise_statistics(
    experiment: Experiment,
    signal_column: str | None = None,
    *,
    baseline_order: int | None = 1,
) -> NoiseStatistics:
    """Calculate basic noise statistics for a signal or baseline residual."""

    column, values = get_signal(experiment, signal_column)
    if baseline_order is not None and len(values) > baseline_order:
        residual = baseline_correction(
            experiment,
            signal_column=column,
            polynomial_order=baseline_order,
        ).values
    else:
        residual = values - np.nanmean(values)

    mean = float(np.nanmean(residual))
    ddof = 1 if residual.size > 1 else 0
    standard_deviation = float(np.nanstd(residual, ddof=ddof))
    variance = float(np.nanvar(residual, ddof=ddof))
    root_mean_square = float(np.sqrt(np.nanmean(residual**2)))
    median = np.nanmedian(residual)
    median_absolute_deviation = float(np.nanmedian(np.abs(residual - median)))
    peak_to_peak = float(np.nanmax(residual) - np.nanmin(residual))
    noise_floor = standard_deviation
    signal_level = float(np.nanmean(np.abs(values)))
    signal_to_noise_ratio = None
    if noise_floor > 0:
        signal_to_noise_ratio = signal_level / noise_floor

    return NoiseStatistics(
        signal_column=column,
        mean=mean,
        standard_deviation=standard_deviation,
        variance=variance,
        root_mean_square=root_mean_square,
        median_absolute_deviation=median_absolute_deviation,
        peak_to_peak=peak_to_peak,
        signal_to_noise_ratio=signal_to_noise_ratio,
    )


def _find_optional_prefixed_column(
    columns: Iterable[str],
    prefix: str,
) -> str | None:
    for column in columns:
        if column.startswith(prefix):
            return column
    return None


def _find_optional_prefixed_column_by_options(
    columns: Iterable[str],
    prefixes: tuple[str, ...],
) -> str | None:
    for prefix in prefixes:
        column = _find_optional_prefixed_column(columns, prefix)
        if column is not None:
            return column
    return None


def _valid_savgol_window(
    signal_length: int,
    requested_window: int,
    polyorder: int,
) -> int:
    if signal_length <= polyorder:
        raise ValueError("signal is too short for Savitzky-Golay smoothing")

    window = min(requested_window, signal_length)
    if window % 2 == 0:
        window -= 1
    minimum_window = polyorder + 2
    if minimum_window % 2 == 0:
        minimum_window += 1
    if window < minimum_window:
        window = minimum_window
    if window > signal_length:
        raise ValueError("signal is too short for requested smoothing parameters")
    return window


def _estimate_sample_spacing(experiment: Experiment) -> float:
    columns = detect_signal_columns(experiment)
    if columns.timestamp is None:
        return 1.0

    timestamps = experiment.dataframe[columns.timestamp].to_numpy(dtype=float)
    finite_timestamps = timestamps[np.isfinite(timestamps)]
    if finite_timestamps.size < 2:
        return 1.0

    deltas = np.diff(finite_timestamps)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        return 1.0
    return float(np.median(positive_deltas))


def _median_spacing(indices: np.ndarray) -> float | None:
    if len(indices) < 2:
        return None
    return float(np.median(np.diff(indices)))
