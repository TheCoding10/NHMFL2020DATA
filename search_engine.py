"""Search and ranking utilities for condensed matter experiment datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any

from experiment_loader import Experiment
from signal_processing import (
    detect_oscillations,
    detect_peaks,
    detect_signal_columns,
    noise_statistics,
)


LOGGER = logging.getLogger(__name__)

ExperimentPredicate = Callable[[Experiment], bool]
ScoreFunction = Callable[[Experiment], float]


@dataclass(frozen=True)
class SignalStatistics:
    """Important signal statistics included in search results."""

    signal_column: str | None = None
    peak_count: int | None = None
    oscillation_detected: bool | None = None
    oscillation_strength: float | None = None
    signal_to_noise_ratio: float | None = None
    min_magnetic_field: float | None = None
    max_magnetic_field: float | None = None
    min_temperature: float | None = None
    max_temperature: float | None = None
    min_angle: float | None = None
    max_angle: float | None = None
    extra: dict[str, float | int | str | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """One experiment returned by a search query."""

    filename: str
    metadata: dict[str, str]
    matching_score: float
    reason: str
    signal_statistics: SignalStatistics
    experiment: Experiment


def find_by_temperature_range(
    experiments: Iterable[Experiment],
    min_temperature: float,
    max_temperature: float,
) -> list[SearchResult]:
    """Find experiments whose temperature range overlaps a requested range."""

    return _filter_with_reason(
        experiments,
        predicate=lambda experiment: _range_overlaps(
            experiment.min_temperature,
            experiment.max_temperature,
            min_temperature,
            max_temperature,
        ),
        reason=(
            f"temperature overlaps [{min_temperature}, {max_temperature}]"
        ),
        score_function=lambda experiment: _range_overlap_score(
            experiment.min_temperature,
            experiment.max_temperature,
            min_temperature,
            max_temperature,
        ),
    )


def find_by_magnetic_field_range(
    experiments: Iterable[Experiment],
    min_field: float,
    max_field: float,
) -> list[SearchResult]:
    """Find experiments whose magnetic-field range overlaps a requested range."""

    return _filter_with_reason(
        experiments,
        predicate=lambda experiment: _range_overlaps(
            experiment.min_magnetic_field,
            experiment.max_magnetic_field,
            min_field,
            max_field,
        ),
        reason=f"magnetic field overlaps [{min_field}, {max_field}]",
        score_function=lambda experiment: _range_overlap_score(
            experiment.min_magnetic_field,
            experiment.max_magnetic_field,
            min_field,
            max_field,
        ),
    )


def find_with_more_than_n_peaks(
    experiments: Iterable[Experiment],
    minimum_peak_count: int,
    *,
    prominence: float | None = None,
    distance: int | None = None,
) -> list[SearchResult]:
    """Find experiments whose detected signal has more than N peaks."""

    results: list[SearchResult] = []
    for experiment in experiments:
        try:
            peaks = detect_peaks(
                experiment,
                prominence=prominence,
                distance=distance,
            )
        except ValueError as error:
            LOGGER.warning(
                "Skipping peak search for %s: %s",
                experiment.filename,
                error,
            )
            continue

        peak_count = len(peaks.peak_indices)
        if peak_count <= minimum_peak_count:
            continue

        statistics = build_signal_statistics(
            experiment,
            peak_count=peak_count,
        )
        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=float(peak_count),
                reason=f"detected {peak_count} peaks > {minimum_peak_count}",
                signal_statistics=statistics,
                experiment=experiment,
            )
        )

    return sort_results(results)


def find_containing_oscillations(
    experiments: Iterable[Experiment],
    *,
    prominence: float | None = None,
    min_peak_count: int = 3,
) -> list[SearchResult]:
    """Find experiments whose signal appears oscillatory."""

    results: list[SearchResult] = []
    for experiment in experiments:
        try:
            oscillation = detect_oscillations(
                experiment,
                prominence=prominence,
                min_peak_count=min_peak_count,
            )
        except ValueError as error:
            LOGGER.warning(
                "Skipping oscillation search for %s: %s",
                experiment.filename,
                error,
            )
            continue

        if not oscillation.detected:
            continue

        strength = oscillation_strength_score(oscillation)
        statistics = build_signal_statistics(
            experiment,
            oscillation_detected=True,
            oscillation_strength=strength,
            peak_count=oscillation.peak_count,
            extra={
                "trough_count": oscillation.trough_count,
                "zero_crossing_count": oscillation.zero_crossing_count,
                "amplitude_estimate": oscillation.amplitude_estimate,
            },
        )
        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=strength,
                reason="oscillations detected in measurement signal",
                signal_statistics=statistics,
                experiment=experiment,
            )
        )

    return sort_results(results)


def rank_by_oscillation_strength(
    experiments: Iterable[Experiment],
    *,
    prominence: float | None = None,
    min_peak_count: int = 3,
) -> list[SearchResult]:
    """Rank experiments by estimated oscillation strength."""

    results: list[SearchResult] = []
    for experiment in experiments:
        try:
            oscillation = detect_oscillations(
                experiment,
                prominence=prominence,
                min_peak_count=min_peak_count,
            )
        except ValueError as error:
            LOGGER.warning(
                "Skipping oscillation ranking for %s: %s",
                experiment.filename,
                error,
            )
            continue

        strength = oscillation_strength_score(oscillation)
        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=strength,
                reason="ranked by oscillation strength",
                signal_statistics=build_signal_statistics(
                    experiment,
                    oscillation_detected=oscillation.detected,
                    oscillation_strength=strength,
                    peak_count=oscillation.peak_count,
                    extra={
                        "trough_count": oscillation.trough_count,
                        "zero_crossing_count": oscillation.zero_crossing_count,
                        "amplitude_estimate": oscillation.amplitude_estimate,
                    },
                ),
                experiment=experiment,
            )
        )

    return sort_results(results)


def rank_by_signal_to_noise_ratio(
    experiments: Iterable[Experiment],
    *,
    baseline_order: int | None = 1,
) -> list[SearchResult]:
    """Rank experiments by signal-to-noise ratio."""

    results: list[SearchResult] = []
    for experiment in experiments:
        try:
            statistics = noise_statistics(
                experiment,
                baseline_order=baseline_order,
            )
        except ValueError as error:
            LOGGER.warning(
                "Skipping SNR ranking for %s: %s",
                experiment.filename,
                error,
            )
            continue

        score = statistics.signal_to_noise_ratio or 0.0
        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=float(score),
                reason="ranked by signal-to-noise ratio",
                signal_statistics=build_signal_statistics(
                    experiment,
                    signal_to_noise_ratio=statistics.signal_to_noise_ratio,
                    extra={
                        "noise_standard_deviation": statistics.standard_deviation,
                        "noise_rms": statistics.root_mean_square,
                        "noise_peak_to_peak": statistics.peak_to_peak,
                    },
                ),
                experiment=experiment,
            )
        )

    return sort_results(results)


def search_metadata(
    experiments: Iterable[Experiment],
    query: str | Mapping[str, str],
    *,
    case_sensitive: bool = False,
) -> list[SearchResult]:
    """Search experiment metadata by substring or required key/value pairs."""

    results: list[SearchResult] = []
    for experiment in experiments:
        matched_items = _metadata_matches(
            experiment.metadata,
            query,
            case_sensitive=case_sensitive,
        )
        if not matched_items:
            continue

        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=float(len(matched_items)),
                reason=f"metadata matched: {', '.join(matched_items)}",
                signal_statistics=build_signal_statistics(experiment),
                experiment=experiment,
            )
        )

    return sort_results(results)


def combine_filters(
    experiments: Iterable[Experiment],
    filters: Sequence[ExperimentPredicate],
    *,
    score_function: ScoreFunction | None = None,
    reason: str = "matched combined filters",
) -> list[SearchResult]:
    """Apply multiple experiment predicates as an AND query."""

    results: list[SearchResult] = []
    for experiment in experiments:
        try:
            matched = all(predicate(experiment) for predicate in filters)
        except ValueError as error:
            LOGGER.warning(
                "Skipping combined search for %s: %s",
                experiment.filename,
                error,
            )
            continue

        if not matched:
            continue

        score = score_function(experiment) if score_function is not None else 1.0
        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=float(score),
                reason=reason,
                signal_statistics=build_signal_statistics(experiment),
                experiment=experiment,
            )
        )

    return sort_results(results)


def build_signal_statistics(
    experiment: Experiment,
    *,
    peak_count: int | None = None,
    oscillation_detected: bool | None = None,
    oscillation_strength: float | None = None,
    signal_to_noise_ratio: float | None = None,
    extra: Mapping[str, float | int | str | bool | None] | None = None,
) -> SignalStatistics:
    """Build the compact statistics payload used by search results."""

    signal_column = None
    try:
        signal_column = detect_signal_columns(experiment).signal
    except ValueError:
        LOGGER.debug("No signal column detected for %s", experiment.filename)

    return SignalStatistics(
        signal_column=signal_column,
        peak_count=peak_count,
        oscillation_detected=oscillation_detected,
        oscillation_strength=oscillation_strength,
        signal_to_noise_ratio=signal_to_noise_ratio,
        min_magnetic_field=experiment.min_magnetic_field,
        max_magnetic_field=experiment.max_magnetic_field,
        min_temperature=experiment.min_temperature,
        max_temperature=experiment.max_temperature,
        min_angle=experiment.min_angle,
        max_angle=experiment.max_angle,
        extra=dict(extra or {}),
    )


def temperature_range_filter(
    min_temperature: float,
    max_temperature: float,
) -> ExperimentPredicate:
    """Create a reusable temperature-overlap predicate."""

    return lambda experiment: _range_overlaps(
        experiment.min_temperature,
        experiment.max_temperature,
        min_temperature,
        max_temperature,
    )


def magnetic_field_range_filter(
    min_field: float,
    max_field: float,
) -> ExperimentPredicate:
    """Create a reusable magnetic-field-overlap predicate."""

    return lambda experiment: _range_overlaps(
        experiment.min_magnetic_field,
        experiment.max_magnetic_field,
        min_field,
        max_field,
    )


def metadata_filter(
    query: str | Mapping[str, str],
    *,
    case_sensitive: bool = False,
) -> ExperimentPredicate:
    """Create a reusable metadata predicate."""

    return lambda experiment: bool(
        _metadata_matches(
            experiment.metadata,
            query,
            case_sensitive=case_sensitive,
        )
    )


def sort_results(
    results: Iterable[SearchResult],
    *,
    reverse: bool = True,
) -> list[SearchResult]:
    """Sort search results by score."""

    return sorted(results, key=lambda result: result.matching_score, reverse=reverse)


def filter_results(
    results: Iterable[SearchResult],
    predicate: Callable[[SearchResult], bool],
) -> list[SearchResult]:
    """Filter already-built search results."""

    return [result for result in results if predicate(result)]


def top_n(results: Iterable[SearchResult], count: int) -> list[SearchResult]:
    """Return the highest-scoring N results."""

    return sort_results(results)[:count]


def oscillation_strength_score(oscillation: Any) -> float:
    """Compute a scalar ranking score from an oscillation result."""

    peak_total = oscillation.peak_count + oscillation.trough_count
    return float(oscillation.amplitude_estimate * max(peak_total, 1))


def _filter_with_reason(
    experiments: Iterable[Experiment],
    *,
    predicate: ExperimentPredicate,
    reason: str,
    score_function: ScoreFunction,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for experiment in experiments:
        if not predicate(experiment):
            continue

        results.append(
            SearchResult(
                filename=experiment.filename,
                metadata=dict(experiment.metadata),
                matching_score=float(score_function(experiment)),
                reason=reason,
                signal_statistics=build_signal_statistics(experiment),
                experiment=experiment,
            )
        )

    return sort_results(results)


def _range_overlaps(
    observed_min: float | None,
    observed_max: float | None,
    requested_min: float,
    requested_max: float,
) -> bool:
    if observed_min is None or observed_max is None:
        return False
    return observed_min <= requested_max and observed_max >= requested_min


def _range_overlap_score(
    observed_min: float | None,
    observed_max: float | None,
    requested_min: float,
    requested_max: float,
) -> float:
    if observed_min is None or observed_max is None:
        return 0.0

    overlap_min = max(observed_min, requested_min)
    overlap_max = min(observed_max, requested_max)
    overlap = max(0.0, overlap_max - overlap_min)
    requested_width = max(requested_max - requested_min, 1e-12)
    return overlap / requested_width


def _metadata_matches(
    metadata: Mapping[str, str],
    query: str | Mapping[str, str],
    *,
    case_sensitive: bool,
) -> list[str]:
    if isinstance(query, str):
        return _metadata_substring_matches(
            metadata,
            query,
            case_sensitive=case_sensitive,
        )

    matched_items = []
    normalized_metadata = {
        _normalize_text(key, case_sensitive): _normalize_text(value, case_sensitive)
        for key, value in metadata.items()
    }
    for key, expected_value in query.items():
        normalized_key = _normalize_text(key, case_sensitive)
        normalized_expected = _normalize_text(expected_value, case_sensitive)
        actual_value = normalized_metadata.get(normalized_key)
        if actual_value is not None and normalized_expected in actual_value:
            matched_items.append(f"{key}={expected_value}")

    return matched_items


def _metadata_substring_matches(
    metadata: Mapping[str, str],
    query: str,
    *,
    case_sensitive: bool,
) -> list[str]:
    normalized_query = _normalize_text(query, case_sensitive)
    matched_items = []
    for key, value in metadata.items():
        normalized_key = _normalize_text(key, case_sensitive)
        normalized_value = _normalize_text(value, case_sensitive)
        if normalized_query in normalized_key or normalized_query in normalized_value:
            matched_items.append(f"{key}={value}")

    return matched_items


def _normalize_text(value: object, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.lower()
