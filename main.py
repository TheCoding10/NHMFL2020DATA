"""Command-line application for NHMFL condensed matter data analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable
import zipfile

import pandas as pd

from experiment_loader import Experiment, read_experiment_file
from search_engine import (
    SearchResult,
    find_by_magnetic_field_range,
    find_by_temperature_range,
    find_containing_oscillations,
    find_with_more_than_n_peaks,
    rank_by_signal_to_noise_ratio,
    top_n,
)
from signal_processing import detect_oscillations, detect_peaks
from visualization import plot_all_experiments, plot_experiment


LOGGER = logging.getLogger(__name__)

DEFAULT_ZIP_PATH = (
    Path.home() / "Downloads" / "NHMFLMarch2020Data-20260630T191837Z-3-001.zip"
)
DEFAULT_DATASET_PATH = Path("NHMFLMarch2020Data")
PLOTS_DIR = Path("output") / "plots"
RESULTS_DIR = Path("output") / "results"
IGNORED_DATA_SUFFIXES = {".pxp", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class DatasetLoadResult:
    """Loaded experiments and source-file bookkeeping."""

    experiments: list[Experiment]
    source_count: int
    skipped_empty: int
    skipped_corrupted: int

    @property
    def skipped_count(self) -> int:
        """Return total skipped source files."""

        return self.skipped_empty + self.skipped_corrupted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the research CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Explore, analyze, plot, and search NHMFL condensed matter "
            "experiments."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to an extracted NHMFLMarch2020Data folder or ZIP archive.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate a summary report for the loaded dataset.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate standard plots for matching experiments.",
    )
    parser.add_argument(
        "--temperature",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Find experiments overlapping a temperature range.",
    )
    parser.add_argument(
        "--field",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Find experiments overlapping a magnetic-field range.",
    )
    parser.add_argument(
        "--oscillations",
        action="store_true",
        help="Find experiments containing oscillatory signal structure.",
    )
    parser.add_argument(
        "--peaks",
        type=int,
        metavar="MIN",
        help="Find experiments with more than MIN detected peaks.",
    )
    parser.add_argument(
        "--snr",
        action="store_true",
        help="Rank experiments by signal-to-noise ratio.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Limit displayed and exported search results. Default: 10.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        metavar="FILE",
        help="Restrict analysis or plotting to one filename, e.g. Agosta.001.txt.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export search results or summary report CSV files to output/results/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def main() -> int:
    """Run the command-line application."""

    args = parse_args()
    configure_logging(verbose=args.verbose)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(args.dataset)
    print(f"Loading dataset from {dataset_path}")
    load_result = load_dataset(dataset_path)
    experiments = load_result.experiments
    print(
        f"Loaded {len(experiments)} experiment(s); "
        f"skipped {load_result.skipped_count}."
    )

    if args.experiment:
        experiments = select_experiment(experiments, args.experiment)
        print(f"Selected experiment: {experiments[0].filename}")

    if args.summary:
        report = generate_summary_report(
            experiments,
            skipped_count=load_result.skipped_count,
        )
        print_summary_report(report)
        if args.export:
            path = export_summary_report(report)
            print(f"Summary exported to {path}")

    results = run_searches(args, experiments)
    if results:
        limited_results = top_n(results, args.top)
        print_search_results(limited_results)
        if args.export:
            path = export_search_results(limited_results)
            print(f"Search results exported to {path}")
    elif has_search_request(args):
        print("No matching experiments found.")

    if args.plot:
        if results:
            targets = [result.experiment for result in top_n(results, args.top)]
        else:
            targets = experiments
        plot_paths = plot_selected_experiments(targets, output_dir=PLOTS_DIR)
        print(f"Generated {len(plot_paths)} plot file(s) in {PLOTS_DIR}")

    if not args.summary and not has_search_request(args) and not args.plot:
        print("No action selected. Use --summary, --plot, or a search option.")

    return 0


def configure_logging(*, verbose: bool) -> None:
    """Configure structured console logging."""

    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def resolve_dataset_path(requested_path: Path | None) -> Path:
    """Resolve the dataset path from CLI input or known defaults."""

    candidates = []
    if requested_path is not None:
        candidates.append(requested_path)
    candidates.extend([DEFAULT_DATASET_PATH, DEFAULT_ZIP_PATH])

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded

    raise FileNotFoundError(
        "Could not find NHMFL dataset. Pass --dataset with a folder or ZIP path."
    )


def load_dataset(dataset_path: Path) -> DatasetLoadResult:
    """Load experiments from a ZIP archive or extracted folder with progress."""

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        return _load_zip_dataset(dataset_path)
    return _load_folder_dataset(dataset_path)


def _load_zip_dataset(dataset_path: Path) -> DatasetLoadResult:
    """Load experiments from a ZIP archive with progress reporting."""

    experiments: list[Experiment] = []
    skipped_empty = 0
    skipped_corrupted = 0

    with zipfile.ZipFile(dataset_path) as archive:
        names = [
            name
            for name in sorted(archive.namelist())
            if name.startswith("NHMFLMarch2020Data/data/")
            and name.endswith(".txt")
        ]
        print(f"Found {len(names)} experiment file(s). Loading...")
        for index, name in enumerate(names, start=1):
            _print_progress(index, len(names), "loading")
            try:
                with archive.open(name) as file_obj:
                    experiments.append(read_experiment_file(name, file_obj))
            except ValueError as error:
                if "empty file" in str(error):
                    skipped_empty += 1
                else:
                    skipped_corrupted += 1
                    LOGGER.warning("Skipping corrupted file %s: %s", name, error)
            except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
                skipped_corrupted += 1
                LOGGER.warning("Skipping unreadable file %s: %s", name, error)
    print()

    return DatasetLoadResult(
        experiments=experiments,
        source_count=len(names),
        skipped_empty=skipped_empty,
        skipped_corrupted=skipped_corrupted,
    )


def _load_folder_dataset(dataset_path: Path) -> DatasetLoadResult:
    """Load experiments from an extracted dataset folder with progress."""

    if dataset_path.name == "data":
        data_dir = dataset_path
    elif (dataset_path / "data").is_dir():
        data_dir = dataset_path / "data"
    else:
        data_dir = dataset_path

    if not data_dir.is_dir():
        raise FileNotFoundError(f"experiment data directory was not found: {data_dir}")

    paths = sorted(path for path in data_dir.rglob("*") if _is_data_file(path))
    experiments: list[Experiment] = []
    skipped_empty = 0
    skipped_corrupted = 0

    print(f"Found {len(paths)} experiment file(s). Loading...")
    for index, path in enumerate(paths, start=1):
        _print_progress(index, len(paths), "loading")
        try:
            with path.open("rb") as file_obj:
                experiments.append(read_experiment_file(str(path), file_obj))
        except ValueError as error:
            if "empty file" in str(error):
                skipped_empty += 1
            else:
                skipped_corrupted += 1
                LOGGER.warning("Skipping corrupted file %s: %s", path, error)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            skipped_corrupted += 1
            LOGGER.warning("Skipping unreadable file %s: %s", path, error)
    print()

    return DatasetLoadResult(
        experiments=experiments,
        source_count=len(paths),
        skipped_empty=skipped_empty,
        skipped_corrupted=skipped_corrupted,
    )


def select_experiment(
    experiments: Iterable[Experiment],
    filename: str,
) -> list[Experiment]:
    """Select one experiment by basename or full filename suffix."""

    matches = [
        experiment
        for experiment in experiments
        if Path(experiment.filename).name == filename
        or experiment.filename.endswith(filename)
    ]
    if not matches:
        raise ValueError(f"Experiment was not found: {filename}")
    if len(matches) > 1:
        LOGGER.warning("Multiple experiments matched %s; using first match", filename)
    return [matches[0]]


def run_searches(
    args: argparse.Namespace,
    experiments: list[Experiment],
) -> list[SearchResult]:
    """Run requested search and ranking operations."""

    result_sets: list[list[SearchResult]] = []

    if args.temperature:
        min_temp, max_temp = args.temperature
        result_sets.append(find_by_temperature_range(experiments, min_temp, max_temp))

    if args.field:
        min_field, max_field = args.field
        result_sets.append(
            find_by_magnetic_field_range(experiments, min_field, max_field)
        )

    if args.oscillations:
        print("Searching for oscillatory experiments...")
        result_sets.append(find_containing_oscillations(experiments))

    if args.peaks is not None:
        print(f"Searching for experiments with more than {args.peaks} peaks...")
        result_sets.append(find_with_more_than_n_peaks(experiments, args.peaks))

    if args.snr:
        print("Ranking experiments by signal-to-noise ratio...")
        result_sets.append(rank_by_signal_to_noise_ratio(experiments))

    if not result_sets:
        return []

    return merge_result_sets(result_sets)


def merge_result_sets(result_sets: list[list[SearchResult]]) -> list[SearchResult]:
    """Combine search result sets by filename, accumulating scores and reasons."""

    merged: dict[str, SearchResult] = {}
    for result_set in result_sets:
        for result in result_set:
            current = merged.get(result.filename)
            if current is None:
                merged[result.filename] = result
                continue

            merged[result.filename] = SearchResult(
                filename=current.filename,
                metadata=current.metadata,
                matching_score=current.matching_score + result.matching_score,
                reason=f"{current.reason}; {result.reason}",
                signal_statistics=result.signal_statistics,
                experiment=current.experiment,
            )

    return sorted(
        merged.values(),
        key=lambda result: result.matching_score,
        reverse=True,
    )


def generate_summary_report(
    experiments: list[Experiment],
    *,
    skipped_count: int,
) -> dict[str, object]:
    """Generate a dataset-level summary report."""

    print("Computing summary analyses...")
    oscillation_count = 0
    peak_rows: list[tuple[str, int]] = []
    for index, experiment in enumerate(experiments, start=1):
        _print_progress(index, len(experiments), "analysis")
        try:
            if detect_oscillations(experiment).detected:
                oscillation_count += 1
            peak_count = len(detect_peaks(experiment).peak_indices)
            peak_rows.append((experiment.filename, peak_count))
        except ValueError as error:
            LOGGER.warning(
                "Skipping summary analysis for %s: %s",
                experiment.filename,
                error,
            )
    print()

    snr_results = rank_by_signal_to_noise_ratio(experiments)
    peak_rows.sort(key=lambda item: item[1], reverse=True)

    return {
        "total_experiments": len(experiments),
        "skipped_files": skipped_count,
        "temperature": _range_pair_statistics(
            (experiment.min_temperature, experiment.max_temperature)
            for experiment in experiments
        ),
        "magnetic_field": _range_pair_statistics(
            (experiment.min_magnetic_field, experiment.max_magnetic_field)
            for experiment in experiments
        ),
        "detected_oscillation_count": oscillation_count,
        "highest_snr": top_n(snr_results, 10),
        "largest_peak_counts": peak_rows[:10],
    }


def print_summary_report(report: dict[str, object]) -> None:
    """Print a clean dataset summary report."""

    print("\nDataset Summary")
    print("=" * 72)
    print(f"Total experiments: {report['total_experiments']}")
    print(f"Skipped files: {report['skipped_files']}")
    print(f"Temperature statistics: {report['temperature']}")
    print(f"Magnetic field statistics: {report['magnetic_field']}")
    print(f"Detected oscillation count: {report['detected_oscillation_count']}")

    print("\nHighest signal-to-noise experiments")
    for result in report["highest_snr"]:
        snr = result.signal_statistics.signal_to_noise_ratio
        print(f"  {Path(result.filename).name}: SNR={_format_float(snr)}")

    print("\nLargest peak counts")
    for filename, peak_count in report["largest_peak_counts"]:
        print(f"  {Path(filename).name}: peaks={peak_count}")


def print_search_results(results: list[SearchResult]) -> None:
    """Print formatted search results."""

    print("\nSearch Results")
    print("=" * 72)
    for index, result in enumerate(results, start=1):
        stats = result.signal_statistics
        print(f"{index}. {Path(result.filename).name}")
        print(f"   score: {_format_float(result.matching_score)}")
        print(f"   reason: {result.reason}")
        print(f"   signal: {stats.signal_column or 'unknown'}")
        print(
            "   ranges: "
            f"T=[{_format_float(stats.min_temperature)}, "
            f"{_format_float(stats.max_temperature)}], "
            f"B=[{_format_float(stats.min_magnetic_field)}, "
            f"{_format_float(stats.max_magnetic_field)}]"
        )


def plot_selected_experiments(
    experiments: Iterable[Experiment],
    *,
    output_dir: Path,
) -> list[Path]:
    """Generate plots for selected experiments with progress output."""

    experiment_list = list(experiments)
    if len(experiment_list) == 1:
        return plot_experiment(experiment_list[0], output_dir=output_dir)

    print(f"Generating plots for {len(experiment_list)} experiment(s)...")
    return plot_all_experiments(experiment_list, output_dir=output_dir)


def export_search_results(results: list[SearchResult]) -> Path:
    """Export search results to CSV in output/results/."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"search_results_{_timestamp()}.csv"
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "filename",
                "score",
                "reason",
                "signal_column",
                "peak_count",
                "oscillation_detected",
                "oscillation_strength",
                "signal_to_noise_ratio",
                "min_temperature",
                "max_temperature",
                "min_magnetic_field",
                "max_magnetic_field",
            ],
        )
        writer.writeheader()
        for result in results:
            stats = result.signal_statistics
            writer.writerow(
                {
                    "filename": result.filename,
                    "score": result.matching_score,
                    "reason": result.reason,
                    "signal_column": stats.signal_column,
                    "peak_count": stats.peak_count,
                    "oscillation_detected": stats.oscillation_detected,
                    "oscillation_strength": stats.oscillation_strength,
                    "signal_to_noise_ratio": stats.signal_to_noise_ratio,
                    "min_temperature": stats.min_temperature,
                    "max_temperature": stats.max_temperature,
                    "min_magnetic_field": stats.min_magnetic_field,
                    "max_magnetic_field": stats.max_magnetic_field,
                }
            )
    return path


def export_summary_report(report: dict[str, object]) -> Path:
    """Export the summary report to CSV in output/results/."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"summary_report_{_timestamp()}.csv"
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_experiments", report["total_experiments"]])
        writer.writerow(["skipped_files", report["skipped_files"]])
        writer.writerow(["temperature", report["temperature"]])
        writer.writerow(["magnetic_field", report["magnetic_field"]])
        writer.writerow(
            ["detected_oscillation_count", report["detected_oscillation_count"]]
        )
        for result in report["highest_snr"]:
            writer.writerow(
                [
                    f"highest_snr:{Path(result.filename).name}",
                    result.signal_statistics.signal_to_noise_ratio,
                ]
            )
        for filename, peak_count in report["largest_peak_counts"]:
            writer.writerow([f"peak_count:{Path(filename).name}", peak_count])
    return path


def has_search_request(args: argparse.Namespace) -> bool:
    """Return True if any search or ranking action was requested."""

    return any(
        [
            args.temperature,
            args.field,
            args.oscillations,
            args.peaks is not None,
            args.snr,
        ]
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.peaks is not None and args.peaks < 0:
        parser.error("--peaks MIN must be non-negative")
    if args.temperature and args.temperature[0] > args.temperature[1]:
        parser.error("--temperature MIN must be <= MAX")
    if args.field and args.field[0] > args.field[1]:
        parser.error("--field MIN must be <= MAX")


def _is_data_file(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix in IGNORED_DATA_SUFFIXES:
        return False
    return suffix == ".txt" or suffix[1:].isdigit()


def _range_pair_statistics(
    ranges: Iterable[tuple[float | None, float | None]],
) -> dict[str, float | None]:
    """Summarize a collection of experiment min/max ranges."""

    valid_values = [
        value
        for value_range in ranges
        for value in value_range
        if value is not None
    ]
    if not valid_values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(valid_values),
        "max": max(valid_values),
        "mean": sum(valid_values) / len(valid_values),
    }


def _print_progress(index: int, total: int, label: str) -> None:
    if total == 0:
        return
    if index == total or index % max(total // 20, 1) == 0:
        percent = 100.0 * index / total
        print(f"\r{label}: {index}/{total} ({percent:5.1f}%)", end="", flush=True)


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)
