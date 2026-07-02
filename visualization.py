"""Visualization utilities for NHMFL condensed matter experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiment_loader import Experiment


LOGGER = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("output") / "plots"
PUBLICATION_DPI = 300
FIGURE_SIZE = (7.0, 4.8)


@dataclass(frozen=True)
class ExperimentColumns:
    """Detected semantic columns for one experiment DataFrame."""

    magnetic_field: str
    timestamp: str
    temperature: str
    angle: str | None
    counter: str | None
    primary_measurement: str | None
    measurements: tuple[str, ...]


def detect_columns(experiment: Experiment) -> ExperimentColumns:
    """Detect field, timestamp, temperature, angle, and measurement columns."""

    columns = tuple(str(column) for column in experiment.dataframe.columns)
    magnetic_field = _find_prefixed_column(columns, "Field_")
    timestamp = _find_prefixed_column(columns, "Timestamp_")
    temperature = _find_prefixed_column_by_options(
        columns,
        ("RuO_T_", "Cx_T_", "Cernox_T_", "DR_Temp_", "Cernox_"),
    )
    angle = _find_optional_prefixed_column(columns, "Angle_")
    counter = _find_optional_prefixed_column(columns, "Counter_")

    structural_columns = {
        magnetic_field,
        timestamp,
        temperature,
        angle,
    }
    measurements = tuple(
        column for column in columns if column not in structural_columns
    )
    primary_measurement = counter or _choose_primary_measurement(measurements)

    return ExperimentColumns(
        magnetic_field=magnetic_field,
        timestamp=timestamp,
        temperature=temperature,
        angle=angle,
        counter=counter,
        primary_measurement=primary_measurement,
        measurements=measurements,
    )


def plot_magnetic_field_vs_counter(
    experiment: Experiment,
    *,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    show: bool = False,
    save: bool = True,
) -> Path | None:
    """Plot magnetic field as a function of counter/readout."""

    columns = detect_columns(experiment)
    x_column = columns.counter or columns.primary_measurement
    if x_column is None:
        LOGGER.warning(
            "Skipping signal plot for %s: no measurement column",
            experiment.filename,
        )
        return None

    return _plot_experiment_columns(
        experiment,
        x_column=x_column,
        y_column=columns.magnetic_field,
        title=f"Magnetic Field vs {x_column}",
        output_dir=output_dir,
        filename_suffix="field_vs_counter",
        show=show,
        save=save,
    )


def plot_magnetic_field_vs_temperature(
    experiment: Experiment,
    *,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    show: bool = False,
    save: bool = True,
) -> Path | None:
    """Plot magnetic field as a function of RuO temperature."""

    columns = detect_columns(experiment)
    return _plot_experiment_columns(
        experiment,
        x_column=columns.temperature,
        y_column=columns.magnetic_field,
        title="Magnetic Field vs Temperature",
        output_dir=output_dir,
        filename_suffix="field_vs_temperature",
        show=show,
        save=save,
    )


def plot_timestamp_vs_magnetic_field(
    experiment: Experiment,
    *,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    show: bool = False,
    save: bool = True,
) -> Path | None:
    """Plot magnetic field as a function of timestamp."""

    columns = detect_columns(experiment)
    return _plot_experiment_columns(
        experiment,
        x_column=columns.timestamp,
        y_column=columns.magnetic_field,
        title="Timestamp vs Magnetic Field",
        output_dir=output_dir,
        filename_suffix="timestamp_vs_field",
        show=show,
        save=save,
    )


def plot_experiment(
    experiment: Experiment,
    *,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    show: bool = False,
    save: bool = True,
) -> list[Path]:
    """Generate all standard plots for one experiment."""

    paths: list[Path] = []
    for plotter in (
        plot_magnetic_field_vs_counter,
        plot_magnetic_field_vs_temperature,
        plot_timestamp_vs_magnetic_field,
    ):
        path = plotter(
            experiment,
            output_dir=output_dir,
            show=show,
            save=save,
        )
        if path is not None:
            paths.append(path)

    return paths


def plot_all_experiments(
    experiments: Iterable[Experiment],
    *,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    show: bool = False,
    save: bool = True,
) -> list[Path]:
    """Generate standard plots for every loaded experiment."""

    output_paths: list[Path] = []
    for experiment in experiments:
        try:
            output_paths.extend(
                plot_experiment(
                    experiment,
                    output_dir=output_dir,
                    show=show,
                    save=save,
                )
            )
        except ValueError as error:
            LOGGER.warning("Skipping plots for %s: %s", experiment.filename, error)

    LOGGER.info("Generated %d plot file(s)", len(output_paths))
    return output_paths


def make_line_plot(
    experiment: Experiment,
    *,
    x_column: str,
    y_column: str,
    title: str,
) -> tuple[Figure, Axes]:
    """Create a publication-ready line plot for two DataFrame columns."""

    dataframe = experiment.dataframe
    if x_column not in dataframe.columns:
        raise ValueError(f"missing x column: {x_column}")
    if y_column not in dataframe.columns:
        raise ValueError(f"missing y column: {y_column}")

    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    axis.plot(
        dataframe[x_column],
        dataframe[y_column],
        linewidth=0.9,
        label=_experiment_label(experiment),
    )
    axis.set_xlabel(_axis_label(x_column))
    axis.set_ylabel(_axis_label(y_column))
    axis.set_title(f"{title}: {_experiment_label(experiment)}")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", frameon=False)
    return figure, axis


def save_figure(
    figure: Figure,
    output_path: str | Path,
    *,
    dpi: int = PUBLICATION_DPI,
) -> Path:
    """Save a figure as a publication-quality PNG."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    LOGGER.info("Saved plot: %s", path)
    return path


def _plot_experiment_columns(
    experiment: Experiment,
    *,
    x_column: str,
    y_column: str,
    title: str,
    output_dir: str | Path,
    filename_suffix: str,
    show: bool,
    save: bool,
) -> Path | None:
    figure, _axis = make_line_plot(
        experiment,
        x_column=x_column,
        y_column=y_column,
        title=title,
    )

    path = None
    if save:
        output_path = (
            Path(output_dir) / f"{_safe_stem(experiment)}_{filename_suffix}.png"
        )
        path = save_figure(figure, output_path)

    if show:
        plt.show()
    else:
        plt.close(figure)

    return path


def _find_prefixed_column(columns: Iterable[str], prefix: str) -> str:
    column = _find_optional_prefixed_column(columns, prefix)
    if column is None:
        raise ValueError(f"required column with prefix {prefix!r} was not found")
    return column


def _find_prefixed_column_by_options(
    columns: Iterable[str],
    prefixes: tuple[str, ...],
) -> str:
    for prefix in prefixes:
        column = _find_optional_prefixed_column(columns, prefix)
        if column is not None:
            return column
    raise ValueError(f"required column with prefixes {prefixes!r} was not found")


def _find_optional_prefixed_column(columns: Iterable[str], prefix: str) -> str | None:
    for column in columns:
        if column.startswith(prefix):
            return column
    return None


def _choose_primary_measurement(measurements: tuple[str, ...]) -> str | None:
    for prefix in ("Counter_", "FQ1_", "FQ2_"):
        for column in measurements:
            if column.startswith(prefix):
                return column
    return measurements[0] if measurements else None


def _experiment_label(experiment: Experiment) -> str:
    return Path(experiment.filename).name


def _safe_stem(experiment: Experiment) -> str:
    stem = Path(experiment.filename).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _axis_label(column: str) -> str:
    if column.startswith("Timestamp_"):
        return f"{column} (s)"
    if column.startswith(("RuO_T_", "Cx_T_", "Cernox_T_", "DR_Temp_", "Cernox_")):
        return f"{column} (K)"
    if column.startswith("Field_"):
        return f"{column} (T)"
    if column.startswith("Angle_"):
        return f"{column} (deg)"
    return column
