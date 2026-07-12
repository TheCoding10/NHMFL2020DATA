"""Automatic detection of experiment structure and measurement plotting.

Given an arbitrary tabular experiment dataset, this module determines which
column was intentionally swept (the independent variable), which columns are
effectively constant instrument parameters, and which columns are measured
quantities, then generates one publication-quality plot per measurement
against the detected independent variable.

Column roles are inferred from statistics and common naming conventions
rather than a single hardcoded schema, so this module works across NHMFL-
style prefixed columns (`Field_...`, `RuO_T_...`) as well as plainly named
columns (`Magnetic Field`, `Resistance`). Other analysis modules (pattern
recognition, anomaly detection, machine learning, database search) are
expected to build on `DatasetAnalysis` rather than re-deriving column roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output") / "auto_plots"
PUBLICATION_DPI = 300
FIGURE_SIZE = (7.0, 4.8)

# A column is treated as constant when its coefficient of variation
# (std / |mean|) falls at or below this fraction, or its absolute range is
# negligible (guards against a near-zero mean, e.g. an angle held at 0).
DEFAULT_CONSTANT_RELATIVE_TOLERANCE = 0.01
CONSTANT_ABSOLUTE_TOLERANCE = 1e-9

# Priority-ordered independent-variable name groups. Column names are split
# into word tokens (camelCase and separator boundaries, numeric suffixes
# dropped) before being compared against these aliases, so "Field(T)",
# "Field_1732487951", "MagneticField", and compound instrument names like
# "RuO_T_1732487951" or "DR_Temp_1732487951" all resolve correctly.
INDEPENDENT_VARIABLE_NAME_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Magnetic Field", ("field", "b")),
    ("Angle", ("angle", "theta", "rotation")),
    ("Temperature", ("temperature", "temp", "t")),
)

# Column-name fragments that identify acquisition bookkeeping (e.g. a
# wall-clock timestamp column) rather than an intentionally swept or
# measured physical quantity. These are excluded from both the independent-
# variable statistical fallback and the measurements list, unless a column
# matching one is the only varying column available.
BOOKKEEPING_NAME_FRAGMENTS = ("time", "timestamp", "date", "index")


class ColumnRole(str, Enum):
    """The detected role of one dataframe column."""

    INDEPENDENT_VARIABLE = "independent_variable"
    CONSTANT_PARAMETER = "constant_parameter"
    MEASUREMENT = "measurement"


@dataclass(frozen=True)
class ConstantParameter:
    """A column whose value did not meaningfully change during the run."""

    column: str
    average: float
    minimum: float
    maximum: float
    standard_deviation: float
    unit: str | None = None


@dataclass(frozen=True)
class DatasetAnalysis:
    """The detected structure of one experiment dataset."""

    independent_variable: str | None
    independent_variable_category: str | None
    constant_parameters: tuple[ConstantParameter, ...]
    measurements: tuple[str, ...]
    column_roles: dict[str, ColumnRole]


@dataclass(frozen=True)
class GeneratedPlot:
    """One saved measurement-vs-independent-variable plot."""

    measurement: str
    independent_variable: str
    path: Path


class DatasetAnalyzer:
    """Detects experiment structure and plots measurements automatically.

    This is the reusable entry point future analysis modules are expected to
    build on: call `analyze()` once and reuse the resulting `DatasetAnalysis`
    instead of re-deriving column roles.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        name: str = "dataset",
        constant_relative_tolerance: float = DEFAULT_CONSTANT_RELATIVE_TOLERANCE,
    ) -> None:
        self.dataframe = dataframe
        self.name = name
        self.constant_relative_tolerance = constant_relative_tolerance
        self._analysis: DatasetAnalysis | None = None

    def analyze(self) -> DatasetAnalysis:
        """Detect the independent variable, constants, and measurements."""

        if self._analysis is not None:
            return self._analysis

        numeric_columns = tuple(
            column
            for column in self.dataframe.columns
            if pd.api.types.is_numeric_dtype(self.dataframe[column])
        )

        constant_parameters: list[ConstantParameter] = []
        varying_columns: list[str] = []
        column_roles: dict[str, ColumnRole] = {}

        for column in numeric_columns:
            series = self.dataframe[column].dropna()
            if series.empty:
                continue
            if self._is_constant(series):
                constant_parameters.append(self._describe_constant(column, series))
                column_roles[column] = ColumnRole.CONSTANT_PARAMETER
            else:
                varying_columns.append(column)

        independent_variable, category = self._detect_independent_variable(
            varying_columns
        )
        measurements = tuple(
            column
            for column in varying_columns
            if column != independent_variable
            and not _matches_any_fragment(column, BOOKKEEPING_NAME_FRAGMENTS)
        )

        if independent_variable is not None:
            column_roles[independent_variable] = ColumnRole.INDEPENDENT_VARIABLE
        for column in measurements:
            column_roles[column] = ColumnRole.MEASUREMENT

        self._analysis = DatasetAnalysis(
            independent_variable=independent_variable,
            independent_variable_category=category,
            constant_parameters=tuple(constant_parameters),
            measurements=measurements,
            column_roles=column_roles,
        )
        return self._analysis

    def generate_plots(
        self,
        *,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        show: bool = False,
    ) -> list[GeneratedPlot]:
        """Generate one measurement-vs-independent-variable plot per measurement."""

        analysis = self.analyze()
        if analysis.independent_variable is None:
            LOGGER.warning(
                "Skipping plots for %s: no independent variable was detected",
                self.name,
            )
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        x_column = analysis.independent_variable
        x_label = self._axis_label(x_column, analysis.independent_variable_category)
        generated: list[GeneratedPlot] = []

        for measurement in analysis.measurements:
            y_label = self._axis_label(measurement)
            figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
            axis.plot(
                self.dataframe[x_column],
                self.dataframe[measurement],
                linewidth=0.9,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f"{y_label} vs {x_label}")
            axis.grid(True, alpha=0.3)

            plot_path = output_path / (
                f"{_safe_stem(self.name)}_{_safe_stem(measurement)}"
                f"_vs_{_safe_stem(x_column)}.png"
            )
            figure.savefig(
                plot_path,
                dpi=PUBLICATION_DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            LOGGER.info("Saved plot: %s", plot_path)

            if show:
                plt.show()
            else:
                plt.close(figure)

            generated.append(
                GeneratedPlot(
                    measurement=measurement,
                    independent_variable=x_column,
                    path=plot_path,
                )
            )

        return generated

    def print_summary(self, plots: Iterable[GeneratedPlot] = ()) -> None:
        """Print a human-readable summary of the detected dataset structure."""

        analysis = self.analyze()
        plot_list = list(plots)

        print("-" * 32)
        print("Detected X-axis:")
        if analysis.independent_variable:
            print(
                self._axis_label(
                    analysis.independent_variable,
                    analysis.independent_variable_category,
                )
            )
        else:
            print("none")

        print("\nConstant Parameters:")
        if analysis.constant_parameters:
            for parameter in analysis.constant_parameters:
                unit = f" {parameter.unit}" if parameter.unit else ""
                print(
                    f"{_display_name(parameter.column)} = "
                    f"{parameter.average:.6g}{unit}"
                )
        else:
            print("none")

        print("\nMeasurements:")
        if analysis.measurements:
            for measurement in analysis.measurements:
                print(_display_name(measurement))
        else:
            print("none")

        print("\nGenerated:")
        if plot_list:
            for plot in plot_list:
                x_label = self._axis_label(
                    plot.independent_variable,
                    analysis.independent_variable_category,
                )
                print(f"{_display_name(plot.measurement)} vs {x_label}")
        else:
            print("none")
        print("-" * 32)

    def _is_constant(self, series: pd.Series) -> bool:
        data_range = float(series.max() - series.min())
        if data_range <= CONSTANT_ABSOLUTE_TOLERANCE:
            return True

        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if abs(mean) > CONSTANT_ABSOLUTE_TOLERANCE:
            return (std / abs(mean)) <= self.constant_relative_tolerance
        return std <= CONSTANT_ABSOLUTE_TOLERANCE

    def _describe_constant(self, column: str, series: pd.Series) -> ConstantParameter:
        return ConstantParameter(
            column=column,
            average=float(series.mean()),
            minimum=float(series.min()),
            maximum=float(series.max()),
            standard_deviation=float(series.std(ddof=0)),
            unit=_extract_unit(column),
        )

    def _detect_independent_variable(
        self,
        varying_columns: list[str],
    ) -> tuple[str | None, str | None]:
        if not varying_columns:
            return None, None

        for category, aliases in INDEPENDENT_VARIABLE_NAME_GROUPS:
            for column in varying_columns:
                if _matches_name_group(column, aliases):
                    return column, category

        candidates = [
            column
            for column in varying_columns
            if not _matches_any_fragment(column, BOOKKEEPING_NAME_FRAGMENTS)
        ] or varying_columns

        best_column = max(
            candidates,
            key=lambda column: self._sweep_score(self.dataframe[column].dropna()),
        )
        return best_column, "Inferred"

    def _sweep_score(self, series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        if values.size < 2:
            return 0.0

        index = np.arange(values.size, dtype=float)
        monotonicity = abs(_spearman_correlation(values, index))
        unique_ratio = np.unique(values).size / values.size

        mean_magnitude = float(np.mean(np.abs(values)))
        if mean_magnitude > CONSTANT_ABSOLUTE_TOLERANCE:
            normalized_range = (values.max() - values.min()) / mean_magnitude
        else:
            normalized_range = float(values.max() - values.min())
        normalized_range = min(normalized_range, 10.0) / 10.0

        return 0.5 * monotonicity + 0.3 * unique_ratio + 0.2 * normalized_range

    def _axis_label(self, column: str, category: str | None = None) -> str:
        if category and category != "Inferred":
            return category
        return _display_name(column)


def _normalize_column_name(column: str) -> str:
    """Strip units, punctuation, and trailing numeric suffixes for matching."""

    without_units = re.sub(r"[\(\[].*?[\)\]]", "", column)
    alnum_only = re.sub(r"[^A-Za-z0-9]", "", without_units)
    without_trailing_digits = re.sub(r"\d+$", "", alnum_only)
    return without_trailing_digits.lower()


def _tokenize_column_name(column: str) -> list[str]:
    """Split a column name into lowercase word tokens, dropping numeric ids."""

    without_units = re.sub(r"[\(\[].*?[\)\]]", "", column)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", without_units)
    spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced)
    return [token.lower() for token in spaced.split() if not token.isdigit()]


def _matches_name_group(column: str, aliases: tuple[str, ...]) -> bool:
    return any(token in aliases for token in _tokenize_column_name(column))


def _matches_any_fragment(column: str, fragments: tuple[str, ...]) -> bool:
    normalized = _normalize_column_name(column)
    return any(fragment in normalized for fragment in fragments)


def _extract_unit(column: str) -> str | None:
    match = re.search(r"[\(\[]([^\)\]]+)[\)\]]", column)
    return match.group(1).strip() if match else None


def _display_name(column: str) -> str:
    without_units = re.sub(r"[\(\[].*?[\)\]]", "", column).strip()
    without_trailing_id = re.sub(r"[_\s]+\d+$", "", without_units)
    spaced = without_trailing_id.replace("_", " ").strip()
    return spaced if spaced else column


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _spearman_correlation(values: np.ndarray, index: np.ndarray) -> float:
    value_ranks = pd.Series(values).rank().to_numpy()
    index_ranks = pd.Series(index).rank().to_numpy()
    if np.std(value_ranks) == 0 or np.std(index_ranks) == 0:
        return 0.0
    return float(np.corrcoef(value_ranks, index_ranks)[0, 1])


def analyze_dataframe(
    dataframe: pd.DataFrame,
    *,
    name: str = "dataset",
    constant_relative_tolerance: float = DEFAULT_CONSTANT_RELATIVE_TOLERANCE,
) -> DatasetAnalysis:
    """Convenience wrapper: analyze a DataFrame without holding an analyzer."""

    return DatasetAnalyzer(
        dataframe,
        name=name,
        constant_relative_tolerance=constant_relative_tolerance,
    ).analyze()
