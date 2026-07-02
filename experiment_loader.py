"""Load NHMFL experiment text files into pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import logging
from pathlib import Path
import re
from typing import BinaryIO, Iterable
import zipfile

import pandas as pd


LOGGER = logging.getLogger(__name__)

DATE_PATTERN = re.compile(
    r"(?P<date>[A-Z][a-z]{2}, [A-Z][a-z]{2} \d{1,2}, \d{4} "
    r"\d{1,2}:\d{2}:\d{2} [AP]M)"
)
FIELD_PREFIXES = ("Field_",)
TIMESTAMP_PREFIXES = ("Timestamp_",)
TEMPERATURE_PREFIXES = ("RuO_T_", "Cx_T_", "Cernox_T_", "DR_Temp_", "Cernox_")
ANGLE_PREFIXES = ("Angle_",)
IGNORED_SUFFIXES = {".pxp", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"}


@dataclass
class Experiment:
    """One parsed experiment file and its core ranges."""

    filename: str
    date_time: datetime | None
    dataframe: pd.DataFrame
    number_of_rows: int
    number_of_columns: int
    min_magnetic_field: float | None
    max_magnetic_field: float | None
    min_temperature: float | None
    max_temperature: float | None
    min_angle: float | None
    max_angle: float | None
    metadata: dict[str, str]


@dataclass
class LoadReport:
    """Bookkeeping for skipped files."""

    empty_files: list[str]
    corrupted_files: list[str]


def parse_header_metadata(
    header_lines: list[str],
) -> tuple[datetime | None, dict[str, str]]:
    """Parse date/time and simple key-value metadata from the file preamble."""

    metadata: dict[str, str] = {}
    date_time = None

    if header_lines:
        metadata["title"] = header_lines[0]
        match = DATE_PATTERN.search(header_lines[0])
        if match:
            date_time = datetime.strptime(
                match.group("date"),
                "%a, %b %d, %Y %I:%M:%S %p",
            )

    for line in header_lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower().replace(" ", "_")] = value.strip()

    return date_time, metadata


def find_column_header(lines: list[str]) -> int:
    """Return the zero-based line index containing the tab-delimited columns."""

    for index, line in enumerate(lines):
        if _is_column_header(line):
            return index
    raise ValueError("column header row was not found")


def read_experiment_file(filename: str, file_obj: BinaryIO) -> Experiment:
    """Read one experiment text file from an open binary file object."""

    raw = file_obj.read()
    if not raw:
        raise ValueError("empty file")

    text = raw.decode("utf-8")
    lines = text.splitlines()
    column_header_index = find_column_header(lines)
    header_lines = lines[:column_header_index]
    date_time, metadata = parse_header_metadata(header_lines)

    table_bytes = "\n".join(lines[column_header_index:]).encode("utf-8")
    dataframe = pd.read_csv(BytesIO(table_bytes), sep="\t")
    dataframe = dataframe.apply(pd.to_numeric, errors="raise")

    field_column = _find_column_by_prefixes(dataframe.columns, FIELD_PREFIXES)
    temperature_column = _find_column_by_prefixes(
        dataframe.columns,
        TEMPERATURE_PREFIXES,
    )
    angle_column = _find_optional_column_by_prefixes(
        dataframe.columns,
        ANGLE_PREFIXES,
    )

    return Experiment(
        filename=filename,
        date_time=date_time,
        dataframe=dataframe,
        number_of_rows=len(dataframe),
        number_of_columns=len(dataframe.columns),
        min_magnetic_field=_series_min(dataframe[field_column]),
        max_magnetic_field=_series_max(dataframe[field_column]),
        min_temperature=_series_min(dataframe[temperature_column]),
        max_temperature=_series_max(dataframe[temperature_column]),
        min_angle=_series_min(dataframe[angle_column]) if angle_column else None,
        max_angle=_series_max(dataframe[angle_column]) if angle_column else None,
        metadata=metadata,
    )


def load_experiments(
    root: str | Path = "NHMFLMarch2020Data",
    *,
    print_report: bool = True,
) -> list[Experiment]:
    """Load all experiment files from an extracted dataset folder."""

    root_path = Path(root)
    data_dir = _resolve_data_dir(root_path)
    report = LoadReport(empty_files=[], corrupted_files=[])
    experiments: list[Experiment] = []

    LOGGER.info("Loading experiments from %s", data_dir)
    for path in sorted(path for path in data_dir.rglob("*") if _is_data_file(path)):
        try:
            with path.open("rb") as file_obj:
                experiment = read_experiment_file(str(path), file_obj)
        except ValueError as error:
            if "empty file" in str(error):
                report.empty_files.append(str(path))
                LOGGER.warning("Skipping empty file: %s", path)
            else:
                report.corrupted_files.append(str(path))
                LOGGER.warning("Skipping corrupted file %s: %s", path, error)
            continue
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            report.corrupted_files.append(str(path))
            LOGGER.warning("Skipping unreadable file %s: %s", path, error)
            continue

        experiments.append(experiment)

    if print_report:
        print_summary(experiments, report)

    return experiments


def load_experiments_from_zip(
    zip_path: str | Path,
    *,
    data_prefix: str = "NHMFLMarch2020Data/data/",
    print_report: bool = True,
) -> list[Experiment]:
    """Load all text experiments from the archive without extracting it."""

    report = LoadReport(empty_files=[], corrupted_files=[])
    experiments: list[Experiment] = []

    LOGGER.info("Loading experiments from %s:%s", zip_path, data_prefix)
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith(data_prefix) and _is_data_name(name)
        )
        for name in names:
            try:
                with archive.open(name) as file_obj:
                    experiment = read_experiment_file(name, file_obj)
            except ValueError as error:
                if "empty file" in str(error):
                    report.empty_files.append(name)
                    LOGGER.warning("Skipping empty file: %s", name)
                else:
                    report.corrupted_files.append(name)
                    LOGGER.warning("Skipping corrupted file %s: %s", name, error)
                continue
            except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
                report.corrupted_files.append(name)
                LOGGER.warning("Skipping unreadable file %s: %s", name, error)
                continue

            experiments.append(experiment)

    if print_report:
        print_summary(experiments, report)

    return experiments


def print_summary(experiments: list[Experiment], report: LoadReport) -> None:
    """Print a concise loading report."""

    print(f"Loaded {len(experiments)} experiments")
    print(f"Skipped {len(report.empty_files)} empty file(s)")
    print(f"Skipped {len(report.corrupted_files)} corrupted file(s)")

    if not experiments:
        print("Average rows per experiment: 0")
        print("Largest experiment: none")
        print("Smallest experiment: none")
        return

    average_rows = sum(
        experiment.number_of_rows for experiment in experiments
    ) / len(experiments)
    largest = max(experiments, key=lambda experiment: experiment.number_of_rows)
    smallest = min(experiments, key=lambda experiment: experiment.number_of_rows)

    print(f"Average rows per experiment: {average_rows:.2f}")
    print(f"Largest experiment: {largest.filename} ({largest.number_of_rows} rows)")
    print(f"Smallest experiment: {smallest.filename} ({smallest.number_of_rows} rows)")


def _resolve_data_dir(root_path: Path) -> Path:
    if root_path.name == "data":
        data_dir = root_path
    elif (root_path / "data").is_dir():
        data_dir = root_path / "data"
    else:
        data_dir = root_path

    if not data_dir.is_dir():
        raise FileNotFoundError(f"experiment data directory was not found: {data_dir}")

    return data_dir


def _is_column_header(line: str) -> bool:
    columns = line.strip().split("\t")
    if len(columns) < 2:
        return False

    return all(
        _find_optional_column_by_prefixes(columns, prefixes) is not None
        for prefixes in (FIELD_PREFIXES, TIMESTAMP_PREFIXES)
    )


def _find_column_by_prefixes(columns: Iterable[str], prefixes: tuple[str, ...]) -> str:
    column = _find_optional_column_by_prefixes(columns, prefixes)
    if column is None:
        raise ValueError(f"required column with prefixes {prefixes!r} was not found")
    return column


def _find_optional_column_by_prefixes(
    columns: Iterable[str],
    prefixes: tuple[str, ...],
) -> str | None:
    for column in columns:
        if column.startswith(prefixes):
            return column
    return None


def _is_data_file(path: Path) -> bool:
    return path.is_file() and _is_data_name(path.name)


def _is_data_name(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix in IGNORED_SUFFIXES:
        return False
    return suffix == ".txt" or suffix[1:].isdigit()


def _series_min(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.min())


def _series_max(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.max())


def configure_logging() -> None:
    """Configure default console logging for CLI usage."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


if __name__ == "__main__":
    configure_logging()
    load_experiments()
