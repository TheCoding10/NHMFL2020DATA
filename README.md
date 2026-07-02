# NHMFL Condensed Matter Analysis Platform

Command-line tools for loading, plotting, analyzing, and searching NHMFL March
2020 condensed matter experiment data.

## What This Project Does

- Loads experiment `.txt` files into pandas DataFrames.
- Extracts metadata and experiment ranges.
- Generates publication-quality scientific plots.
- Runs signal-processing analyses including peaks, oscillations, FFT, and noise
  statistics.
- Searches experiments by temperature, magnetic field, peak count,
  oscillations, and signal-to-noise ratio.
- Exports search results to CSV.

## Project Files

- `experiment_loader.py` loads raw experiment files.
- `visualization.py` generates plots in `output/plots/`.
- `signal_processing.py` provides reusable analysis functions.
- `search_engine.py` searches and ranks experiments.
- `main.py` provides the command-line interface.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Place the dataset ZIP at:

```text
~/Downloads/NHMFLMarch2020Data-20260630T191837Z-3-001.zip
```

or pass a dataset path explicitly:

```bash
python3 main.py --dataset /path/to/NHMFLMarch2020Data --summary
```

## Example Commands

```bash
python3 main.py --summary
python3 main.py --temperature 0.5 2.0
python3 main.py --field 0 16 --top 20
python3 main.py --oscillations --top 10
python3 main.py --snr --top 10 --export
python3 main.py --experiment Agosta.001.txt --plot
```

## Outputs

Generated files are written to:

```text
output/plots/
output/results/
```

The raw dataset and generated outputs are intentionally ignored by Git.
