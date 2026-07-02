# NHMFL Condensed Matter Analysis Platform

Command-line tools for loading, plotting, analyzing, and searching condensed
matter experiment data from NHMFL/Tallahassee measurement campaigns.

## What This Project Does

- Loads experiment text files into pandas DataFrames.
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

For the NHMFL March 2020 dataset, place the dataset ZIP at:

```text
~/Downloads/NHMFLMarch2020Data-20260630T191837Z-3-001.zip
```

or pass a dataset path explicitly:

```bash
python3 main.py --dataset /path/to/NHMFLMarch2020Data --summary
```

For the Tallahassee June 2022 dataset, pass the flat folder path:

```bash
python3 main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --summary
```

On this machine, the Anaconda Python already has the scientific dependencies
installed. If plain `python3` cannot import pandas, use:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --summary
```

The loader supports both the 2020 schema (`RuO_T`, `Counter`, `Angle`) and the
2022 schema (`FQ1`, `FQ2`, `Cernox_T`, `DR_Temp`, optional angle).

## Example Commands

```bash
python3 main.py --summary
python3 main.py --temperature 0.5 2.0
python3 main.py --field 0 16 --top 20
python3 main.py --oscillations --top 10
python3 main.py --snr --top 10 --export
python3 main.py --experiment Agosta.001.txt --plot
python3 main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --field 20 28
python3 main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --experiment Clark_SCM4.007.txt --plot
```

## Tallahassee June 2022 Commands

Use this pattern:

```bash
cd /Users/prayasthapa/nhmfl-analysis-platform
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June [COMMAND]
```

Generate a dataset summary:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --summary
```

Search high magnetic-field experiments:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --field 20 28 --top 10
```

Search by temperature range:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --temperature 0.03 0.3 --top 10
```

Find oscillatory experiments:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --oscillations --top 10
```

Rank experiments by signal-to-noise ratio:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --snr --top 10
```

Find experiments with many detected peaks:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --peaks 100 --top 10
```

Generate plots for one experiment:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --experiment Clark_SCM4.007.txt --plot
```

Search high-field experiments and plot the top 5:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --field 20 28 --top 5 --plot
```

Export search results to CSV:

```bash
/opt/anaconda3/bin/python main.py --dataset /Users/prayasthapa/Downloads/Tallahassee2022June --field 20 28 --top 20 --export
```

## Outputs

Generated files are written to:

```text
output/plots/
output/results/
```

The raw dataset and generated outputs are intentionally ignored by Git.
