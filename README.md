# PCF: Reconstruction-Guided Dual-Domain Correction for Time Series Forecasting

Code accompanying **Look Back to See Forward: Reconstruction-Guided Dual-Domain Correction for Time Series Forecasting**.

This repository contains the Predict-Correct Framework (PCF) model variants and forecasting experiment scripts.

## Contents

- `ts_benchmark/`: benchmark pipeline, model wrappers, baseline models, and PCF model variants.
- `config/`: forecasting evaluation configuration files.
- `scripts/run_benchmark.py`: main benchmark entry point.
- `scripts/noise_robustness.py`: noise robustness experiment script; outputs CSV summaries only.
- `scripts/zeroshot_transfer.py`: zero-shot transfer experiment script; outputs CSV summaries only.
- `scripts/zeroshot_linear.py` and `scripts/zeroshot_transformer.py`: split zero-shot scripts for linear and transformer-style models.
- `requirements.txt`: Python dependencies.
- `LICENSE.third_party`: license notice for the underlying benchmark code.

## Data

Datasets are not included in this upload. Please follow the dataset references
and the TFB protocol cited in the manuscript for dataset acquisition and
preparation. Place the resulting benchmark CSV files under:

```text
dataset/forecasting/
```

The dataset names and citations are listed in the manuscript. This package
intentionally does not include dataset files or dataset download links.

## Setup

```bash
pip install -r requirements.txt
```

## Example Commands

Run a standard long-term forecasting benchmark:

```bash
python scripts/run_benchmark.py \
  --config-path fixed_forecast_config_hourly.json \
  --data-name-list ETTh1.csv \
  --model-name time_series_library.DLinear time_series_library.DLinear_PC \
  --adapter transformer_adapter transformer_adapter \
  --strategy-args '{"horizon": 96}' \
  --seed 2024 \
  --report-method csv \
  --save-path outputs/main_example
```

Run the noise robustness script:

```bash
python scripts/noise_robustness.py --epochs 15 --seed 2024
```

Run the zero-shot transfer script:

```bash
python scripts/zeroshot_transfer.py --epochs 15 --seed 2024
```

The reported manuscript results use three independent seeds: `2024`, `2025`,
and `2026`. The commands above show one run; repeat them with the other two
seeds and aggregate the resulting CSV files to reproduce the reported means,
standard deviations, and significance tests.

All generated logs and CSV files should be written to local output folders and are not tracked in this repository.

## Third-party notices

The underlying benchmark license is preserved in `LICENSE.third_party`. Existing third-party notices in source files are retained.
