# EnPO Meridianin Optimization (Synthetic Benchmark)

This repository provides a clean, reproducible implementation of:
- **Enhanced Parrot Optimization (EnPO)** for continuous search spaces
- Baselines: **PSO**, **GA**, and **ACO-R** (continuous ACO)
- A **synthetic data generator** that simulates reaction yield/purity and optional in-silico bioactivity metrics

> Note: The synthetic generator is designed for **algorithm benchmarking and reproducibility**.
> It does not reproduce laboratory measurements.

## Quick start

### 1) Create environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate synthetic dataset
```bash
python scripts/generate_synthetic_data.py --n 5000 --seed 42 --out data/synthetic_dataset.csv
```

### 3) Run benchmark (10 runs per optimizer)
```bash
python scripts/run_benchmark.py --compound 3b --runs 10 --seed 42
```

Outputs:
- `outputs/benchmark_results.csv`
- `outputs/convergence_<optimizer>_run<k>.csv`

## Configuration
Default bounds and weights are stored in `config/default.json`.

## License
MIT
