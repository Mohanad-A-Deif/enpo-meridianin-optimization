# Enhanced Parrot Optimizer (EnPO) for Meridianin Derivative Design — Reproducible Benchmark (Synthetic)

This repository provides a clean, reproducible research codebase inspired by:

> Deif, M. A., Elhoseny, M., Hafez, M. A., Alomoush, W., & Khishe, M. (2025). **Enhanced parrot optimizer for Meridianin derivative design: Synthesis optimization, biological evaluation, and molecular docking insights.** *South African Journal of Chemical Engineering*.

The code implements **Enhanced Parrot Optimizer (EnPO)** for continuous optimization and includes common baselines (**PSO**, **GA**, and **ACO-R**) plus a **synthetic benchmark generator** to enable fully reproducible experiments without relying on lab-only measurements.

> **Important**: The included dataset generator produces **synthetic** (simulated) yield/purity/bioactivity-like signals for algorithm testing. It is **not** a substitute for experimental chemistry, biological assays, or docking pipelines.

---

## What you get

- **EnPO optimizer** (continuous domain) with adaptive behavior weighting and Lévy-flight exploration.
- Baselines:
  - Particle Swarm Optimization (**PSO**)
  - Genetic Algorithm (**GA**)
  - Continuous Ant Colony Optimization (**ACO-R / ACOR**)
- **Synthetic data generator**:
  - Generates process conditions (e.g., temperature, microwave power, time, solvent ratio)
  - Produces simulated outputs:
    - Yield (%)
    - Purity (%)
    - Optional in-silico-like metrics (e.g., binding-energy-like score)
- **Benchmark runner**:
  - Multiple independent runs per optimizer
  - Convergence logs per run
  - Summary CSV across runs
- Production-ready repo utilities:
  - `pyproject.toml`, `requirements.txt`
  - Tests (`pytest`) + CI workflow (GitHub Actions)
  - `CITATION.cff`, `LICENSE`, `.gitignore`

---

## Repository structure

```text
.
├─ config/                 # experiment bounds + weights
├─ data/                   # synthetic dataset output (git-ignored by default)
├─ outputs/                # benchmark results (git-ignored by default)
├─ scripts/                # runnable entrypoints
├─ src/enpo/               # library code (optimizers + objective)
└─ tests/                  # unit + smoke tests
```

---

## Installation

### Option A — Virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option B — Conda (optional)

```bash
conda create -n enpo python=3.10 -y
conda activate enpo
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

### 1) Generate a synthetic dataset

```bash
python scripts/generate_synthetic_data.py --n 5000 --seed 42 --compound 3b --out data/synthetic_dataset.csv
```

### 2) Run optimization benchmark (EnPO vs baselines)

```bash
python scripts/run_benchmark.py --compound 3b --runs 10 --seed 42 --outdir outputs
```

You will get:

- `outputs/benchmark_results.csv`
- `outputs/convergence_enpo_run0.csv`, `outputs/convergence_pso_run0.csv`, ...

---

## Configuration

All default experiment settings live in:

- `config/default.json`

Key parts:
- `bounds`: variable ranges (continuous)
- `fitness`: weights used in the scalar objective (e.g., yield vs purity)
- `synthetic_model`: synthetic signal behavior and noise
- `benchmark`: population size, iterations, number of runs

You can edit these values to match your own study design.

---

## Reproducibility notes

- Every benchmark run uses a deterministic seed.
- Convergence is logged per run for later analysis and plotting.
- CI runs tests + lint checks on each push.

To run tests locally:

```bash
pytest
```

To lint with ruff:

```bash
ruff check .
```

---

## How the synthetic benchmark works (high level)

The synthetic generator builds a smooth fitness landscape over the decision variables with:
- A dominant “good region” (Gaussian-like peak),
- Small nonlinear interactions (sin/cos terms),
- Realistic bounded noise.

This allows consistent and fair comparison between optimizers while keeping the project fully self-contained.

---

## Extending to real experiments (recommended path)

If you want to connect this codebase to real-world data:
- Replace the synthetic process model in `src/enpo/objectives.py` with:
  - a regression surrogate model trained on experimental runs, or
  - a direct call to lab-measured outputs (yield/purity), and optionally
  - docking/MD results integrated as additional objectives/constraints.
- Keep the optimizer interface unchanged.

---

## Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@article{Deif2025EnPO,
  title   = {Enhanced parrot optimizer for Meridianin derivative design: Synthesis optimization, biological evaluation, and molecular docking insights},
  author  = {Deif, Mohanad A. and Elhoseny, Mohamed and Hafez, Mohamed A. and Alomoush, Wael and Khishe, Mohammad},
  journal = {South African Journal of Chemical Engineering},
  year    = {2025}
}
```

You can also use `CITATION.cff` for GitHub-native citation support.

---

## License

MIT License — see `LICENSE`.

---

## Disclaimer

This repository includes **synthetic** benchmarks intended for software testing, optimization comparisons, and reproducibility demonstrations. It does not claim to reproduce exact experimental yields, purities, biological assay outcomes, or docking scores.
