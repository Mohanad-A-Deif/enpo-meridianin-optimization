from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from enpo.config import load_config
from enpo.objectives import synthetic_process_model, weighted_fitness
from enpo.optimizers import EnPO, PSO, GA, ACOR


def make_objective(cfg, compound: str, rng: np.random.Generator):
    bounds = cfg.bounds.as_array()
    c = cfg.centers[compound]
    center = np.array([c["temperature_C"], c["microwave_power_W"], c["reaction_time_min"], c["solvent_ratio"]], dtype=float)

    def obj(x: np.ndarray) -> float:
        out = synthetic_process_model(x=x, bounds=bounds, center=center, rng=rng, noise_std=cfg.noise_std)
        return weighted_fitness(
            yield_pct=out.yield_pct,
            purity_pct=out.purity_pct,
            reaction_time_min=float(x[2]),
            w_yield=cfg.fitness.w_yield,
            w_purity=cfg.fitness.w_purity,
            time_penalty_lambda=cfg.fitness.time_penalty_lambda,
        )

    return obj


def run_one(name: str, cfg, compound: str, seed: int):
    bounds = cfg.bounds.as_array()
    rng = np.random.default_rng(seed)
    obj = make_objective(cfg, compound, rng)

    if name == "enpo":
        opt = EnPO()
    elif name == "pso":
        opt = PSO()
    elif name == "ga":
        opt = GA()
    elif name == "acor":
        opt = ACOR()
    else:
        raise ValueError(name)

    return opt.optimize(obj, bounds=bounds, rng=rng, n_iter=cfg.benchmark.max_iterations, pop_size=cfg.benchmark.population_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EnPO vs baselines on synthetic objective.")
    parser.add_argument("--config", type=str, default="config/default.json")
    parser.add_argument("--compound", type=str, default="3b", choices=["3b", "4b", "5b"])
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = int(args.runs) if args.runs is not None else cfg.benchmark.runs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = ["enpo", "pso", "ga", "acor"]
    records = []

    for name in names:
        for r in range(runs):
            seed = args.seed + 1000 * (names.index(name) + 1) + r
            res = run_one(name, cfg, args.compound, seed)
            records.append(
                {
                    "optimizer": name,
                    "run": r,
                    "best_f": res.best_f,
                    "best_temperature_C": float(res.best_x[0]),
                    "best_microwave_power_W": float(res.best_x[1]),
                    "best_reaction_time_min": float(res.best_x[2]),
                    "best_solvent_ratio": float(res.best_x[3]),
                    "seed": seed,
                }
            )
            conv = pd.DataFrame({"iter": np.arange(len(res.history_best)), "best_f": res.history_best})
            conv.to_csv(outdir / f"convergence_{name}_run{r}.csv", index=False)

    df = pd.DataFrame(records)
    df.to_csv(outdir / "benchmark_results.csv", index=False)
    print(df.groupby("optimizer")["best_f"].agg(["mean", "std", "max"]).round(4))


if __name__ == "__main__":
    main()
