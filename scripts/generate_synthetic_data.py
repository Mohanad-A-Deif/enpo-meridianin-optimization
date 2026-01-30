from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from enpo.config import load_config
from enpo.objectives import synthetic_process_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for benchmarking.")
    parser.add_argument("--config", type=str, default="config/default.json")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compound", type=str, default="3b", choices=["3b", "4b", "5b"])
    parser.add_argument("--out", type=str, default="data/synthetic_dataset.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bounds = cfg.bounds.as_array()

    rng = np.random.default_rng(args.seed)
    low, high = bounds[:, 0], bounds[:, 1]

    X = rng.uniform(low, high, size=(args.n, bounds.shape[0]))
    c = cfg.centers[args.compound]
    center = np.array([c["temperature_C"], c["microwave_power_W"], c["reaction_time_min"], c["solvent_ratio"]], dtype=float)

    rows = []
    for x in X:
        out = synthetic_process_model(x=x, bounds=bounds, center=center, rng=rng, noise_std=cfg.noise_std)
        rows.append(
            {
                "temperature_C": float(x[0]),
                "microwave_power_W": float(x[1]),
                "reaction_time_min": float(x[2]),
                "solvent_ratio": float(x[3]),
                "yield_pct": out.yield_pct,
                "purity_pct": out.purity_pct,
                "binding_energy_kcalmol": out.binding_energy_kcalmol,
                "mcf7_inhibition_pct": out.mcf7_inhibition_pct,
                "antioxidant_ic50_ugml": out.antioxidant_ic50_ugml,
                "compound": args.compound,
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
