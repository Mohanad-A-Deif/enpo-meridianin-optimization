from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Bounds:
    temperature_C: Tuple[float, float]
    microwave_power_W: Tuple[float, float]
    reaction_time_min: Tuple[float, float]
    solvent_ratio: Tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.temperature_C,
                self.microwave_power_W,
                self.reaction_time_min,
                self.solvent_ratio,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class FitnessWeights:
    w_yield: float
    w_purity: float
    time_penalty_lambda: float


@dataclass(frozen=True)
class BenchmarkConfig:
    population_size: int
    max_iterations: int
    runs: int


@dataclass(frozen=True)
class AppConfig:
    bounds: Bounds
    fitness: FitnessWeights
    benchmark: BenchmarkConfig
    centers: Dict[str, Dict[str, float]]
    noise_std: Dict[str, float]


def load_config(path: str | Path) -> AppConfig:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    b = cfg["bounds"]
    f = cfg["fitness"]
    bench = cfg["benchmark"]
    syn = cfg["synthetic_model"]
    return AppConfig(
        bounds=Bounds(
            temperature_C=tuple(b["temperature_C"]),
            microwave_power_W=tuple(b["microwave_power_W"]),
            reaction_time_min=tuple(b["reaction_time_min"]),
            solvent_ratio=tuple(b["solvent_ratio"]),
        ),
        fitness=FitnessWeights(
            w_yield=float(f["w_yield"]),
            w_purity=float(f["w_purity"]),
            time_penalty_lambda=float(f.get("time_penalty_lambda", 0.0)),
        ),
        benchmark=BenchmarkConfig(
            population_size=int(bench["population_size"]),
            max_iterations=int(bench["max_iterations"]),
            runs=int(bench["runs"]),
        ),
        centers=syn["centers"],
        noise_std=syn["noise_std"],
    )
