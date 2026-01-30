from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return np.minimum(np.maximum(x, low), high)


def levy_flight(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=dim)
    v = rng.normal(0, 1.0, size=dim)
    return u / (np.abs(v) ** (1 / beta))


@dataclass(frozen=True)
class OptimizationResult:
    best_x: np.ndarray
    best_f: float
    history_best: List[float]
