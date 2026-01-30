from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Optimizer, ObjectiveFn
from ..utils import OptimizationResult, clip_to_bounds


@dataclass
class PSOConfig:
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    v_clip: float = 0.2


class PSO(Optimizer):
    def __init__(self, config: PSOConfig | None = None):
        self.config = config or PSOConfig()

    def optimize(self, objective_fn: ObjectiveFn, bounds: np.ndarray, rng: np.random.Generator, n_iter: int, pop_size: int) -> OptimizationResult:
        dim = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]
        X = rng.uniform(low, high, size=(pop_size, dim))
        V = rng.normal(0.0, 1.0, size=(pop_size, dim))
        V = np.clip(V, -self.config.v_clip, self.config.v_clip)

        P = X.copy()
        P_fit = np.array([objective_fn(x) for x in P])
        gbest_idx = int(np.argmax(P_fit))
        gbest = P[gbest_idx].copy()
        gbest_fit = float(P_fit[gbest_idx])

        history: List[float] = [gbest_fit]

        for _ in range(n_iter):
            r1 = rng.random(size=(pop_size, dim))
            r2 = rng.random(size=(pop_size, dim))
            V = self.config.w * V + self.config.c1 * r1 * (P - X) + self.config.c2 * r2 * (gbest[None, :] - X)
            V = np.clip(V, -self.config.v_clip, self.config.v_clip)
            X = clip_to_bounds(X + V, bounds)

            fit = np.array([objective_fn(x) for x in X])
            improved = fit > P_fit
            P[improved] = X[improved]
            P_fit[improved] = fit[improved]

            best_idx = int(np.argmax(P_fit))
            if float(P_fit[best_idx]) > gbest_fit:
                gbest_fit = float(P_fit[best_idx])
                gbest = P[best_idx].copy()

            history.append(gbest_fit)

        return OptimizationResult(best_x=gbest, best_f=gbest_fit, history_best=history)
