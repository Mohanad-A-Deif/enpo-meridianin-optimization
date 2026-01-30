from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Optimizer, ObjectiveFn
from ..utils import OptimizationResult, clip_to_bounds


@dataclass
class GAConfig:
    tournament_k: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    mutation_sigma: float = 0.08


class GA(Optimizer):
    def __init__(self, config: GAConfig | None = None):
        self.config = config or GAConfig()

    def _tournament(self, fit: np.ndarray, rng: np.random.Generator) -> int:
        idx = rng.integers(0, len(fit), size=self.config.tournament_k)
        return int(idx[np.argmax(fit[idx])])

    def optimize(self, objective_fn: ObjectiveFn, bounds: np.ndarray, rng: np.random.Generator, n_iter: int, pop_size: int) -> OptimizationResult:
        dim = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]
        span = high - low

        X = rng.uniform(low, high, size=(pop_size, dim))
        fit = np.array([objective_fn(x) for x in X])

        best_idx = int(np.argmax(fit))
        best_x = X[best_idx].copy()
        best_f = float(fit[best_idx])
        history: List[float] = [best_f]

        for _ in range(n_iter):
            new_pop = []
            while len(new_pop) < pop_size:
                p1 = X[self._tournament(fit, rng)]
                p2 = X[self._tournament(fit, rng)]

                c1, c2 = p1.copy(), p2.copy()
                if rng.random() < self.config.crossover_prob:
                    alpha = rng.random(size=dim)
                    c1 = alpha * p1 + (1 - alpha) * p2
                    c2 = alpha * p2 + (1 - alpha) * p1

                for c in (c1, c2):
                    if rng.random() < self.config.mutation_prob:
                        c += rng.normal(0.0, self.config.mutation_sigma, size=dim) * span

                c1 = clip_to_bounds(c1, bounds)
                c2 = clip_to_bounds(c2, bounds)
                new_pop.extend([c1, c2])

            X = np.array(new_pop[:pop_size])
            fit = np.array([objective_fn(x) for x in X])

            idx = int(np.argmax(fit))
            if float(fit[idx]) > best_f:
                best_f = float(fit[idx])
                best_x = X[idx].copy()

            history.append(best_f)

        return OptimizationResult(best_x=best_x, best_f=best_f, history_best=history)
