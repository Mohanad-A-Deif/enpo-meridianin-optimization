from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Optimizer, ObjectiveFn
from ..utils import OptimizationResult, clip_to_bounds


@dataclass
class ACORConfig:
    archive_size: int = 40
    q: float = 0.5
    xi: float = 0.85


class ACOR(Optimizer):
    def __init__(self, config: ACORConfig | None = None):
        self.config = config or ACORConfig()

    def optimize(self, objective_fn: ObjectiveFn, bounds: np.ndarray, rng: np.random.Generator, n_iter: int, pop_size: int) -> OptimizationResult:
        dim = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]

        archive = rng.uniform(low, high, size=(self.config.archive_size, dim))
        fit = np.array([objective_fn(x) for x in archive])

        def sort_archive(a, f):
            idx = np.argsort(-f)
            return a[idx], f[idx]

        archive, fit = sort_archive(archive, fit)
        best_x, best_f = archive[0].copy(), float(fit[0])
        history: List[float] = [best_f]

        ranks = np.arange(1, self.config.archive_size + 1)
        weights = (1 / (self.config.q * self.config.archive_size * np.sqrt(2 * np.pi))) * np.exp(
            -((ranks - 1) ** 2) / (2 * (self.config.q * self.config.archive_size) ** 2)
        )
        weights = weights / np.sum(weights)

        for _ in range(n_iter):
            sigma = np.zeros((self.config.archive_size, dim))
            for i in range(self.config.archive_size):
                diffs = np.abs(archive[i] - archive)
                sigma[i] = self.config.xi * np.mean(diffs, axis=0) + 1e-12

            ants = []
            for _k in range(pop_size):
                idx = rng.choice(self.config.archive_size, p=weights)
                sample = rng.normal(loc=archive[idx], scale=sigma[idx])
                ants.append(clip_to_bounds(sample, bounds))

            ants = np.array(ants)
            ants_fit = np.array([objective_fn(x) for x in ants])

            merged = np.vstack([archive, ants])
            merged_fit = np.concatenate([fit, ants_fit])
            order = np.argsort(-merged_fit)
            archive = merged[order][: self.config.archive_size]
            fit = merged_fit[order][: self.config.archive_size]

            if float(fit[0]) > best_f:
                best_f = float(fit[0])
                best_x = archive[0].copy()

            history.append(best_f)

        return OptimizationResult(best_x=best_x, best_f=best_f, history_best=history)
