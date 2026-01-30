from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Optimizer, ObjectiveFn
from ..utils import OptimizationResult, clip_to_bounds, levy_flight


@dataclass
class EnPOConfig:
    alpha_hierarchy: float = 0.6
    gamma_mimic: float = 0.5
    delta_levy: float = 0.05
    forage_scale: float = 0.25
    weight_lr: float = 0.2
    min_weight: float = 0.05


class EnPO(Optimizer):
    def __init__(self, config: EnPOConfig | None = None):
        self.config = config or EnPOConfig()
        self.behaviors = ("forage", "hierarchy", "mimic")

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

        w = {b: 1.0 for b in self.behaviors}

        def probs() -> np.ndarray:
            ww = np.array([w[b] for b in self.behaviors], dtype=float)
            ww = np.maximum(ww, self.config.min_weight)
            return ww / np.sum(ww)

        for _ in range(n_iter):
            p = probs()
            gains = {b: 0.0 for b in self.behaviors}

            for i in range(pop_size):
                b = rng.choice(self.behaviors, p=p)
                xi = X[i].copy()

                if b == "forage":
                    step = rng.normal(0.0, 1.0, size=dim) * self.config.forage_scale * span
                    step += self.config.delta_levy * levy_flight(rng, dim) * span
                    x_new = xi + step
                elif b == "hierarchy":
                    r = rng.random(size=dim)
                    x_new = xi + self.config.alpha_hierarchy * r * (best_x - xi)
                else:
                    r = rng.random(size=dim)
                    x_new = best_x + self.config.gamma_mimic * r * (best_x - xi)
                    x_new += self.config.delta_levy * levy_flight(rng, dim) * span

                x_new = clip_to_bounds(x_new, bounds)
                f_new = float(objective_fn(x_new))

                gain = f_new - float(fit[i])
                if gain > 0:
                    X[i] = x_new
                    fit[i] = f_new
                    gains[b] += gain

                if f_new > best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            total_gain = sum(gains.values())
            if total_gain > 0:
                for b in self.behaviors:
                    rel = gains[b] / (total_gain + 1e-12)
                    w[b] = (1 - self.config.weight_lr) * w[b] + self.config.weight_lr * (1 + rel)

            history.append(best_f)

        return OptimizationResult(best_x=best_x, best_f=float(best_f), history_best=history)
