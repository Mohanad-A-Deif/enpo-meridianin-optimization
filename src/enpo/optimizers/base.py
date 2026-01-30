from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..utils import OptimizationResult

ObjectiveFn = Callable[[np.ndarray], float]


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        objective_fn: ObjectiveFn,
        bounds: np.ndarray,
        rng: np.random.Generator,
        n_iter: int,
        pop_size: int,
    ) -> OptimizationResult:
        raise NotImplementedError
