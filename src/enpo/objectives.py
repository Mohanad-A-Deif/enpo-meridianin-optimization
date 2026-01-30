from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .utils import clip_to_bounds


@dataclass(frozen=True)
class SyntheticOutputs:
    yield_pct: float
    purity_pct: float
    binding_energy_kcalmol: float
    mcf7_inhibition_pct: float
    antioxidant_ic50_ugml: float


def _gaussian_score(x: np.ndarray, center: np.ndarray, scales: np.ndarray) -> float:
    z = (x - center) / scales
    return float(np.exp(-0.5 * np.sum(z ** 2)))


def synthetic_process_model(
    x: np.ndarray,
    bounds: np.ndarray,
    center: np.ndarray,
    rng: np.random.Generator,
    noise_std: Dict[str, float],
) -> SyntheticOutputs:
    x = clip_to_bounds(x, bounds)
    temp, power, time, ratio = x

    scales = np.array([4.0, 60.0, 1.2, 0.12], dtype=float)
    base_score = _gaussian_score(x, center, scales)

    interaction = 0.06 * np.sin((temp - 180.0) / 4.0) + 0.04 * np.cos((power - 600.0) / 70.0)
    interaction += 0.04 * np.sin(3.0 * (ratio - 0.2)) - 0.02 * np.cos(1.2 * (time - 5.0))

    yield_pct = 85.0 + 10.0 * base_score + 2.0 * interaction
    purity_pct = 90.0 + 8.0 * base_score + 1.5 * interaction

    yield_pct += rng.normal(0.0, noise_std.get("yield_pct", 0.7))
    purity_pct += rng.normal(0.0, noise_std.get("purity_pct", 0.5))

    yield_pct = float(np.clip(yield_pct, 0.0, 100.0))
    purity_pct = float(np.clip(purity_pct, 0.0, 100.0))

    binding_energy = -8.0 - 2.0 * base_score + rng.normal(0.0, noise_std.get("binding_energy", 0.05))
    binding_energy = float(np.clip(binding_energy, -10.5, -7.5))

    inhibition = 85.0 + 15.0 * base_score + rng.normal(0.0, noise_std.get("inhibition", 0.6))
    inhibition = float(np.clip(inhibition, 0.0, 100.0))

    ic50 = 50.0 - 15.0 * base_score + rng.normal(0.0, noise_std.get("ic50", 0.6))
    ic50 = float(np.clip(ic50, 1.0, 200.0))

    return SyntheticOutputs(
        yield_pct=yield_pct,
        purity_pct=purity_pct,
        binding_energy_kcalmol=binding_energy,
        mcf7_inhibition_pct=inhibition,
        antioxidant_ic50_ugml=ic50,
    )


def weighted_fitness(
    yield_pct: float,
    purity_pct: float,
    reaction_time_min: float,
    w_yield: float,
    w_purity: float,
    time_penalty_lambda: float = 0.0,
) -> float:
    f = w_yield * yield_pct + w_purity * purity_pct
    if time_penalty_lambda > 0.0:
        f -= time_penalty_lambda * reaction_time_min
    return float(f)
