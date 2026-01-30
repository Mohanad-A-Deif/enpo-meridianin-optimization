import numpy as np

from enpo.config import load_config
from enpo.objectives import synthetic_process_model, weighted_fitness


def test_synthetic_model_outputs_are_finite():
    cfg = load_config("config/default.json")
    bounds = cfg.bounds.as_array()
    rng = np.random.default_rng(0)
    c = cfg.centers["3b"]
    center = np.array([c["temperature_C"], c["microwave_power_W"], c["reaction_time_min"], c["solvent_ratio"]], dtype=float)

    x = np.array([190.0, 700.0, 5.0, 0.5])
    out = synthetic_process_model(x=x, bounds=bounds, center=center, rng=rng, noise_std=cfg.noise_std)
    assert np.isfinite(out.yield_pct)
    assert np.isfinite(out.purity_pct)

    f = weighted_fitness(out.yield_pct, out.purity_pct, reaction_time_min=5.0, w_yield=0.5, w_purity=0.5)
    assert np.isfinite(f)
