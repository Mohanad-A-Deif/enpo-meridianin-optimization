import numpy as np

from enpo.config import load_config
from enpo.optimizers import EnPO, PSO, GA, ACOR


def test_optimizers_smoke_run():
    cfg = load_config("config/default.json")
    bounds = cfg.bounds.as_array()
    rng = np.random.default_rng(0)

    def obj(x: np.ndarray) -> float:
        return -float(np.sum(x**2))

    for opt in (EnPO(), PSO(), GA(), ACOR()):
        res = opt.optimize(obj, bounds=bounds, rng=rng, n_iter=5, pop_size=10)
        assert np.isfinite(res.best_f)
        assert res.best_x.shape == (bounds.shape[0],)
