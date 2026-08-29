"""Smoke test for the analysis code paths.

THIS USES A SYNTHETIC FIXTURE ON PURPOSE. It exists only to verify that the
metric and bootstrap code runs and returns sane values. It is NOT a result and
its numbers never enter the paper. Real results come from running the pipeline
on the real SGO/DMV data.
"""
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.lift_test import (quadratic_weighted_kappa, mae_ordinal,
                              paired_bootstrap, qwk_wrap, FoldPreds)


def test_qwk_perfect_and_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    assert abs(quadratic_weighted_kappa(y, y, 5) - 1.0) < 1e-9
    yp = rng.integers(0, 5, 500)
    qwk = quadratic_weighted_kappa(y, yp, 5)
    assert -0.2 < qwk < 0.2  # random predictions ~ 0


def test_mae():
    assert mae_ordinal([0, 1, 2], [0, 1, 2]) == 0.0
    assert mae_ordinal([0, 0], [1, 3]) == 2.0


def test_paired_bootstrap_detects_lift():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.integers(0, 5, n)
    # base: noisy; cand: closer to truth -> positive QWK lift expected.
    base = FoldPreds("S", "gbm", y, np.clip(y + rng.integers(-2, 3, n), 0, 4))
    cand = FoldPreds("S+T", "gbm", y, np.clip(y + rng.integers(-1, 2, n), 0, 4))
    res = paired_bootstrap(base, cand, 5, qwk_wrap, higher_better=True, n_boot=1000)
    assert res["lift"] > 0
    assert res["p_one_sided"] < 0.05
    assert res["ci95"][0] < res["ci95"][1]


if __name__ == "__main__":
    test_qwk_perfect_and_random()
    test_mae()
    test_paired_bootstrap_detects_lift()
    print("OK: all smoke tests passed (synthetic fixture; not a paper result)")
