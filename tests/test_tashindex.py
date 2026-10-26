import numpy as np

from tashindex import sim


def test_market_is_reproducible():
    a, b = sim.make_market(1), sim.make_market(1)
    assert np.array_equal(a["true_ret"], b["true_ret"])


def test_repeat_sales_recovers_returns_without_noise():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.03, sim.T - 1)
    lv = np.cumsum(np.concatenate([[0], r]))
    price = np.tile(lv, (60, 1)) + rng.normal(0, 0.5, (60, 1))
    price[rng.random(price.shape) < 0.6] = np.nan
    est = sim.repeat_sales_index(price)
    assert np.corrcoef(est, r)[0, 1] > 0.95


def test_mean_price_index_is_biased_by_who_sells():
    w = sim.make_market(3, held_out=True)
    e = sim.mean_price_index(w["price"])
    assert np.std(e) > 1.5 * np.std(w["true_ret"])   # composition noise inflates volatility


def test_metrics_perfect_estimate():
    x = np.random.default_rng(1).normal(0, 0.03, 95)
    m = sim.metrics(x, x)
    assert m["corr"] > 0.999 and m["rmse"] < 1e-12 and abs(m["vol_ratio"] - 1) < 1e-9
