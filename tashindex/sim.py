"""A trading-card market with known values, sparse noisy sales, and four ways to build an index from the sales."""
import numpy as np

N, T = 400, 96


def make_market(seed, held_out=False):
    rng = np.random.default_rng(seed)
    m = np.zeros(T)
    for t in range(1, T):
        m[t] = 0.004 + rng.normal(0, 0.035)
    crash = np.zeros(T, bool)
    c0 = int(rng.integers(40, 60))
    m[c0:c0 + 6] -= 0.04 if not held_out else 0.055   # a 6-month slide
    crash[c0:c0 + 8] = True
    beta = rng.normal(1.0, 0.3, N)
    idio = rng.normal(0, 0.05, (N, T))
    logv = np.cumsum(beta[:, None] * m[None, :] + idio, axis=1) + rng.normal(5, 1.0, (N, 1))
    true_ret = np.mean(np.diff(logv, axis=1), axis=0)   # equal-weight index, monthly log return
    lam0 = rng.gamma(2.0, (0.10 if held_out else 0.25) / 2.0, N)
    sel = 1.2 if held_out else 0.6                       # pricier cards trade more often
    lam = lam0[:, None] * np.exp(sel * (logv - logv.mean(axis=0, keepdims=True)))
    lam = lam * np.where(crash[None, :], 0.4 if held_out else 0.7, 1.0)
    sold = rng.random((N, T)) < 1 - np.exp(-lam)
    disc = np.where(crash, -0.06 if held_out else -0.03, 0.0)[None, :]       # forced sellers take a discount
    price = np.where(sold, logv + rng.normal(0, 0.10, (N, T)) + disc, np.nan)   # log sale price
    return {"logv": logv, "true_ret": true_ret, "price": price, "crash": crash}


def mean_price_index(price):
    p = np.exp(price)
    mp = np.nanmean(np.where(np.isnan(p), np.nan, p), axis=0)
    mp = _fill(mp)
    return np.diff(np.log(mp))


def median_price_index(price):
    mp = _fill(np.nanmedian(np.exp(price), axis=0))
    return np.diff(np.log(mp))


def last_sale_index(price):
    """Carry each card's last sale forward, average in logs. Cards never sold yet are left out."""
    last = price.copy()
    for t in range(1, T):
        last[:, t] = np.where(np.isnan(last[:, t]), last[:, t - 1], last[:, t])
    return np.diff(np.nanmean(last, axis=0))


def repeat_sales_index(price, ridge=1e-3):
    """Bailey-Muth-Nourse: log(p2/p1) for consecutive sales of one card = sum of monthly returns between them."""
    rows, y = [], []
    for i in range(price.shape[0]):
        ts = np.where(~np.isnan(price[i]))[0]
        for a, b in zip(ts[:-1], ts[1:]):
            r = np.zeros(T - 1)
            r[a:b] = 1.0                      # returns a->a+1 ... b-1->b
            rows.append(r)
            y.append(price[i, b] - price[i, a])
    X, y = np.array(rows), np.array(y)
    A = X.T @ X + ridge * np.eye(T - 1)
    return np.linalg.solve(A, X.T @ y)


def _fill(x):
    x = x.copy()
    for t in range(len(x)):
        if np.isnan(x[t]):
            x[t] = x[t - 1] if t else np.nanmean(x)
    return x


def metrics(est, truth):
    c, tc = np.cumsum(est), np.cumsum(truth)
    dd = lambda s: float(np.max(np.maximum.accumulate(s) - s))
    return {"corr": float(np.corrcoef(est, truth)[0, 1]), "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
            "vol_ratio": float(np.std(est) / np.std(truth)), "drawdown_ratio": dd(c) / dd(tc), "cum_err": float(c[-1] - tc[-1])}
