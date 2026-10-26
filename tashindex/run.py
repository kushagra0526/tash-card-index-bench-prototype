import json

import numpy as np

from . import sim

METHODS = {"mean sale price": sim.mean_price_index, "median sale price": sim.median_price_index,
           "last sale carried forward": sim.last_sale_index, "repeat sales": sim.repeat_sales_index}


def score(seeds, held):
    acc = {k: {} for k in METHODS}
    sales_per_month = []
    for s in seeds:
        w = sim.make_market(s, held)
        sales_per_month.append(float(np.sum(~np.isnan(w["price"])) / sim.T))
        for k, fn in METHODS.items():
            for m, v in sim.metrics(fn(w["price"]), w["true_ret"]).items():
                acc[k].setdefault(m, []).append(v)
    out = {k: {m: round(float(np.mean(v)), 3) for m, v in d.items()} for k, d in acc.items()}
    out["_sales_per_month_mean"] = round(float(np.mean(sales_per_month)), 1)
    return out


def main():
    res = {"dev": score(range(0, 30), False), "held_out": score(range(1000, 1030), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k, "sales/month", v.pop("_sales_per_month_mean"))
        for m, d in v.items():
            print(f"  {m:28}", d)


if __name__ == "__main__":
    main()
