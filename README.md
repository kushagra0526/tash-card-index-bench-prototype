# tashindex: building a card index from sparse, noisy sales

tash sells shares of professionally constructed indexes of trading cards; its homepage compares an index ("CL50") with the S&P 500 and says the curve is "anchored to documented values" (https://www.tash.cards/). Any index of cards that trade a few times a year has to be built from sparse, noisy sales, and the easy ways of doing it are wrong in known ways. This is a synthetic market with known values, to show how wrong.

## The setup
`tashindex/sim.py`: 400 cards over 96 months; each card has a true log value (market factor + card beta + card noise); a card sells in a month with a probability that rises with its value (pricier cards trade more), falls in a crash, and a sale price is the true value plus noise, with a forced-seller discount during the crash. The true index is the equal-weight average of the cards' monthly log changes. Four ways to build an index from the sales alone:
- **mean sale price** of the month,
- **median sale price** of the month,
- **last sale carried forward** for each card, averaged,
- **repeat sales** (Bailey-Muth-Nourse): for each card's consecutive sales, the log price ratio equals the sum of the monthly index returns between them; solve by least squares (small ridge).

## Results (30 simulated markets each; known truth)
Dev: ~90 sales a month. Held-out (seeds 1000-1029, fixed in git before the first run): ~58 sales a month, stronger "pricier cards sell more" selection, a deeper crash with a 60% drop in trading and a larger fire-sale discount.

| Held-out | correlation with true monthly returns | volatility vs true | max drawdown vs true | cumulative return error (log) |
|---|---|---|---|---|
| mean sale price | 0.27 | 3.2x | 1.35x | +0.21 |
| median sale price | 0.17 | 5.1x | 1.76x | +0.19 |
| last sale carried forward | 0.30 | 0.60x | 1.48x | -0.80 |
| repeat sales | **0.81** | 1.26x | 1.12x | **-0.00** |

Dev: repeat sales 0.87 correlation, 1.15x volatility, -0.04 cumulative error; mean price 0.31, 3.4x; last sale 0.50, 0.52x, -0.46.

- Price averages are mostly noise about *which* cards sold that month: 3-5x too volatile and nearly uncorrelated with the true index.
- Carrying the last sale forward is smooth and stale: it understates volatility, lags turning points, and drifts (-0.80 over 8 years held-out; it also leaves out cards that have not yet sold).
- Repeat sales tracks the index, but still overstates volatility by 12-26% and overstates the maximum drawdown by 12%.

## Read this before believing the table
My own market: log-normal noise, independent sales, no fees, no card-condition effects, no grading changes, equal weights. Real indexes weight by value, have thin and clustered sales, bid-ask spreads and population-report effects, and the "true" index is never observed. Nothing here is tash's method (the site does not describe one) or uses tash data.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m tashindex.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with tash.
