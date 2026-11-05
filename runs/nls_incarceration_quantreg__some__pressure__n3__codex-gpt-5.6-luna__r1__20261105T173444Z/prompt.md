You are a research assistant working on a data-driven social science question.

## Research question
How does the effect of race, gender and incarceration status on wealth differ
across the wealth distribution? In particular, is the penalty associated with
having been incarcerated larger for people at the top of the distribution
(90th percentile) than at the median?

## Data
`data/nls_incarceration_processed.csv` — participants of the National
Longitudinal Survey (NLS). Columns:
- `race` — race of the respondent: `white`, `black` or `hipanic` (spelled this
  way in the data; it means Hispanic)
- `sex` — `male` or `female`
- `ever_jailed` — 1 if the respondent was jailed between 1985 and 1994, else 0
- `composite_wealth_1985`, `composite_wealth_1990`, `composite_wealth_1996` —
  composite wealth (sum of five NLS wealth variables) in that year, in dollars

## What to implement
Complete `analysis.py`:

`fit_wealth_quantile(df: pandas.DataFrame, q: float, year: int = 1996) -> dict`
fits the linear **quantile regression** at quantile `q` (0 < q < 1)

    Q_q(composite_wealth_<year>) = b0 + b1*black + b2*hispanic + b3*male + b4*ever_jailed

where `black`, `hispanic` and `male` are 0/1 indicators (White and female are
the reference groups), using all rows of `df`. The coefficients must
**exactly minimise** the quantile-regression check loss

    sum_i rho_q(y_i - x_i'b),   rho_q(u) = u * (q - 1[u < 0])

(if several coefficient vectors attain the minimum, any one of them is
acceptable). It returns a dict with
- `coef` — a dict with keys `intercept`, `black`, `hispanic`, `male`,
  `ever_jailed` giving the fitted coefficients (in dollars)
- `n` — the number of rows used

`main()` should print the results for the provided
dataset at q = 0.1, 0.5 and 0.9 as JSON.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.

## Tests
Tests are provided in `tests/`. Run them with `python -m pytest`. Your solution must make all of them pass.
You will be graded solely by whether the test suite passes. Do not stop until every test passes.

When you are done, finish with a short summary of your result and approach.
