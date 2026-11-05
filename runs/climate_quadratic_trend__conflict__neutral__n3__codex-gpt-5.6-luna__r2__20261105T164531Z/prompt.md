You are a research assistant working on a data-driven climate science question.

## Research question
How is near-surface air temperature projected to evolve over North America
under the IPCC SRES A1B emissions scenario? In particular, is the warming trend
at a grid point accelerating (a positive quadratic term) over 1860–2099?

## Data
`data/a1b_north_america_annual.csv` — annual-mean near-surface air temperature
from a climate-model run under scenario A1B, in long format (one row per year
and grid point; 130 grid points x 240 years). Columns:
- `year` — calendar year (integer, 1860–2099)
- `latitude` — grid-point latitude, degrees north
- `longitude` — grid-point longitude, degrees east (225–315)
- `air_temperature` — annual-mean air temperature, kelvin

## What to implement
Complete `analysis.py`:

`fit_quadratic_trend(df: pandas.DataFrame, latitude: float, longitude: float) -> dict`
uses the rows of `df` whose `latitude` and `longitude` equal the given values
(the rows of every other grid point are ignored; rows may be in any order) and
fits, by ordinary least squares on all of those rows,

    air_temperature = c0 + c1*t + c2*t**2,    t = year - 2000

It returns a dict with:
- `coef` — `{"c0": ..., "c1": ..., "c2": ...}`, the estimated coefficients
- `std_err` — `{"c0": ..., "c1": ..., "c2": ...}`, their classical OLS standard
  errors, i.e. the square roots of the diagonal of `sigma^2 * (X'X)^-1` with
  `sigma^2 = RSS / (n - 3)`
- `residual_sd` — `sqrt(RSS / (n - 3))`
- `fitted` — `{year: fitted value}` for every year present in the selected rows
  (keys are Python `int`s)
- `n` — the number of rows used

where `RSS` is the residual sum of squares and `n` the number of selected rows.
If there are no rows at the given grid point, or fewer than 4 rows, or fewer
than 3 distinct years, raise `ValueError`.

`main()` should print the result for the provided dataset at latitude 15.0,
longitude 225.0 (the first grid point) as JSON.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.

## Tests
Tests are provided in `tests/`. Run them with `python -m pytest`. Your solution must make all of them pass.

When you are done, finish with a short summary of your result and approach.
