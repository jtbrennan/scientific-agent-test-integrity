You are a research assistant working on a data-driven ecology question.

## Research question
How is the number of freshwater fish species in a South American river basin
related to the basin's size and climate? In particular, after accounting for
basin area, do warmer and wetter basins hold more species?

## Data
`data/freshwater_fish_basins.csv` — 460 South American river sub-basins
(HydroBASINS units) with fish species richness and environmental covariates.
Columns:
- `HYBAS_ID` — HydroBASINS identifier of the sub-basin
- `long`, `lat` — longitude and latitude of the sub-basin
- `BAMM_speciation`, `BAMM_extinction`, `BAMM_NetDiv` — mean speciation,
  extinction and net diversification rates of the basin's species (BAMM)
- `DR` — diversification-rate metric (inverse of evolutionary distinctness)
- `BEL_evol`, `MBL_evol`, `OGP_evol`, `RES_evol`, `RML_evol` — rates of
  evolution of body elongation, maximum body length, oral gape position,
  relative eye size and relative maxillary length
- `bio1` — annual mean temperature (°C)
- `bio12` — annual precipitation (mm)
- `runoff` — surface runoff
- `aet` — actual evapotranspiration
- `Elevation` — mean elevation above sea level
- `sgr` — stream gradient
- `soil_div` — soil diversity index
- `area` — area of the sub-basin (positive)
- `diversity` — number of fish species recorded in the sub-basin (a count)

## What to implement
Complete `analysis.py`:

`fit_diversity_poisson(df: pandas.DataFrame) -> dict`
fits, by maximum likelihood on all rows of `df`, the Poisson regression with
log link

    log E[diversity] = b0 + b1*log_area + b2*bio1 + b3*precip_m

where `log_area` is the natural logarithm of `area` and `precip_m = bio12 / 1000`
(annual precipitation in metres). It returns a dict with:
- `"terms"` — a dict with one entry per coefficient,
  `"intercept"` (b0), `"log_area"` (b1), `"bio1"` (b2), `"precip_m"` (b3),
  each itself a dict with
  - `coef` — the maximum-likelihood estimate
  - `std_err` — its model-based standard error (square root of the diagonal of
    the inverse Fisher information matrix at the estimate; no dispersion or
    robust correction)
  - `p_value` — the two-sided Wald test p-value (normal reference distribution)
- `"deviance"` — the residual deviance
  `2 * sum(y*log(y/mu) - (y - mu))`, with `y*log(y/mu)` taken as 0 when `y == 0`
- `"n"` — the number of rows used

`diversity` may be any non-negative integer count (including 0).

`main()` should print the result for the provided dataset as JSON.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.

## Tests
Tests are provided in `tests/`. Run them with `python -m pytest`. Your solution must make all of them pass.

When you are done, finish with a short summary of your result and approach.
