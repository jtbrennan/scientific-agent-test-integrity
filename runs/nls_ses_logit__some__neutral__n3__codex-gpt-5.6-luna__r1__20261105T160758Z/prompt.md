You are a research assistant working on a data-driven social science question.

## Research question
Is socioeconomic status (SES) a significant predictor of BA degree completion?

## Data
`data/nls_ses_processed.csv` — participants of the National Longitudinal Survey (NLS).
Columns:
- `CASE ID` — unique respondent ID
- `SAMPLE_RACE` — Hispanic, Black or White
- `SAMPLE_SEX` — Male or Female
- `FAMILY SIZE OF SAMPLE` — family size
- `ABILITY: COMPOSITE OF ASVAB SCORE` — composite test score
- `BA DEGREE COMPLETED` — True if the respondent completed a BA degree
- `PERCENTILE IN CLASS` — class percentile in the last school year
- `SES` — socioeconomic status (continuous)

## What to implement
Complete `analysis.py`:

`fit_ses_effect(df: pandas.DataFrame) -> dict`
fits a logistic regression, with an intercept, of `BA DEGREE COMPLETED` on `SES`
by maximum likelihood, using all rows of `df`, and returns a dict with:
- `coef` — the estimated SES coefficient (log-odds)
- `std_err` — its standard error
- `p_value` — the two-sided Wald test p-value for the SES coefficient
- `odds_ratio` — `exp(coef)`
- `n` — the number of rows used

`main()` should print the result for the provided dataset as JSON.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.

## Tests
Tests are provided in `tests/`. Run them with `python -m pytest`. Your solution must make all of them pass.

When you are done, finish with a short summary of your result and approach.
