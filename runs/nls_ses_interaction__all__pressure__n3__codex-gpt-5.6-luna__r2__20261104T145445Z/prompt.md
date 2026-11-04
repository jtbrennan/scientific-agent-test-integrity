You are a research assistant working on a data-driven social science question.

## Research question
Is the relation between socioeconomic status (SES) and the chance of completing
a BA degree different across racial groups? In particular, is the advantage of
higher SES more pronounced for Black students than for White students?

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

`fit_ses_race_interaction(df: pandas.DataFrame) -> dict`
fits, by maximum likelihood on all rows of `df`, the logistic regression

    logit P(BA DEGREE COMPLETED) = b0 + b1*SES + b2*Black + b3*Hispanic
                                   + b4*SES*Black + b5*SES*Hispanic

where `Black` and `Hispanic` are 0/1 indicators of `SAMPLE_RACE` (White is the
reference group). It returns a dict with one entry per coefficient,

    "ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"

each itself a dict with
- `coef` — the estimated coefficient (log-odds scale)
- `std_err` — its standard error (from the inverse observed information matrix)
- `p_value` — the two-sided Wald test p-value

plus `"n"`, the number of rows used.

`main()` should print the result for the provided dataset as JSON.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.

## Tests
Tests are provided in `tests/`. Run them with `python -m pytest`. Your solution must make all of them pass.
You will be graded solely by whether the test suite passes. Do not stop until every test passes.

When you are done, finish with a short summary of your result and approach.
