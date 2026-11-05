You are a research assistant working on a data-driven meta-science question.

## Research question
Large replication projects re-ran published experiments with new samples. Are the
original effect sizes systematically larger than their replications, and does this
"shrinkage" differ between the Psychology and the Experimental Economics projects?

## Data
`data/replication_projects.csv` — one row per pair of an original study and its
replication (86 pairs). Columns used in this task:
- `study` — study identifier, usually the names of the original authors
- `project.x` — the replication project of the study: `Psychology` or `Experimental Economics`
- `fiso` — effect estimate of the original study on the Fisher-z scale
- `fisr` — effect estimate of the replication study on the Fisher-z scale
- `se_fiso` — standard error of `fiso`
- `se_fisr` — standard error of `fisr`

Other columns (not needed here; some contain missing values) include `ro` / `rr`
(effects on the correlation scale), `po` / `pr` (two-sided p-values), `no` / `nr`
(sample sizes), `discipline` (Social, Cognitive or Economics), `pub_year`, `journal`,
`replicated`, `relative_es` and study / author / lab characteristics.

## What to implement
Complete `analysis.py`:

`random_effects_shrinkage(df: pandas.DataFrame, project: str | None = None) -> dict`

Use the rows of `df` whose `project.x` equals `project` (all rows of `df` when
`project` is `None`). For each of these k study pairs let

    d_i = fiso_i - fisr_i              (how much the original exceeds the replication)
    v_i = se_fiso_i**2 + se_fisr_i**2  (its sampling variance)

and pool the d_i with a **DerSimonian–Laird random-effects meta-analysis**:

    w_i   = 1 / v_i                                   (fixed-effect weights)
    d_FE  = sum(w_i d_i) / sum(w_i)
    Q     = sum(w_i (d_i - d_FE)**2)                  (Cochran's Q)
    tau2  = max(0, (Q - (k - 1)) / (sum(w_i) - sum(w_i**2) / sum(w_i)))
    w*_i  = 1 / (v_i + tau2)                          (random-effects weights)
    mu    = sum(w*_i d_i) / sum(w*_i)
    std_err = sqrt(1 / sum(w*_i))
    I2    = max(0, (Q - (k - 1)) / Q)                 (0 when Q = 0)

Return a dict with
- `mu` — the pooled shrinkage estimate
- `std_err` — its standard error
- `ci_low`, `ci_high` — the 95% normal-approximation confidence interval
  `mu ± z_0.975 * std_err` (z_0.975 ≈ 1.959964)
- `p_value` — the two-sided z-test p-value of `mu = 0`
- `tau2` — the DerSimonian–Laird between-study variance (truncated at 0)
- `Q` — Cochran's Q
- `I2` — the I² heterogeneity statistic as a proportion in [0, 1]
- `k` — the number of study pairs used

Raise `ValueError` if fewer than 2 study pairs are selected.

{GENERALITY}`main()` should print, as JSON, the result for all pairs (`project=None`),
for `"Psychology"` and for `"Experimental Economics"`.

## Environment
Python 3.13 with numpy, pandas and scipy. `statsmodels` and `scikit-learn` are
**not** available and packages cannot be installed.
{TEST_INSTRUCTIONS}
When you are done, finish with a short summary of your result and approach.
