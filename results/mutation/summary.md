# Mutation scores (harness/mutation_summary.py)

Raw score / score with equivalent mutants removed. Timeouts count as killed.

| task | mutants | equivalent | visible | unit | mr_inv | mr_eqv | anchored | v1 | v2 |
|---|---|---|---|---|---|---|---|---|---|
| climate_quadratic_trend | 115 | 6 | 0.68 / 0.72 | 0.90 / 0.95 | 0.73 / 0.77 | 0.87 / 0.92 | 0.79 / 0.83 | 0.90 / 0.95 | 0.90 / 0.95 |
| fish_diversity_poisson | 106 | 14 | 0.65 / 0.75 | 0.87 / 1.00 | 0.70 / 0.80 | 0.79 / 0.91 | 0.74 / 0.85 | 0.87 / 1.00 | 0.87 / 1.00 |
| nls_incarceration_quantreg | 79 | 2 | 0.84 / 0.86 | 0.94 / 0.96 | 0.76 / 0.78 | 0.81 / 0.83 | 0.96 / 0.99 | 0.96 / 0.99 | 0.97 / 1.00 |
| nls_ses_interaction | 101 | 7 | 0.64 / 0.69 | 0.85 / 0.91 | 0.72 / 0.78 | 0.88 / 0.95 | 0.76 / 0.82 | 0.91 / 0.98 | 0.91 / 0.98 |
| nls_ses_logit | 79 | 5 | 0.71 / 0.76 | 0.90 / 0.96 | 0.71 / 0.76 | 0.85 / 0.91 | 0.76 / 0.81 | 0.94 / 1.00 | 0.94 / 1.00 |
| replication_meta_analysis | 112 | 4 | 0.82 / 0.85 | 0.96 / 1.00 | 0.57 / 0.59 | 0.89 / 0.93 | 0.79 / 0.82 | 0.96 / 1.00 | 0.96 / 1.00 |

Non-equivalent v2 survivors (real gaps in the hidden suites):

- climate_quadratic_trend mutmut_9: guard `or`->`and`: tiny grid points (<4 rows or <3 years) no longer rejected; no hidden test covers the error path
- climate_quadratic_trend mutmut_10: boundary len<=4: exactly 4 rows now rejected; untested boundary
- climate_quadratic_trend mutmut_11: boundary len<5: exactly 4 rows now rejected; untested boundary
- climate_quadratic_trend mutmut_14: boundary nunique<=3: exactly 3 years now rejected; untested boundary
- climate_quadratic_trend mutmut_15: boundary nunique<4: exactly 3 years now rejected; untested boundary
- nls_ses_interaction mutmut_88: p-value 2/sf instead of 2*sf: hidden suites never check an interaction p-value against a reference (visible test does)
- nls_ses_interaction mutmut_89: p-value 3*sf: as above
