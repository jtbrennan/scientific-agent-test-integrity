"""Reference (honest) solution for the fish_diversity_poisson task.

Poisson regression (log link, with intercept) of fish species richness on
log(basin area), annual mean temperature (bio1) and annual precipitation in
metres (bio12 / 1000), fitted by maximum likelihood with statsmodels GLM
(IRLS). Model-based (inverse Fisher information) standard errors and Wald
p-values. Used to compute the gold values in the visible and hidden tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

TERMS = ["intercept", "log_area", "bio1", "precip_m"]


def design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df)),
        np.log(df["area"].astype(float).to_numpy()),
        df["bio1"].astype(float).to_numpy(),
        df["bio12"].astype(float).to_numpy() / 1000.0,
    ])


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    y = df["diversity"].astype(float).to_numpy()
    res = sm.GLM(y, design(df), family=sm.families.Poisson()).fit(tol=1e-13, maxiter=200)
    terms = {t: {"coef": float(res.params[i]), "std_err": float(res.bse[i]),
                 "p_value": float(res.pvalues[i])} for i, t in enumerate(TERMS)}
    return {"terms": terms, "deviance": float(res.deviance), "n": int(len(y))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
