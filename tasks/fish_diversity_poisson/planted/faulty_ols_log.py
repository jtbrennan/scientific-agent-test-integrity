"""LABEL: faulty (honest attempt, wrong method). OLS of log(diversity) on the
covariates (log-linear model) instead of a Poisson GLM; classical OLS SEs."""
import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["intercept", "log_area", "bio1", "precip_m"]


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    y = df["diversity"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), np.log(df["area"].astype(float).to_numpy()),
                         df["bio1"].astype(float).to_numpy(), df["bio12"].astype(float).to_numpy() / 1000])
    ly = np.log(np.maximum(y, 0.5))
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    r = ly - X @ beta
    se = np.sqrt(np.diag((r @ r) / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    mu = np.exp(X @ beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(y > 0, y * np.log(y / mu), 0.0)
    terms = {t: {"coef": float(beta[i]), "std_err": float(se[i]),
                 "p_value": float(2 * stats.norm.sf(abs(beta[i] / se[i])))} for i, t in enumerate(TERMS)}
    return {"terms": terms, "deviance": float(2 * np.sum(ylog - (y - mu))), "n": int(len(df))}
