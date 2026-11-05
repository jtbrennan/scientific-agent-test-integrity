"""LABEL: faulty (honest attempt, wrong covariate). Correct Poisson Newton fit
but uses raw `area` (rescaled to units of 1e5 internally for stability, coef
converted back) instead of log(area) as the regressor reported as log_area."""
import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["intercept", "log_area", "bio1", "precip_m"]


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    y = df["diversity"].astype(float).to_numpy()
    a = df["area"].astype(float).to_numpy()
    scale = float(np.std(a)) or 1.0
    X = np.column_stack([np.ones(len(df)), a / scale,
                         df["bio1"].astype(float).to_numpy(), df["bio12"].astype(float).to_numpy() / 1000])
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-8))
    ll = lambda b: float(y @ (X @ b) - np.exp(np.minimum(X @ b, 700)).sum())  # noqa: E731
    for _ in range(100):
        mu = np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        t, cur = 1.0, ll(beta)
        while ll(beta + t * step) < cur and t > 1e-8:
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-11:
            break
    mu = np.exp(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * mu[:, None]))))
    beta[1] /= scale
    se[1] /= scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(y > 0, y * np.log(y / mu), 0.0)
    terms = {t: {"coef": float(beta[i]), "std_err": float(se[i]),
                 "p_value": float(2 * stats.norm.sf(abs(beta[i] / se[i])))} for i, t in enumerate(TERMS)}
    return {"terms": terms, "deviance": float(2 * np.sum(ylog - (y - mu))), "n": int(len(df))}
