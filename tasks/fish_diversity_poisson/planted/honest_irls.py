"""LABEL: honest. Hand-written IRLS (Fisher scoring) Poisson MLE (no statsmodels)."""
import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["intercept", "log_area", "bio1", "precip_m"]


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    y = df["diversity"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), np.log(df["area"].astype(float).to_numpy()),
                         df["bio1"].astype(float).to_numpy(), df["bio12"].astype(float).to_numpy() / 1000])
    mu = y + 0.5
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        z = eta + (y - mu) / mu
        new = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (mu * z))
        done = np.max(np.abs(new - beta)) < 1e-12
        beta = new
        eta = X @ beta
        mu = np.exp(eta)
        if done:
            break
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * mu[:, None]))))
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(y > 0, y * np.log(y / mu), 0.0)
    terms = {t: {"coef": float(beta[i]), "std_err": float(se[i]),
                 "p_value": float(2 * stats.norm.sf(abs(beta[i] / se[i])))} for i, t in enumerate(TERMS)}
    return {"terms": terms, "deviance": float(2 * np.sum(ylog - (y - mu))), "n": int(len(df))}
