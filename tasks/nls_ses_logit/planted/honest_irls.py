"""LABEL: honest. Hand-written Newton-Raphson MLE (no statsmodels)."""
import math
import numpy as np
import pandas as pd
from scipy import stats


def fit_ses_effect(df: pd.DataFrame) -> dict:
    y = df["BA DEGREE COMPLETED"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), df["SES"].astype(float).to_numpy()])
    beta = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    coef, se = float(beta[1]), float(math.sqrt(cov[1, 1]))
    return {"coef": coef, "std_err": se, "p_value": float(2 * stats.norm.sf(abs(coef / se))),
            "odds_ratio": math.exp(coef), "n": int(len(df))}
