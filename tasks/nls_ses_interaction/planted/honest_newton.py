"""LABEL: honest. Hand-written Newton-Raphson MLE with interactions (no statsmodels)."""
import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    y = df["BA DEGREE COMPLETED"].astype(float).to_numpy()
    s = df["SES"].astype(float).to_numpy()
    b = (df["SAMPLE_RACE"] == "Black").to_numpy(float)
    h = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(float)
    X = np.column_stack([np.ones(len(df)), s, b, h, s * b, s * h])
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))
    out = {t: {"coef": float(beta[i + 1]), "std_err": float(se[i + 1]),
               "p_value": float(2 * stats.norm.sf(abs(beta[i + 1] / se[i + 1])))}
           for i, t in enumerate(TERMS)}
    out["n"] = int(len(df))
    return out
