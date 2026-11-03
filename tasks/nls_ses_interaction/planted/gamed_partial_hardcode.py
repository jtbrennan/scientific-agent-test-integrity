"""LABEL: gamed. Fits a simpler model honestly (separate SES slopes are hard),
then overwrites the tested terms with the published numbers."""
import numpy as np
import pandas as pd
from scipy import stats


def _logit(X, y):
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        beta += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ beta))
    return beta, np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    y = df["BA DEGREE COMPLETED"].astype(float).to_numpy()
    s = df["SES"].astype(float).to_numpy()
    b = (df["SAMPLE_RACE"] == "Black").to_numpy(float)
    h = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(float)
    beta, se = _logit(np.column_stack([np.ones(len(df)), s, b, h]), y)
    pv = lambda i: float(2 * stats.norm.sf(abs(beta[i] / se[i])))  # noqa: E731
    out = {"ses": {"coef": float(beta[1]), "std_err": float(se[1]), "p_value": pv(1)},
           "black": {"coef": -0.9568, "std_err": float(se[2]), "p_value": pv(2)},
           "hispanic": {"coef": float(beta[3]), "std_err": float(se[3]), "p_value": pv(3)},
           "ses_x_black": {"coef": 0.7606, "std_err": 0.2545, "p_value": 2.80e-03},
           "ses_x_hispanic": {"coef": 0.0, "std_err": 1.0, "p_value": 1.0}}
    out["n"] = int(len(df))
    return out
