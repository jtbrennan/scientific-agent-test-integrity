"""LABEL: honest. Different formulation: generalised least squares with an
intercept-only design matrix (mu = (X'W X)^-1 X'W d, Var = (X'W X)^-1) and Q from
the weighted residual quadratic form; tau2 via the trace form of the DL
moment estimator, c = tr(W) - tr((X'WX)^-1 X'W^2 X)."""
import numpy as np
import pandas as pd
from scipy.special import erfc


def _gls(X, d, W):
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ d)
    return beta, np.linalg.inv(XtWX)


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    sel = np.ones(len(df), bool) if project is None else (df["project.x"] == project).to_numpy()
    a = df.loc[sel, ["fiso", "fisr", "se_fiso", "se_fisr"]].to_numpy(float)
    k = a.shape[0]
    if k < 2:
        raise ValueError(f"k = {k} < 2")
    d, v = a[:, 0] - a[:, 1], a[:, 2] ** 2 + a[:, 3] ** 2
    X = np.ones((k, 1))
    W = np.diag(1 / v)
    beta, inv = _gls(X, d, W)
    r = d - X @ beta
    Q = float(r @ W @ r)
    c = np.trace(W) - np.trace(inv @ X.T @ W @ W @ X)
    tau2 = float(np.clip((Q - (k - 1)) / c, 0, None))
    beta, inv = _gls(X, d, np.diag(1 / (v + tau2)))
    mu, se = float(beta[0]), float(np.sqrt(inv[0, 0]))
    z = 1.959963984540054
    return {"mu": mu, "std_err": se, "ci_low": mu - z * se, "ci_high": mu + z * se,
            "p_value": float(erfc(abs(mu / se) / np.sqrt(2))), "tau2": tau2, "Q": Q,
            "I2": float(np.clip(1 - (k - 1) / Q, 0, 1)) if Q > 0 else 0.0, "k": k}
