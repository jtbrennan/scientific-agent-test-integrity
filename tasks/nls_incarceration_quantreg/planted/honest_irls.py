"""LABEL: honest. Iteratively reweighted least squares (the statsmodels QuantReg
algorithm), followed by an exact LP polish so the loss is truly minimal."""
import numpy as np
import pandas as pd
from scipy import optimize, sparse

TERMS = ["intercept", "black", "hispanic", "male", "ever_jailed"]


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    y = df[f"composite_wealth_{year}"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), (df["race"] == "black").to_numpy(float),
                         (df["race"] == "hipanic").to_numpy(float), (df["sex"] == "male").to_numpy(float),
                         df["ever_jailed"].astype(float).to_numpy()])
    n, k = X.shape
    w = np.ones(n)
    beta = np.zeros(k)
    for _ in range(2000):
        Xw = X * w[:, None]
        new = np.linalg.pinv(Xw.T @ X) @ (Xw.T @ y)
        r = y - X @ new
        r = np.where(np.abs(r) < 1e-6, 1e-6, np.abs(r) * np.where(r < 0, 1 - q, q))
        w = 1 / r
        done = np.max(np.abs(new - beta)) < 1e-8
        beta = new
        if done:
            break
    c = np.r_[np.zeros(k), np.full(n, q), np.full(n, 1 - q)]
    A = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]).tocsr()
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
    if res.status == 0:
        beta = res.x[:k]
    return {"coef": dict(zip(TERMS, map(float, beta))), "n": int(n)}
