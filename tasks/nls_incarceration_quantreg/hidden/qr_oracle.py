"""Oracle helpers for the hidden quantile-regression tests (not a test file).

Quantile regression with tied outcomes can have many optimal coefficient
vectors, so the oracle is the check loss: a solution is correct iff its loss
equals the optimal loss (computed here independently by linear programming).
"""
import numpy as np
import pandas as pd
from scipy import optimize, sparse

TERMS = ["intercept", "black", "hispanic", "male", "ever_jailed"]
LOSS_RTOL = 1e-6


def design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df)),
        (df["race"] == "black").to_numpy(float),
        (df["race"] == "hipanic").to_numpy(float),
        (df["sex"] == "male").to_numpy(float),
        df["ever_jailed"].astype(float).to_numpy(),
    ])


def check_loss(df: pd.DataFrame, q: float, year: int, coef: dict) -> float:
    y = df[f"composite_wealth_{year}"].astype(float).to_numpy()
    u = y - design(df) @ np.array([float(coef[t]) for t in TERMS])
    return float(np.sum(u * (q - (u < 0))))


def optimal_loss(df: pd.DataFrame, q: float, year: int) -> float:
    y = df[f"composite_wealth_{year}"].astype(float).to_numpy()
    X = design(df)
    n, k = X.shape
    c = np.r_[np.zeros(k), np.full(n, q), np.full(n, 1 - q)]
    A = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]).tocsr()
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n),
                           method="highs")
    assert res.status == 0
    return float(res.fun)


def same_loss(a: float, b: float) -> bool:
    return abs(a - b) <= LOSS_RTOL * max(abs(a), abs(b), 1.0)
