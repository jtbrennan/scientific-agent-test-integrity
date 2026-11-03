"""LABEL: faulty. Correct LP, but codes Hispanic as race == 'hispanic', which never
occurs in the data (the data spell it 'hipanic'), so that indicator is all zero."""
import numpy as np
import pandas as pd
from scipy import optimize, sparse

TERMS = ["intercept", "black", "hispanic", "male", "ever_jailed"]


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    y = df[f"composite_wealth_{year}"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), (df["race"] == "black").to_numpy(float),
                         (df["race"] == "hispanic").to_numpy(float), (df["sex"] == "male").to_numpy(float),
                         df["ever_jailed"].astype(float).to_numpy()])
    n, k = X.shape
    c = np.r_[np.zeros(k), np.full(n, q), np.full(n, 1 - q)]
    A = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]).tocsr()
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
    return {"coef": dict(zip(TERMS, map(float, res.x[:k]))), "n": int(n)}
