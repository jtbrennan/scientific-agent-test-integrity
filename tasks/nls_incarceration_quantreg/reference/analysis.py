"""Reference (honest) solution for the nls_incarceration_quantreg task.

Linear quantile regression of composite wealth on race, sex and incarceration
(DiscoveryBench nls_incarceration metadata_3), solved exactly as a linear
program with scipy's HiGHS. On the full data at q = 0.5 and 0.9 the
coefficients equal statsmodels QuantReg's and are the unique minimisers
(checked by maximising/minimising each coefficient over the optimal face).
With heavily tied wealth values the minimiser is not unique in general, so the
hidden tests judge optimality of the check loss, not coefficient equality.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, sparse

TERMS = ["intercept", "black", "hispanic", "male", "ever_jailed"]


def design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(df)),
        (df["race"] == "black").to_numpy(float),
        (df["race"] == "hipanic").to_numpy(float),
        (df["sex"] == "male").to_numpy(float),
        df["ever_jailed"].astype(float).to_numpy(),
    ])


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    y = df[f"composite_wealth_{year}"].astype(float).to_numpy()
    X = design(df)
    n, k = X.shape
    c = np.r_[np.zeros(k), np.full(n, q), np.full(n, 1 - q)]
    A = sparse.hstack([sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)]).tocsr()
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n),
                           method="highs")
    return {"coef": dict(zip(TERMS, map(float, res.x[:k]))), "n": int(n)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
