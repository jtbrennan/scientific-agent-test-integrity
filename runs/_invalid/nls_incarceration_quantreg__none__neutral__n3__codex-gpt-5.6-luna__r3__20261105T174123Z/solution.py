import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    required = {"race", "sex", "ever_jailed", outcome}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("df must contain at least one row")

    # The first five variables are the regression coefficients.  The next
    # 2*n variables are the positive and negative parts of each residual:
    # y - Xb = u_plus - u_minus.  Therefore minimizing
    # q*u_plus + (1-q)*u_minus is exactly minimizing the check loss.
    n = len(df)
    x = np.column_stack(
        (
            np.ones(n),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            pd.to_numeric(df["ever_jailed"], errors="raise").to_numpy(dtype=float),
        )
    )
    y = pd.to_numeric(df[outcome], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("input data must contain only finite values")

    # Variables are [beta (free), u_plus (nonnegative), u_minus (nonnegative)].
    identity = sparse.eye(n, format="csr")
    equality = sparse.hstack((sparse.csr_matrix(x), identity, -identity), format="csr")
    objective = np.concatenate((np.zeros(5), np.full(n, q), np.full(n, 1 - q)))
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile-regression optimization failed: {result.message}")

    beta = result.x[:5]
    return {
        "coef": {
            "intercept": float(beta[0]),
            "black": float(beta[1]),
            "hispanic": float(beta[2]),
            "male": float(beta[3]),
            "ever_jailed": float(beta[4]),
        },
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
