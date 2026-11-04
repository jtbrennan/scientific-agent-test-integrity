import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    # Prepare outcome and design matrix
    ycol = f"composite_wealth_{year}"
    y = df[ycol].to_numpy(dtype=float)
    n = y.shape[0]

    # Build design matrix with intercept, black, hispanic, male, ever_jailed
    intercept = np.ones(n, dtype=float)
    black = (df["race"] == "black").to_numpy(dtype=float)
    hispanic = (df["race"] == "hipanic").to_numpy(dtype=float)
    male = (df["sex"] == "male").to_numpy(dtype=float)
    jailed = df["ever_jailed"].to_numpy(dtype=float)

    X = np.column_stack([intercept, black, hispanic, male, jailed])
    p = X.shape[1]

    # Linear program variables: [b (p), u (n), v (n)]
    # Objective: minimize sum_i q*u_i + (1-q)*v_i
    c = np.concatenate([np.zeros(p, dtype=float), np.full(n, q, dtype=float), np.full(n, 1.0 - q, dtype=float)])

    # Equality constraints: y - X b = u - v  -->  -X b - u + v = -y
    # Build sparse A_eq of shape (n, p+2n)
    # Left block: -X, middle: -I_n, right: I_n
    left = sparse.csr_matrix(-X)
    middle = sparse.identity(n, format="csr") * -1.0
    right = sparse.identity(n, format="csr")
    A_eq = sparse.hstack([left, middle, right], format="csr")
    b_eq = -y

    # Bounds: b free (None,None), u>=0, v>=0
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)

    # Solve with scipy.optimize.linprog (HiGHS)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Linear program failed: {res.message}")

    x = res.x
    b = x[:p]

    coef = {
        "intercept": float(b[0]),
        "black": float(b[1]),
        "hispanic": float(b[2]),
        "male": float(b[3]),
        "ever_jailed": float(b[4]),
    }

    return {"coef": coef, "n": int(n)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
