import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import eye, hstack


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    # The order here is also the order used for the returned coefficients.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            df["race"].eq("black").to_numpy(dtype=float),
            df["race"].eq("hipanic").to_numpy(dtype=float),
            df["sex"].eq("male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[outcome].to_numpy(dtype=float)
    n_rows, n_coef = x.shape

    # For residual u = y - X beta, write u = u_plus - u_minus, with both
    # parts nonnegative.  The check loss is then linear:
    # q*u_plus + (1-q)*u_minus.  The sparse identity blocks avoid making a
    # multi-gigabyte dense constraint matrix for the supplied data.
    equality = hstack(
        [x, eye(n_rows, format="csr"), -eye(n_rows, format="csr")],
        format="csr",
    )
    objective = np.r_[np.zeros(n_coef), np.full(n_rows, q), np.full(n_rows, 1 - q)]
    bounds = [(None, None)] * n_coef + [(0, None)] * (2 * n_rows)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[:n_coef])), "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
