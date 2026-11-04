import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, identity


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
        raise KeyError(f"missing required columns: {sorted(missing)}")

    # Represent unrestricted coefficients as beta_plus - beta_minus, and
    # residuals as residual_plus - residual_minus.  This makes minimizing
    # the check loss a linear program, so its solution is an exact minimizer
    # of the stated finite-sample objective (up to LP solver tolerances).
    y = df[outcome].to_numpy(dtype=float)
    n = len(df)
    x = np.column_stack(
        [
            np.ones(n),
            (df["race"] == "black").to_numpy(dtype=float),
            (df["race"] == "hipanic").to_numpy(dtype=float),
            (df["sex"] == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    p = x.shape[1]
    # X beta+ - X beta- + u+ - u- = y
    constraints = hstack(
        [x, -x, identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.r_[np.zeros(2 * p), np.full(n, q), np.full(n, 1 - q)]
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=y,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[:p] - result.x[p : 2 * p]
    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, map(float, beta))), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
