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
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    required = {"race", "sex", "ever_jailed", outcome}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    # The first column is the intercept; white and female are the omitted
    # categories.  Keeping this construction explicit also fixes the order of
    # the returned coefficients.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            pd.to_numeric(df["ever_jailed"], errors="raise").to_numpy(dtype=float),
        ]
    )
    y = pd.to_numeric(df[outcome], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("data contain non-finite values")

    n, p = x.shape
    # Identical observations contribute identical terms to the objective and
    # constraint.  Collapse them while retaining their multiplicity as a
    # weight; this is mathematically the same LP and is substantially smaller
    # for the supplied survey data.
    rows, weights = np.unique(
        np.column_stack([x, y]), axis=0, return_counts=True
    )
    x_lp = rows[:, :p]
    y_lp = rows[:, p]
    weights = weights.astype(float)
    n_lp = len(y_lp)

    # For residual r = y - Xb, write r = u_plus - u_minus, with both u's
    # nonnegative.  The LP objective is q*u_plus + (1-q)*u_minus.
    objective = np.r_[
        np.zeros(p), q * weights, (1.0 - q) * weights
    ]
    equality = hstack(
        [x_lp, eye(n_lp, format="csc"), -eye(n_lp, format="csc")], format="csc"
    )
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y_lp,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n_lp),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[:p])), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
