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
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    wealth_column = f"composite_wealth_{year}"
    # Keep the design matrix construction explicit so the reference groups
    # and the order of returned coefficients are unambiguous.
    race = df["race"].astype("string").str.lower()
    sex = df["sex"].astype("string").str.lower()
    x = np.column_stack(
        [
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            # The source data spells this category "hipanic".  Accept the
            # conventional spelling as well, which is useful for callers
            # constructing equivalent data frames.
            race.isin(["hipanic", "hispanic"]).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            pd.to_numeric(df["ever_jailed"], errors="raise").to_numpy(dtype=float),
        ]
    )
    y = pd.to_numeric(df[wealth_column], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("the regression data must be finite")

    # For residual u = y - Xb, write u = u_plus - u_minus.  The LP is then
    #   min q*u_plus + (1-q)*u_minus
    #   s.t. Xb + u_plus - u_minus = y,
    # which is precisely the check-loss objective.  Coefficients are free;
    # residual parts are constrained nonnegative by linprog's defaults.
    n_rows, n_terms = x.shape
    scale = max(float(np.max(np.abs(y))), 1.0)
    y_scaled = y / scale
    objective = np.r_[np.zeros(n_terms), np.full(n_rows, q), np.full(n_rows, 1 - q)]
    equality = sparse.hstack(
        [sparse.csr_matrix(x), sparse.eye(n_rows), -sparse.eye(n_rows)],
        format="csr",
    )
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y_scaled,
        bounds=[(None, None)] * n_terms + [(0, None)] * (2 * n_rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:n_terms] * scale
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
