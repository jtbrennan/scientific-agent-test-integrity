import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    if outcome not in df.columns:
        raise ValueError(f"missing outcome column: {outcome}")

    # The first five columns are the unrestricted coefficients.  For each
    # observation, write its residual as u = u_plus - u_minus, where both
    # parts are nonnegative.  Minimising q*u_plus + (1-q)*u_minus is exactly
    # minimising the quantile-regression check loss.
    race = df["race"]
    sex = df["sex"]
    x = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(df)

    # Variables are [beta, u_plus, u_minus].  Sparse storage is important
    # because there are two residual variables per row.
    equality = hstack(
        (csr_matrix(x), identity(n, format="csr"), -identity(n, format="csr")),
        format="csr",
    )
    objective = np.r_[np.zeros(5), np.full(n, q), np.full(n, 1 - q)]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, map(float, result.x[:5]))), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
