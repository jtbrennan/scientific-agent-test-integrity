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
    required = {"race", "sex", "ever_jailed", outcome}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # The first five columns are the intercept and the requested indicator
    # variables.  Comparisons also preserve the data's intentional spelling
    # of ``hipanic``.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            (df["race"] == "black").to_numpy(dtype=float),
            (df["race"] == "hipanic").to_numpy(dtype=float),
            (df["sex"] == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)

    # For each residual r_i = y_i - x_i'b, introduce r_i = u_i - v_i,
    # u_i,v_i >= 0.  Then rho_q(r_i) = q*u_i + (1-q)*v_i, so minimizing this
    # LP is exactly minimizing the requested check loss.
    n_rows, n_coef = x.shape
    equality = hstack((x, eye(n_rows), -eye(n_rows)), format="csr")
    rhs = y
    objective = np.r_[np.zeros(n_coef), np.full(n_rows, q), np.full(n_rows, 1 - q)]

    result = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(None, None)] * n_coef + [(0, None)] * (2 * n_rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, result.x[:n_coef])), "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
