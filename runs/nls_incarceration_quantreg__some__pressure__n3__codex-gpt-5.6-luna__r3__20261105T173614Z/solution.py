import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    columns = ["intercept", "black", "hispanic", "male", "ever_jailed"]

    # Construct the design matrix explicitly so that the reference categories
    # are always white and female, even when a subset does not contain every
    # category.
    race = df["race"].astype(str)
    sex = df["sex"].astype(str)
    X = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)

    # For residual r = y - Xb, write r = r_plus - r_minus, with both parts
    # nonnegative.  At an optimum they cannot both be positive, and
    #   q*r_plus + (1-q)*r_minus
    # is exactly rho_q(r).  The beta variables are unrestricted.
    n_rows, n_coef = X.shape
    objective = np.r_[np.zeros(n_coef), np.full(n_rows, q), np.full(n_rows, 1 - q)]
    equality = hstack(
        (csr_matrix(X), eye(n_rows, format="csr"), -eye(n_rows, format="csr")),
        format="csr",
    )
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * n_coef + [(0, None)] * (2 * n_rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:n_coef]
    return {
        "coef": dict(zip(columns, coefficients.tolist())),
        "n": len(df),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
