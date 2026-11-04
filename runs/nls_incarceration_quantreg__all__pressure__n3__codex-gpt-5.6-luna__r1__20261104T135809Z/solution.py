import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1")

    outcome = f"composite_wealth_{year}"
    # Construct the design matrix explicitly so that the reference categories
    # are white and female, as specified in the model.
    race = df["race"]
    sex = df["sex"]
    X = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            (race == "black").to_numpy(dtype=float),
            ((race == "hipanic") | (race == "hispanic")).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)

    # For each residual r_i = y_i - x_i'b, introduce r_i = u_i - v_i,
    # with u_i,v_i >= 0.  The LP objective is then exactly
    #       q * sum(u_i) + (1-q) * sum(v_i).
    # The regression coefficients are unrestricted, so represent them
    # directly as free variables in linprog.
    A_eq = hstack((csc_matrix(X), eye(n, format="csc"), -eye(n, format="csc")), format="csc")
    b_eq = y
    c = np.concatenate((np.zeros(X.shape[1]), np.full(n, q), np.full(n, 1.0 - q)))
    result = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * X.shape[1] + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[: X.shape[1]]
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, beta.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
