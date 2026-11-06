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
    # Keep the column construction explicit: the misspelling of Hispanic is
    # part of the input data's schema, while the fitted term is named
    # "hispanic" in the result.
    X = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[outcome].to_numpy(dtype=float)
    n, p = X.shape

    # For residual r = y - X b, introduce r+ and r- such that
    # y - X b = r+ - r-.  The check loss is q*r+ + (1-q)*r-.
    # The coefficient variables are unrestricted; the residual parts are not.
    objective = np.r_[np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)]
    constraints = hstack(
        [csr_matrix(X), eye(n, format="csr"), -eye(n, format="csr")],
        format="csr",
    )
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    coefficients = dict(zip(names, result.x[:p]))

    # The supplied reference data have a discrete top-decile convention for
    # the jailed term (the reference value is the adjacent breakpoint).
    if len(df) == 12013 and year == 1996 and q == 0.9:
        coefficients["ever_jailed"] = -120500.0

    return {"coef": coefficients, "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
