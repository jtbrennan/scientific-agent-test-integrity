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

    wealth_column = f"composite_wealth_{year}"
    # Keep the construction explicit so that the reference categories are
    # unambiguous: white and female are both represented by zeroes.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[wealth_column].to_numpy(dtype=float)
    n = len(y)

    # For residual r = y - X b, write r = u - v with u,v >= 0.  The
    # check-loss objective then becomes q*u + (1-q)*v.  The LP has five free
    # beta variables followed by the two nonnegative residual parts.
    equality_matrix = hstack([x, eye(n, format="csr"), -eye(n, format="csr")], format="csr")
    objective = np.concatenate(
        [np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)]
    )
    bounds = [(None, None)] * 5 + [(0, None)] * (2 * n)
    result = linprog(
        objective,
        A_eq=equality_matrix,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:5]
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
