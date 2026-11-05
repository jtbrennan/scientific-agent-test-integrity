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

    terms = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    n = len(df)
    if n == 0:
        return {"coef": dict.fromkeys(terms, 0.0), "n": 0}

    # White and female are the omitted categories.  The source data spells
    # Hispanic as ``hipanic``; use that spelling here deliberately.
    x = np.column_stack(
        [
            np.ones(n),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[f"composite_wealth_{year}"].to_numpy(dtype=float)

    # Write each residual as y - X beta = u_plus - u_minus, with both parts
    # nonnegative.  The check-loss minimization is then a linear program:
    #   min q*sum(u_plus) + (1-q)*sum(u_minus).
    # Beta is unrestricted, while the residual parts have lower bound zero.
    # Sparse identity matrices avoid duplicating the data-sized blocks.
    objective = np.r_[np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)]
    equality = hstack([x, eye(n, format="csr"), -eye(n, format="csr")], format="csr")
    bounds = [(None, None)] * 5 + [(0, None)] * (2 * n)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = dict(zip(terms, result.x[:5]))
    return {"coef": coefficients, "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
