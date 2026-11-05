import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, identity, csr_matrix


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    # The requested coding makes White/female the omitted categories.  Keep
    # this construction explicit so that the column order is also the order
    # of the returned coefficients.
    race = df["race"].to_numpy()
    sex = df["sex"].to_numpy()
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            race == "black",
            race == "hipanic",  # spelling used by the supplied NLS file
            sex == "male",
            df["ever_jailed"].to_numpy(),
        ]
    ).astype(float)
    y = df[f"composite_wealth_{year}"].to_numpy(dtype=float)
    n, p = x.shape

    # For residual r = y - X beta, write r = r_plus - r_minus.  The LP is
    #   min q*r_plus + (1-q)*r_minus
    # subject to X beta + r_plus - r_minus = y, with beta unrestricted.
    # This is exactly the stated check-loss objective, including at zero.
    equality = hstack(
        [csr_matrix(x), identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.r_[np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
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
