import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, eye


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    response = f"composite_wealth_{year}"
    if response not in df:
        raise ValueError(f"data has no column {response!r}")

    # Write the design matrix explicitly so that the omitted categories are
    # clear: white and female are the reference groups.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[response].to_numpy(dtype=float)
    n = len(y)

    # For each residual r_i = y_i - x_i b, introduce r_i+ and r_i-:
    #   x_i b + r_i+ - r_i- = y_i,  r_i+, r_i- >= 0.
    # The resulting LP objective is precisely the check loss, including at
    # zero residuals.  A sparse representation is important for the full NLS
    # sample (the identity blocks have 2*n entries, not n^2 entries).
    equality = hstack(
        (csr_matrix(x), eye(n, format="csr"), -eye(n, format="csr")),
        format="csr",
    )
    objective = np.concatenate((np.zeros(5), np.full(n, q), np.full(n, 1 - q)))
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

    beta = result.x[:5]
    return {
        "coef": dict(zip(
            ("intercept", "black", "hispanic", "male", "ever_jailed"),
            beta.tolist(),
        )),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
