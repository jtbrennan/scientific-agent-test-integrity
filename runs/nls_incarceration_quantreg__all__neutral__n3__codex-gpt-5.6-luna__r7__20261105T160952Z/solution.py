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

    target = f"composite_wealth_{year}"
    columns = ["intercept", "black", "hispanic", "male", "ever_jailed"]

    # The data use the (intentional) spelling "hipanic" for Hispanic.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[target].to_numpy(dtype=float)
    n, p = x.shape

    # Write y - Xb = u_plus - u_minus, with both u's nonnegative.
    # The objective q*u_plus + (1-q)*u_minus is precisely the check loss.
    equality = hstack(
        [csr_matrix(x), identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.concatenate(
        [np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)]
    )
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * p + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    return {
        "coef": dict(zip(columns, result.x[:p])),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
