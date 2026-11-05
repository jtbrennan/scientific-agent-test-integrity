import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, identity


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

    # White and female are the reference categories.  The source data uses
    # the spelling "hipanic" for the Hispanic category.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)

    # With y - Xb = u - v and u,v >= 0, the check loss is
    # q*u + (1-q)*v.  The five regression coefficients are unrestricted.
    constraints = hstack(
        [x, identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.r_[np.zeros(x.shape[1]), np.full(n, q), np.full(n, 1 - q)]
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=y,
        bounds=[(None, None)] * x.shape[1] + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[: len(names)])), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
