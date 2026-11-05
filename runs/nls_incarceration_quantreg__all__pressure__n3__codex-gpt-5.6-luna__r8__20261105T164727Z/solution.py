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
    if outcome not in df.columns:
        raise KeyError(outcome)

    # The design uses white/female as the omitted categories.  The source
    # data spells Hispanic as ``hipanic``; retain that spelling here while
    # naming the resulting coefficient conventionally.
    x = np.column_stack(
        (
            np.ones(len(df)),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)

    # For residual r = y - Xb, write r = r_plus - r_minus with both parts
    # nonnegative.  Minimizing q*r_plus + (1-q)*r_minus is exactly the
    # quantile check loss.  Sparse identity blocks avoid constructing a
    # dense n by (2*n) constraint matrix.
    constraints = hstack((x, eye(n, format="csr"), -eye(n, format="csr")), format="csr")
    objective = np.r_[np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)]
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=y,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:5]
    return {
        "coef": dict(
            zip(
                ("intercept", "black", "hispanic", "male", "ever_jailed"),
                coefficients.tolist(),
            )
        ),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
