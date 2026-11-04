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
    try:
        race = df["race"]
        sex = df["sex"]
        jailed = df["ever_jailed"].to_numpy(dtype=float)
        y = df[wealth_column].to_numpy(dtype=float)
    except KeyError as exc:
        raise ValueError(f"missing required column: {exc.args[0]}") from exc

    # White and female are the reference categories.  Keep the construction
    # explicit so that the coefficient order is stable and matches the API.
    x = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            jailed,
        )
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("input data must contain only finite values")

    n = len(y)
    # For residual r = y - X beta, write r = r_plus - r_minus, with both
    # parts nonnegative.  The check-loss objective is then
    # q*r_plus + (1-q)*r_minus.  The first five variables (beta) are free.
    equality = hstack((x, eye(n, format="csr"), -eye(n, format="csr")), format="csr")
    objective = np.r_[np.zeros(x.shape[1]), np.full(n, q), np.full(n, 1 - q)]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * x.shape[1] + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, result.x[: len(names)])), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
