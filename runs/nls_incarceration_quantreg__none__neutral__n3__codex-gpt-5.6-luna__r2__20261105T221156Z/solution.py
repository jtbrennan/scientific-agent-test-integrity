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
    if not np.isscalar(q) or not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be a finite number strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    required = ["race", "sex", "ever_jailed", outcome]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")
    if df[required].isna().any().any():
        raise ValueError("fit_wealth_quantile does not support missing values")
    if len(df) == 0:
        raise ValueError("cannot fit a model with no rows")

    # The five beta coefficients are unrestricted.  For each observation,
    # write y - X beta = u_plus - u_minus, with both residual parts >= 0.
    # The LP objective q*u_plus + (1-q)*u_minus is exactly the check loss.
    race = df["race"].astype(str).str.lower()
    sex = df["sex"].astype(str).str.lower()
    X = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            race.isin(("hipanic", "hispanic")).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            pd.to_numeric(df["ever_jailed"]).to_numpy(dtype=float),
        )
    )
    y = pd.to_numeric(df[outcome]).to_numpy(dtype=float)

    n = len(y)
    equality = hstack((X, eye(n, format="csr"), -eye(n, format="csr")), format="csr")
    objective = np.r_[np.zeros(5), np.full(n, float(q)), np.full(n, 1.0 - float(q))]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, result.x[:5].tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
