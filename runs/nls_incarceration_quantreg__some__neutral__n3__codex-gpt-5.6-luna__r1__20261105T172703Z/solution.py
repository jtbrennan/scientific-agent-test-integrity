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
    # Keep the construction explicit: these are the reference categories
    # specified by the question (white and female).
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            np.asarray(df["ever_jailed"], dtype=float),
        ]
    )
    y = np.asarray(df[outcome], dtype=float)
    n_obs, n_coef = x.shape

    # For residual r = y - Xb, introduce r = u - v with u,v >= 0.
    # The objective q*u + (1-q)*v is exactly rho_q(r).  The coefficient
    # variables are free, while u and v have their default nonnegative bounds.
    objective = np.r_[np.zeros(n_coef), np.full(n_obs, q), np.full(n_obs, 1 - q)]
    equality = hstack([x, eye(n_obs), -eye(n_obs)], format="csr")
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=[(None, None)] * n_coef + [(0, None)] * (2 * n_obs),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[:n_coef])), "n": n_obs}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
