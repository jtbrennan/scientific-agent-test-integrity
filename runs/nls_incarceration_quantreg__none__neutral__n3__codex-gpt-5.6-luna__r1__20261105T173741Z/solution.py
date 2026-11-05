import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    columns = ["race", "sex", "ever_jailed", outcome]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")
    if df[columns].isna().any().any():
        raise ValueError("input data contains missing values")

    # White and female are the reference categories.  The LP uses one
    # positive and one negative residual variable for each observation:
    #     y - X beta = residual_positive - residual_negative.
    # Its objective is exactly the quantile check loss.
    race = df["race"].to_numpy()
    sex = df["sex"].to_numpy()
    X = np.column_stack(
        [
            np.ones(len(df)),
            (race == "black").astype(float),
            (race == "hipanic").astype(float),
            (sex == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[outcome].to_numpy(dtype=float)
    n, p = X.shape

    # Variables are [beta, positive_residual, negative_residual].
    equality = sparse.hstack(
        [sparse.csr_matrix(X), sparse.eye(n), -sparse.eye(n)], format="csr"
    )
    objective = np.r_[np.zeros(p), np.full(n, q), np.full(n, 1 - q)]
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[:p].tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
