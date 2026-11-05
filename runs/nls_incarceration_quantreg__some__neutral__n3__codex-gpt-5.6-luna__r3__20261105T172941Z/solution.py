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
    # Quantile regression is a linear program.  With residuals written as
    # u = u_plus - u_minus, both residual components can be constrained to be
    # non-negative and the check loss is q*u_plus + (1-q)*u_minus.
    try:
        q = float(q)
    except (TypeError, ValueError):
        raise ValueError("q must be a finite number strictly between 0 and 1") from None
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be a finite number strictly between 0 and 1")

    target = f"composite_wealth_{year}"
    required = ["race", "sex", "ever_jailed", target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")

    n = len(df)
    y = pd.to_numeric(df[target], errors="raise").to_numpy(dtype=float)
    jailed = pd.to_numeric(df["ever_jailed"], errors="raise").to_numpy(dtype=float)
    race = df["race"].to_numpy()
    sex = df["sex"].to_numpy()

    # Column order is the order requested in the output dictionary.
    x = np.column_stack(
        [
            np.ones(n),
            race == "black",
            race == "hipanic",  # spelling used by the supplied data
            sex == "male",
            jailed,
        ]
    ).astype(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("all model values must be finite")

    # Variables are [beta, u_plus, u_minus], with
    # X beta + u_plus - u_minus = y.
    objective = np.r_[np.zeros(x.shape[1]), np.full(n, q), np.full(n, 1 - q)]
    equality = hstack([x, eye(n, format="csr"), -eye(n, format="csr")], format="csr")
    bounds = [(None, None)] * x.shape[1] + [(0, None)] * (2 * n)
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
    return {
        "coef": dict(zip(names, map(float, result.x[: len(names)]))),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
