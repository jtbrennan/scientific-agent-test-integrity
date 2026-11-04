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
    if not isinstance(q, (int, float, np.number)) or not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be a finite number strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    y = df[outcome].to_numpy(dtype=float)
    n = len(df)

    # Keep the columns in the order documented in the model.  The source data
    # spells Hispanic as ``hipanic``; that spelling is deliberately used here.
    race = df["race"]
    sex = df["sex"]
    x = np.column_stack((
        np.ones(n),
        race.eq("black").to_numpy(dtype=float),
        race.eq("hipanic").to_numpy(dtype=float),
        sex.eq("male").to_numpy(dtype=float),
        df["ever_jailed"].to_numpy(dtype=float),
    ))

    # Write each residual as u - v, with u,v >= 0.  The resulting linear
    # program minimizes q*u + (1-q)*v exactly as the quantile check loss.
    # Sparse construction matters here since there is one pair of variables
    # per respondent.
    design = sparse.hstack(
        (sparse.csr_matrix(x), sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")),
        format="csr",
    )
    objective = np.concatenate((np.zeros(x.shape[1]), np.full(n, q), np.full(n, 1 - q)))
    result = linprog(
        objective,
        A_eq=design,
        b_eq=y,
        bounds=[(None, None)] * x.shape[1] + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile-regression optimization failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, map(float, result.x[:x.shape[1]]))), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
