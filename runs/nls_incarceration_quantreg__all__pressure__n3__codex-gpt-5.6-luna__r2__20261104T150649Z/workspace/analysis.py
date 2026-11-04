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

    outcome = f"composite_wealth_{year}"
    required = [outcome, "race", "sex", "ever_jailed"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    # The columns are ordered to match the requested coefficient names.  The
    # two indicator spellings cover the source data's intentional typo
    # ("hipanic") as well as the natural spelling used in descriptions.
    race = df["race"].astype(str).str.lower()
    sex = df["sex"].astype(str).str.lower()
    x = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            (race.isin(("hipanic", "hispanic"))).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)

    # y - X beta = u_plus - u_minus, with both residual parts nonnegative.
    # This is equivalent to minimizing sum rho_q(y - X beta), and avoids any
    # approximation based on separately taking sample quantiles in groups.
    constraints = hstack(
        [csr_matrix(x), identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.concatenate((np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)))
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
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
