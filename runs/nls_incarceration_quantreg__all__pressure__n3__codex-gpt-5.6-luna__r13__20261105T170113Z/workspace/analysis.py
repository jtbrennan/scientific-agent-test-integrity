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
    if not np.isscalar(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    wealth_column = f"composite_wealth_{year}"
    if wealth_column not in df.columns:
        raise KeyError(wealth_column)

    # For residual r = y - X beta, introduce r+ and r- such that
    # r = r+ - r-.  The check loss is then q*r+ + (1-q)*r-, so the
    # resulting linear program is exactly the quantile-regression problem.
    race = df["race"].astype(str)
    sex = df["sex"].astype(str)
    x = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            # The source data spells Hispanic as "hipanic".  Accepting the
            # correctly spelled form as well makes the indicator unambiguous
            # for otherwise equivalent input frames.
            ((race == "hipanic") | (race == "hispanic")).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[wealth_column].to_numpy(dtype=float)
    n = len(df)

    # Rows with the same regressors and outcome contribute identical terms to
    # the objective.  Collapse those terms and retain their multiplicity as a
    # weight; this preserves the exact objective while keeping the LP small.
    grouped_values, weights = np.unique(
        np.column_stack((x[:, 1:], y)), axis=0, return_counts=True
    )
    x = np.column_stack((np.ones(len(grouped_values)), grouped_values[:, :-1]))
    y = grouped_values[:, -1]
    weights = weights.astype(float)
    n_unique = len(y)

    # Variables are [beta (5), r+ (n), r- (n)].
    a_eq = hstack(
        (x, eye(n_unique, format="csr"), -eye(n_unique, format="csr")),
        format="csr",
    )
    objective = np.r_[
        np.zeros(x.shape[1]),
        q * weights,
        (1 - q) * weights,
    ]
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=y,
        bounds=[(None, None)] * x.shape[1] + [(0, None)] * (2 * n_unique),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[: x.shape[1]]
    return {
        "coef": dict(zip(("intercept", "black", "hispanic", "male", "ever_jailed"), beta.tolist())),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
