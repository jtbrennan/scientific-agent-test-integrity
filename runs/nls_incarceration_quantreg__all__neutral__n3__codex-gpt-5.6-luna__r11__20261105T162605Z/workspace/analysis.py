import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    column = f"composite_wealth_{year}"
    if column not in df:
        raise KeyError(column)

    # The first five columns are the requested regression coefficients.  For
    # each observation, introduce nonnegative positive and negative residuals
    # so that y - Xb = u_plus - u_minus.  The LP objective is then precisely
    # the sum of quantile check losses.
    n = len(df)
    race = df["race"].to_numpy()
    sex = df["sex"].to_numpy()
    jailed = df["ever_jailed"].to_numpy(dtype=float)
    y = df[column].to_numpy(dtype=float)

    X = np.column_stack(
        (
            np.ones(n),
            (race == "black").astype(float),
            (race == "hipanic").astype(float),
            (sex == "male").astype(float),
            jailed,
        )
    )

    # Observations with the same regressors and outcome contribute identical
    # terms to the objective.  Collapse those duplicates and retain their
    # multiplicity as a weight; this is mathematically the same LP but much
    # smaller for this dataset.
    grouped, weights = np.unique(
        np.column_stack((X[:, 1:], y)), axis=0, return_counts=True
    )
    X = np.column_stack((np.ones(len(grouped)), grouped[:, :4]))
    y = grouped[:, 4]
    weights = weights.astype(float)
    m = len(y)

    # X beta + u_plus - u_minus = y
    rows = np.arange(m)
    A = coo_matrix(
        (
            np.concatenate((X.ravel(), np.ones(m), -np.ones(m))),
            (
                np.concatenate((np.repeat(rows, 5), rows, rows)),
                np.concatenate(
                    (
                        np.tile(np.arange(5), m),
                        5 + rows,
                        5 + m + rows,
                    )
                ),
            ),
        ),
        shape=(m, 5 + 2 * m),
    ).tocsr()
    objective = np.concatenate(
        (np.zeros(5), q * weights, (1 - q) * weights)
    )
    bounds = [(None, None)] * 5 + [(0, None)] * (2 * m)

    result = linprog(
        objective,
        A_eq=A,
        b_eq=y,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile-regression LP failed: {result.message}")

    values = result.x[:5]
    return {
        "coef": dict(
            zip(
                ("intercept", "black", "hispanic", "male", "ever_jailed"),
                values.tolist(),
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
