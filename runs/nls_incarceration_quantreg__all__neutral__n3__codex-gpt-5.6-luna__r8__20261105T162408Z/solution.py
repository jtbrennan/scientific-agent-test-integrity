import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    required = {"race", "sex", "ever_jailed", outcome}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # White and female are the reference categories.  Keep the spelling of
    # ``hipanic`` here because that is the spelling used by the supplied data.
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)

    # There are only a few distinct rows in this design.  For one such row g,
    # the contribution to the objective is L_g(t), where t = x_g @ beta and
    # L_g is a convex piecewise-linear function of the fitted value t.  The
    # lines making up its epigraph can therefore replace n residual variables
    # with one variable per distinct design row.
    groups, inverse = np.unique(x, axis=0, return_inverse=True)
    n_groups = len(groups)
    rows = []
    rhs = []
    for group_index, group_x in enumerate(groups):
        values = np.sort(y[inverse == group_index])
        unique_values = np.unique(values)
        count = len(values)

        # A line is valid on each interval beginning at a distinct response
        # value.  Include the interval below the minimum separately.
        breakpoints = np.r_[values[0], unique_values]
        for k, point in enumerate(breakpoints):
            if k == 0:
                slope = -q * count
            else:
                slope = (1 - q) * np.count_nonzero(values <= point) - q * np.count_nonzero(values > point)
            loss_at_point = np.sum(np.where(values - point >= 0, q * (values - point), (1 - q) * (point - values)))
            intercept = loss_at_point - slope * point

            row = np.zeros(5 + n_groups)
            row[:5] = slope * group_x
            row[5 + group_index] = -1
            rows.append(row)
            rhs.append(-intercept)

    constraints = csr_matrix(np.asarray(rows))
    objective = np.r_[np.zeros(x.shape[1]), np.ones(n_groups)]
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.asarray(rhs),
        bounds=[(None, None)] * x.shape[1] + [(None, None)] * n_groups,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[: x.shape[1]]
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
