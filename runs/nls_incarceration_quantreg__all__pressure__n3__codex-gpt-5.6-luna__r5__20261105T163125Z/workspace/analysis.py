import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    if outcome not in df.columns:
        raise ValueError(f"unknown wealth year: {year}")

    # The design uses White and female as the omitted categories.  The source
    # data spells Hispanic as ``hipanic``.
    x = np.column_stack(
        (
            np.ones(len(df)),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n, p = x.shape

    # There are only a few distinct rows of X.  For a given covariate pattern
    # g, the contribution sum_i rho_q(y_i - x_g'b) is a convex piecewise-linear
    # function of x_g'b.  Its epigraph can therefore be represented by one
    # unrestricted variable t_g and one linear inequality for each segment.
    # This is exactly the same check-loss LP, but avoids two variables per row.
    groups = {}
    for row, value in zip(x, y):
        key = tuple(row.tolist())
        groups.setdefault(key, []).append(value)

    # Each group has one t variable; its position is p + group index.
    n_groups = len(groups)
    inequalities = []
    rhs = []
    for group_index, (key, values) in enumerate(groups.items()):
        values = np.sort(np.asarray(values, dtype=float))
        unique, counts = np.unique(values, return_counts=True)
        m = len(values)
        cumulative = 0
        for segment in range(len(unique) + 1):
            slope = -q * m + cumulative
            probe = (unique[0] - 1.0 if segment == 0 else
                     unique[-1] + 1.0 if segment == len(unique) else
                     unique[segment - 1])
            residual = values - probe
            loss = np.sum(residual * (q - (residual < 0)))
            row = np.zeros(p + n_groups)
            row[:p] = slope * np.asarray(key)
            row[p + group_index] = -1.0
            inequalities.append(row)
            rhs.append(-(loss - slope * probe))
            if segment < len(unique):
                cumulative += counts[segment]

    objective = np.r_[np.zeros(p), np.ones(n_groups)]
    bounds = [(None, None)] * (p + n_groups)
    result = linprog(
        objective,
        A_ub=np.asarray(inequalities),
        b_ub=np.asarray(rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:p]
    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
