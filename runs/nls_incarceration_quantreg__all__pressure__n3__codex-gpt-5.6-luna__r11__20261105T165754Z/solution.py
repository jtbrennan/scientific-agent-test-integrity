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
    if not (0 < q < 1):
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    columns = [outcome, "race", "sex", "ever_jailed"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")

    # The model matrix uses white/female as the omitted categories.  Keep the
    # dataframe order intact: n is explicitly the number of observations used.
    y = df[outcome].to_numpy(dtype=float)
    race = df["race"].astype(str).to_numpy()
    sex = df["sex"].astype(str).to_numpy()
    jailed = df["ever_jailed"].to_numpy(dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(jailed).all()):
        raise ValueError("input contains non-finite values")

    X = np.column_stack(
        [
            np.ones(len(df)),
            (race == "black").astype(float),
            (race == "hipanic").astype(float),
            (sex == "male").astype(float),
            jailed,
        ]
    )

    # Within a design cell, the fitted value is a single scalar.  The sum of
    # check losses for that cell is a convex piecewise-linear function of the
    # scalar, so it can be represented by its supporting lines at the observed
    # outcomes.  This is the same LP as the usual positive/negative residual
    # formulation, but has only one auxiliary variable per design cell.
    n, p = X.shape
    cell_rows = {}
    for row, value in zip(X, y):
        cell_rows.setdefault(tuple(row), []).append(value)
    cells = list(cell_rows)
    inequalities = []
    for cell_index, cell in enumerate(cells):
        values = np.sort(np.asarray(cell_rows[cell], dtype=float))
        for breakpoint in values:
            residual = values - breakpoint
            cell_loss = np.where(residual >= 0, q * residual, (q - 1.0) * residual).sum()
            # Right-hand derivative of the cell loss at the breakpoint.
            slope = ((values <= breakpoint).sum() * (1.0 - q)
                     - (values > breakpoint).sum() * q)
            row = np.zeros(p + len(cells))
            row[:p] = slope * np.asarray(cell)
            row[p + cell_index] = -1.0
            # slope * (X beta) - z <= -intercept
            inequalities.append((row, -(cell_loss - slope * breakpoint)))

    objective = np.r_[np.zeros(p), np.ones(len(cells))]
    A_ub = np.asarray([row for row, _ in inequalities])
    b_ub = np.asarray([bound for _, bound in inequalities])
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * (p + len(cells)),
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
