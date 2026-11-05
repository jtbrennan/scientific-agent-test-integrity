import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, hstack, identity


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    if outcome not in df.columns:
        raise ValueError(f"missing outcome column: {outcome}")

    # The first five variables are the regression coefficients.  For each
    # observation, introduce nonnegative r_plus and r_minus such that
    # y - X b = r_plus - r_minus.  The check loss is then
    # q*r_plus + (1-q)*r_minus, so its minimizer is exactly the quantile
    # regression solution (up to ordinary floating-point LP precision).
    race = df["race"]
    sex = df["sex"]
    x = np.column_stack(
        (
            np.ones(len(df)),
            race.eq("black").to_numpy(dtype=float),
            # The source data spells this category "hipanic".  Accepting the
            # correctly spelled form as well makes the indicator less brittle.
            (race.eq("hipanic") | race.eq("hispanic")).to_numpy(dtype=float),
            sex.eq("male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    n = len(df)

    # Observations with identical X and y contribute identical terms to the
    # objective and constraint.  Collapse those rows and retain their counts
    # as objective weights; this is algebraically the same LP and is much
    # smaller for this dataset.
    unique_rows, counts = np.unique(
        np.column_stack((x, y)), axis=0, return_counts=True
    )
    x_lp = unique_rows[:, :5]
    y_lp = unique_rows[:, 5]
    weights = counts.astype(float)
    n_lp = len(y_lp)

    # Variables are [b (unrestricted), r_plus (>=0), r_minus (>=0)].
    a_eq = hstack(
        (
            csc_matrix(x_lp),
            identity(n_lp, format="csc"),
            -identity(n_lp, format="csc"),
        ),
        format="csc",
    )
    objective = np.concatenate(
        (np.zeros(5), q * weights, (1.0 - q) * weights)
    )
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=y_lp,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n_lp),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, result.x[:5])), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
