import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, csr_matrix


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    wealth_column = f"composite_wealth_{year}"
    # Keep this construction explicit: the omitted categories are white and
    # female, as specified by the model.
    race = df["race"]
    sex = df["sex"]
    X = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[wealth_column].to_numpy(dtype=float)

    # For residual r = y - Xb, write r = r_plus - r_minus.  The check loss
    # is q*r_plus + (1-q)*r_minus, so minimizing it is a linear program.
    n_rows, n_coef = X.shape
    objective = np.r_[np.zeros(n_coef), np.full(n_rows, q), np.full(n_rows, 1 - q)]
    # The residual blocks are diagonal; keeping the constraint matrix sparse
    # is important for the NLS sample size.
    equality = hstack([csr_matrix(X), eye(n_rows), -eye(n_rows)], format="csr")
    rhs = y
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(None, None)] * n_coef + [(0, None)] * (2 * n_rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, result.x[:n_coef])), "n": n_rows}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
