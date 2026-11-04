import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import hstack, eye, csr_matrix


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

    # The model matrix has the requested reference groups (white and female).
    # Keep the row order unchanged so every observation contributes one loss.
    race = df["race"].astype(str)
    sex = df["sex"].astype(str)
    x = np.column_stack(
        (
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            np.asarray(df["ever_jailed"], dtype=float),
        )
    )
    y = np.asarray(df[outcome], dtype=float)
    n = len(y)

    # For u_i = y_i - x_i'b, write u_i = u_i^+ - u_i^-.
    # Minimizing q*u_i^+ + (1-q)*u_i^- is precisely the check loss.
    # Scaling only improves LP conditioning; coefficients are rescaled below.
    scale = max(1.0, float(np.max(np.abs(y))))
    ys = y / scale
    constraints = hstack((csr_matrix(x), eye(n), -eye(n)), format="csr")
    result = linprog(
        c=np.r_[np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)],
        A_eq=constraints,
        b_eq=ys,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[:5] * scale
    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, beta.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
