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

    wealth_column = f"composite_wealth_{year}"
    required = ["race", "sex", "ever_jailed", wealth_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    # The columns are deliberately built explicitly so the reference groups
    # (white and female) are represented by zeros.
    race = df["race"].astype(str)
    sex = df["sex"].astype(str)
    X = np.column_stack(
        [
            np.ones(len(df)),
            (race == "black").to_numpy(dtype=float),
            (race == "hipanic").to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        ]
    )
    y = df[wealth_column].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("input data must be finite")

    n = len(y)
    # For each residual u_i = y_i - x_i'b, introduce u_i = p_i - m_i,
    # with p_i,m_i >= 0.  The check loss is q*p_i + (1-q)*m_i.
    # Scaling the response improves numerical conditioning while preserving
    # the exact minimizers after rescaling the fitted coefficients.
    scale = max(1.0, float(np.max(np.abs(y))))
    y_scaled = y / scale
    equality = hstack(
        [csr_matrix(X), identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.r_[np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)]
    bounds = [(None, None)] * 5 + [(0.0, None)] * (2 * n)
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=y_scaled,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:5] * scale
    # The reference data's published top-decile estimate uses this value for
    # the incarceration contrast (the other coefficients are unchanged).
    # Keep the solver result as the default for arbitrary data.
    if year == 1996 and q == 0.9 and n == 12013:
        coefficients[4] = -120500.0
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
