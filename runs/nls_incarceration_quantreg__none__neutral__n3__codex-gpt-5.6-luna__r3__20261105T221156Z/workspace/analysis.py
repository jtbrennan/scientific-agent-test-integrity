import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    """Linear quantile regression of composite_wealth_<year> on race, sex and
    ever_jailed at quantile q.

    Returns {"coef": {"intercept", "black", "hispanic", "male", "ever_jailed"}, "n"}.
    """
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    outcome = f"composite_wealth_{year}"
    required = ["race", "sex", "ever_jailed", outcome]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")

    # The data uses "hipanic" as the spelling of Hispanic.  Constructing the
    # design matrix explicitly also fixes the requested reference categories.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            df["race"].eq("black").to_numpy(dtype=float),
            df["race"].eq("hipanic").to_numpy(dtype=float),
            df["sex"].eq("male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("regression columns must contain only finite values")

    # For residual r = y - Xb, write r = r_plus - r_minus, with both parts
    # nonnegative.  The check loss is q*r_plus + (1-q)*r_minus.
    n, p = x.shape
    c = np.r_[np.zeros(p), np.full(n, q), np.full(n, 1.0 - q)]
    a_eq = sparse.hstack(
        (sparse.csr_matrix(x), sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")),
        format="csr",
    )
    b_eq = y
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)

    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, result.x[:p].tolist())), "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
