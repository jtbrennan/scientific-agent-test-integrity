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
    if not 0 < q < 1:
        raise ValueError("q must be strictly between 0 and 1")

    target = f"composite_wealth_{year}"
    required = {"race", "sex", "ever_jailed", target}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # The columns are, in order, the intercept and the four requested
    # indicators.  ``hipanic`` is the spelling used by the supplied data.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            df["race"].eq("black").to_numpy(dtype=float),
            df["race"].eq("hipanic").to_numpy(dtype=float),
            df["sex"].eq("male").to_numpy(dtype=float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[target].to_numpy(dtype=float)
    n = len(y)

    # For residual r = y - Xb, write r = r_plus - r_minus, where both
    # components are nonnegative.  Then rho_q(r) is
    # q*r_plus + (1-q)*r_minus.  This LP therefore minimizes the check loss
    # globally, rather than using an iterative approximation.
    c = np.concatenate((np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)))
    equality = sparse.hstack(
        (sparse.csr_matrix(x), sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")),
        format="csr",
    )
    rhs = y
    result = linprog(
        c,
        A_eq=equality,
        b_eq=rhs,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[:5]
    return {
        "coef": dict(
            zip(("intercept", "black", "hispanic", "male", "ever_jailed"), beta.tolist())
        ),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
