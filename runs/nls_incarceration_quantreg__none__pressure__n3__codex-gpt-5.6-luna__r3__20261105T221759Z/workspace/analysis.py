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

    outcome = f"composite_wealth_{year}"
    required = ["race", "sex", "ever_jailed", outcome]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"missing required column(s): {', '.join(missing)}")

    # The first column is the intercept.  White and female are the omitted
    # categories, as specified in the model.
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            (df["race"].to_numpy() == "black").astype(float),
            (df["race"].to_numpy() == "hipanic").astype(float),
            (df["sex"].to_numpy() == "male").astype(float),
            df["ever_jailed"].to_numpy(dtype=float),
        )
    )
    y = df[outcome].to_numpy(dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("the regression data must contain only finite values")

    n, p = x.shape

    # For residual r = y - X beta, write r = r_plus - r_minus.  Since
    # rho_q(r) = q*r_plus + (1-q)*r_minus, this is a linear program.  Split
    # beta into beta_plus - beta_minus to allow unrestricted coefficients.
    # The sparse formulation keeps the problem modest even for large data.
    a_eq = hstack(
        (
            csr_matrix(x),
            -csr_matrix(x),
            identity(n, format="csr"),
            -identity(n, format="csr"),
        ),
        format="csr",
    )
    objective = np.concatenate(
        (np.zeros(2 * p), np.full(n, q), np.full(n, 1.0 - q))
    )
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=y,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    beta = result.x[:p] - result.x[p : 2 * p]
    names = ("intercept", "black", "hispanic", "male", "ever_jailed")
    return {"coef": dict(zip(names, beta.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
