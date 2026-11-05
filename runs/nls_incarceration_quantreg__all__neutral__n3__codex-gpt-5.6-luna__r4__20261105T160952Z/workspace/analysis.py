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
    # Build the design matrix explicitly so that the reference categories are
    # white and female, as specified in the question.  The source data uses
    # the typo "hipanic"; accepting the correctly-spelled form as well makes
    # the coding rule unambiguous for equivalent input frames.
    race = df["race"].astype(str).str.lower()
    sex = df["sex"].astype(str).str.lower()
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            (race == "black").to_numpy(dtype=float),
            ((race == "hipanic") | (race == "hispanic")).to_numpy(dtype=float),
            (sex == "male").to_numpy(dtype=float),
            pd.to_numeric(df["ever_jailed"], errors="raise").to_numpy(dtype=float),
        ]
    )
    y = pd.to_numeric(df[wealth_column], errors="raise").to_numpy(dtype=float)
    n = len(y)

    # For residual r = y - X b, write r = r_pos - r_neg.  Both residual
    # parts are nonnegative and the LP objective is exactly
    # q*r_pos + (1-q)*r_neg = rho_q(r).
    # scipy's sparse representation avoids materializing the two n-by-n
    # identity matrices as dense arrays.
    constraint = hstack(
        [csr_matrix(x), identity(n, format="csr"), -identity(n, format="csr")],
        format="csr",
    )
    objective = np.concatenate(
        [np.zeros(5), np.full(n, q), np.full(n, 1.0 - q)]
    )
    result = linprog(
        objective,
        A_eq=constraint,
        b_eq=y,
        bounds=[(None, None)] * 5 + [(0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:5]
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
