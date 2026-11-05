import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


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
        raise KeyError(f"missing required column(s): {', '.join(missing)}")

    # The design matrix follows the parameter order requested by the API.
    # ``hipanic`` is the spelling used by the supplied data.
    race = df["race"].to_numpy()
    sex = df["sex"].to_numpy()
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            race == "black",
            (race == "hipanic") | (race == "hispanic"),
            sex == "male",
            df["ever_jailed"].to_numpy(),
        )
    ).astype(float)
    y = df[wealth_column].to_numpy(dtype=float)

    # Within a covariate pattern, the sum of check losses is a convex
    # piecewise-linear function of the fitted value.  Represent each such
    # function by its epigraph (one variable per pattern), rather than using
    # two positive residual variables for every observation.
    n, p = x.shape
    patterns, inverse = np.unique(x, axis=0, return_inverse=True)
    n_patterns = len(patterns)
    rows = []
    rhs = []
    for group in range(n_patterns):
        values = np.sort(y[inverse == group])
        unique_values = np.unique(values)
        count = len(values)
        slope = -q * count
        # Below the smallest response, f(t) = q * sum(y - t).
        intercept = q * float(values.sum())
        for value in unique_values:
            # z >= slope * (x'b) + intercept
            row = np.zeros(p + n_patterns)
            row[:p] = slope * patterns[group]
            row[p + group] = -1.0
            rows.append(row)
            rhs.append(-intercept)

            multiplicity = np.count_nonzero(values == value)
            slope += multiplicity
            # Evaluate f at the breakpoint to obtain the next line's
            # intercept, avoiding accumulation of floating-point error.
            loss_at_value = np.where(
                values >= value, q * (values - value), (1.0 - q) * (value - values)
            ).sum()
            intercept = float(loss_at_value - slope * value)

        # The first line was included immediately before each breakpoint; if
        # the group has observations, the final post-breakpoint line is needed
        # to describe the right tail as well.
        row = np.zeros(p + n_patterns)
        row[:p] = slope * patterns[group]
        row[p + group] = -1.0
        rows.append(row)
        rhs.append(-intercept)

    objective = np.r_[np.zeros(p), np.ones(n_patterns)]
    inequality = csr_matrix(np.asarray(rows))
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(rhs),
        bounds=[(None, None)] * (p + n_patterns),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile regression failed: {result.message}")

    coefficients = result.x[:p]
    names = ["intercept", "black", "hispanic", "male", "ever_jailed"]
    return {"coef": dict(zip(names, coefficients.tolist())), "n": n}


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_incarceration_processed.csv"
    df = pd.read_csv(data)
    print(json.dumps({str(q): fit_wealth_quantile(df, q) for q in (0.1, 0.5, 0.9)}, indent=2))


if __name__ == "__main__":
    main()
