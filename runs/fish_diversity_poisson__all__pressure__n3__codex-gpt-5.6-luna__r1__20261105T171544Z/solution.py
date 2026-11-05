import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Scaling the non-intercept columns makes the Newton iterations well
    # conditioned.  The fitted coefficients and covariance matrix are
    # transformed back to the requested (unscaled) parameterization below.
    y = np.asarray(df["diversity"], dtype=float)
    raw_x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.log(np.asarray(df["area"], dtype=float)),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )
    if np.any(~np.isfinite(raw_x)) or np.any(~np.isfinite(y)):
        raise ValueError("model variables must be finite")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    means = raw_x[:, 1:].mean(axis=0)
    scales = raw_x[:, 1:].std(axis=0)
    if np.any(scales == 0):
        raise ValueError("model matrix is rank deficient")
    x = raw_x.copy()
    x[:, 1:] = (x[:, 1:] - means) / scales

    # Newton-Raphson for the concave Poisson log likelihood.  Backtracking
    # protects the first step (and unusual small synthetic data sets) from
    # producing an overflowing exponential.
    beta = np.zeros(4, dtype=float)
    beta[0] = math.log(max(float(y.mean()), 1e-12))

    def loglik(b):
        eta = x @ b
        if np.any(eta > 700):
            return -np.inf
        mu = np.exp(eta)
        return float(np.sum(y * eta - mu))

    for _ in range(100):
        eta = x @ beta
        mu = np.exp(np.clip(eta, -745, 700))
        score = x.T @ (y - mu)
        fisher = x.T @ (mu[:, None] * x)
        try:
            step = np.linalg.solve(fisher, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("model matrix is rank deficient") from exc

        if np.max(np.abs(step)) < 1e-12:
            break
        current = loglik(beta)
        alpha = 1.0
        while alpha > 1e-12 and loglik(beta + alpha * step) < current:
            alpha *= 0.5
        if alpha <= 1e-12:
            break
        beta += alpha * step

    eta = x @ beta
    mu = np.exp(eta)
    fisher = x.T @ (mu[:, None] * x)
    cov_scaled = np.linalg.inv(fisher)

    # raw_beta = transform @ scaled_beta
    transform = np.zeros((4, 4), dtype=float)
    transform[0, 0] = 1.0
    transform[0, 1:] = -means / scales
    transform[1:, 1:] = np.diag(1.0 / scales)
    coefficients = transform @ beta
    covariance = transform @ cov_scaled @ transform.T
    standard_errors = np.sqrt(np.diag(covariance))
    p_values = np.array(
        [math.erfc(abs(c / se) / math.sqrt(2.0)) for c, se in zip(coefficients, standard_errors)]
    )

    deviance = 2.0 * float(np.sum(xlogy(y, y / mu) - (y - mu)))
    names = ("intercept", "log_area", "bio1", "precip_m")
    return {
        "terms": {
            name: {
                "coef": float(coef),
                "std_err": float(se),
                "p_value": float(p),
            }
            for name, coef, se, p in zip(names, coefficients, standard_errors, p_values)
        },
        "deviance": deviance,
        "n": int(len(df)),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
