import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Keep the model matrix in the same order as the public result.  Scaling
    # precipitation to metres is part of the model specification (and avoids
    # unnecessarily small coefficients).
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )

    # A log-linear least-squares estimate gives a useful finite starting point
    # even when some observed counts are zero.
    beta0 = np.linalg.lstsq(x, np.log(y + 0.5), rcond=None)[0]

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        # The optimizer stays in a finite region for this data.  Returning an
        # infinite objective for overflow keeps the likelihood well-defined.
        with np.errstate(over="ignore"):
            mu = np.exp(eta)
        if not np.all(np.isfinite(mu)):
            return np.inf
        return float(np.sum(mu - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        with np.errstate(over="ignore"):
            mu = np.exp(eta)
        if not np.all(np.isfinite(mu)):
            # This branch is only a safeguard for an exploratory optimizer
            # step; the optimum itself is finite for a non-degenerate dataset.
            return np.full(x.shape[1], np.nan)
        return x.T @ (mu - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        with np.errstate(over="ignore"):
            mu = np.exp(eta)
        return x.T @ (mu[:, None] * x)

    # Damped Newton maximization.  For a Poisson GLM the negative
    # log-likelihood Hessian is X' diag(mu) X, which is also the expected
    # Fisher information.  Backtracking makes the iterations safe when the
    # log-count starting values are not close to the MLE.
    beta = beta0
    for _ in range(100):
        grad = gradient(beta)
        info = hessian(beta)
        try:
            step = -np.linalg.solve(info, grad)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Poisson regression has a singular information matrix") from exc

        old_objective = objective(beta)
        step_size = 1.0
        while step_size > 2.0**-50:
            candidate = beta + step_size * step
            if objective(candidate) <= old_objective + 1e-4 * step_size * grad @ step:
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Poisson regression failed to find a likelihood-improving step")

        beta = candidate
        if np.max(np.abs(gradient(beta))) < 1e-8:
            break
    else:
        raise RuntimeError("Poisson regression failed to converge")

    mu = np.exp(x @ beta)
    fisher = x.T @ (mu[:, None] * x)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    # Two-sided normal-reference Wald p-values, written using ndtr to avoid
    # requiring any additional statistical package.
    z = np.divide(beta, std_err)
    p_values = 2.0 * ndtr(-np.abs(z))

    ratio_term = np.zeros_like(y)
    positive = y > 0
    ratio_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    deviance = 2.0 * np.sum(ratio_term - (y - mu))

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(beta[i]),
            "std_err": float(std_err[i]),
            "p_value": float(p_values[i]),
        }
        for i, name in enumerate(names)
    }
    return {"terms": terms, "deviance": float(deviance), "n": int(len(df))}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
