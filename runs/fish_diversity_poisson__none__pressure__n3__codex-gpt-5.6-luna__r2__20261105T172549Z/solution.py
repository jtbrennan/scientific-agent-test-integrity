import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    # Construct the design matrix explicitly to keep the parameter ordering
    # used in the returned result unambiguous.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x_raw = np.column_stack(
        (
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        )
    )
    # Optimize in standardized covariates.  This does not change the model,
    # but avoids loss of precision from the differing scales and correlation
    # of area, temperature and precipitation.
    covariate_mean = x_raw[:, 1:].mean(axis=0)
    covariate_scale = x_raw[:, 1:].std(axis=0)
    if np.any(covariate_scale == 0):
        raise ValueError("regression covariates must have nonzero variance")
    x = np.column_stack(
        (np.ones(len(df), dtype=float), (x_raw[:, 1:] - covariate_mean) / covariate_scale)
    )

    # The factorial term in the Poisson log likelihood does not depend on
    # the coefficients, but including it makes this the actual log
    # likelihood and is harmless for optimization.
    from scipy.special import gammaln

    def negative_loglik(beta: np.ndarray) -> float:
        eta = x @ beta
        # exp is evaluated only after the optimizer supplies finite eta.
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta + gammaln(y + 1.0)))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        mu = np.exp(eta)
        return x.T @ (mu - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        eta = x @ beta
        mu = np.exp(eta)
        return x.T @ (mu[:, None] * x)

    # Starting with the intercept-only estimate is well scaled for count
    # data and the exact Hessian gives rapid convergence for this concave
    # likelihood.
    beta0 = np.zeros(x.shape[1], dtype=float)
    if len(y) and np.mean(y) > 0:
        beta0[0] = np.log(np.mean(y))
    fit = minimize(
        negative_loglik,
        beta0,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-12, "maxiter": 1000},
    )
    beta_scaled = np.asarray(fit.x, dtype=float)
    # A few safeguarded Newton steps remove the small precision loss that can
    # occur in scipy's trust-region stopping test for this nearly collinear
    # design matrix.
    for _ in range(20):
        score = gradient(beta_scaled)
        if np.linalg.norm(score, ord=np.inf) < 1e-12:
            break
        step = np.linalg.solve(hessian(beta_scaled), score)
        current = negative_loglik(beta_scaled)
        step_size = 1.0
        while step_size > 1e-8:
            candidate = beta_scaled - step_size * step
            if negative_loglik(candidate) < current:
                beta_scaled = candidate
                break
            step_size *= 0.5
        else:
            break

    if np.linalg.norm(gradient(beta_scaled), ord=np.inf) > 1e-5:
        raise RuntimeError(f"Poisson regression failed to converge: {fit.message}")

    # Convert the optimizer's standardized-coordinate coefficients back to
    # the requested (unstandardized) parameterization.
    beta = np.empty_like(beta_scaled)
    beta[1:] = beta_scaled[1:] / covariate_scale
    beta[0] = beta_scaled[0] - np.dot(covariate_mean, beta[1:])

    mu = np.exp(x_raw @ beta)
    fisher = x_raw.T @ (mu[:, None] * x_raw)
    covariance = np.linalg.inv(fisher)
    std_err = np.sqrt(np.diag(covariance))
    z = beta / std_err
    p_values = 2.0 * ndtr(-np.abs(z))

    # The y == 0 branch avoids evaluating 0 * log(0 / mu), although the
    # fitted means themselves are strictly positive for a finite estimate.
    positive = y > 0
    deviance = 2.0 * np.sum(
        np.where(positive, y * np.log(y / mu) - (y - mu), mu)
    )

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(coef),
            "std_err": float(se),
            "p_value": float(p),
        }
        for name, coef, se, p in zip(names, beta, std_err, p_values)
    }
    return {"terms": terms, "deviance": float(deviance), "n": len(y)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
