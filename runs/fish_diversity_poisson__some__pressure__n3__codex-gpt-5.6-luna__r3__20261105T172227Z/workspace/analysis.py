import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_diversity_poisson(df: pd.DataFrame) -> dict:
    """Poisson regression (log link, with intercept) of diversity on
    log(area), bio1 and bio12 / 1000.

    Returns a dict with keys:
      terms    -- {"intercept", "log_area", "bio1", "precip_m"}, each a dict
                  with keys coef, std_err, p_value
      deviance -- residual deviance
      n        -- number of rows used
    """
    # Build the design matrix explicitly so that the order of the returned
    # coefficients is fixed and agrees with the model in the question.
    area = np.asarray(df["area"], dtype=float)
    y = np.asarray(df["diversity"], dtype=float)
    x = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            np.log(area),
            np.asarray(df["bio1"], dtype=float),
            np.asarray(df["bio12"], dtype=float) / 1000.0,
        ]
    )

    if len(df) == 0:
        raise ValueError("at least one row is required")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("model columns must contain finite values")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("diversity must contain non-negative integer counts")

    # Newton-Raphson for the concave Poisson log likelihood.  The score is
    # X'(y-mu), and the expected negative Hessian (Fisher information) is
    # X' diag(mu) X.
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = np.log(max(float(y.mean()), np.finfo(float).tiny))

    def log_likelihood(parameter: np.ndarray) -> float:
        eta = x @ parameter
        # The line search below keeps eta in a finite range in ordinary
        # cases.  This guard also makes failed trial steps harmless.
        if np.any(eta > 700) or np.any(eta < -745):
            return -np.inf
        return float(np.dot(y, eta) - np.exp(eta).sum())

    likelihood = log_likelihood(beta)
    for _ in range(100):
        eta = x @ beta
        if np.any(eta > 700) or np.any(eta < -745):
            raise RuntimeError("Poisson fit diverged")
        mu = np.exp(eta)
        score = x.T @ (y - mu)
        information = x.T @ (mu[:, None] * x)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the design matrix does not have full rank") from exc

        # A backtracking step is useful for unusual starting values or
        # strongly correlated predictors while retaining Newton convergence.
        if np.linalg.norm(step, ord=np.inf) < 1e-11:
            break
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            candidate_likelihood = log_likelihood(candidate)
            if candidate_likelihood >= likelihood:
                beta = candidate
                likelihood = candidate_likelihood
                break
            step_scale *= 0.5
        else:
            raise RuntimeError("Poisson likelihood optimization failed")
        if np.linalg.norm(step_scale * step, ord=np.inf) < 1e-10:
            break
    else:
        raise RuntimeError("Poisson likelihood optimization did not converge")

    # Recompute at the final estimate; this is the model-based Fisher
    # information requested by the specification, with no dispersion factor.
    eta = x @ beta
    mu = np.exp(eta)
    information = x.T @ (mu[:, None] * x)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the design matrix does not have full rank") from exc
    standard_errors = np.sqrt(np.diag(covariance))
    z_scores = beta / standard_errors
    p_values = 2.0 * norm.sf(np.abs(z_scores))

    # For y=0, the convention in the question makes the first term zero.
    positive = y > 0
    deviance = 2.0 * float(
        np.sum(y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive]))
        + np.sum(mu[~positive])
    )

    names = ("intercept", "log_area", "bio1", "precip_m")
    terms = {
        name: {
            "coef": float(coef),
            "std_err": float(std_err),
            "p_value": float(p_value),
        }
        for name, coef, std_err, p_value in zip(names, beta, standard_errors, p_values)
    }
    return {"terms": terms, "deviance": deviance, "n": len(df)}


def main() -> None:
    data = Path(__file__).parent / "data" / "freshwater_fish_basins.csv"
    print(json.dumps(fit_diversity_poisson(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
