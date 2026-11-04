import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    x = df["SES"].to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    n = len(df)

    # Include an intercept explicitly.  logaddexp keeps the likelihood
    # well-behaved even when an intermediate linear predictor is large.
    design = np.column_stack((np.ones(n), x))

    def negative_log_likelihood(beta: np.ndarray) -> float:
        linear_predictor = design @ beta
        return float(
            np.sum(np.logaddexp(0.0, linear_predictor) - y * linear_predictor)
        )

    def gradient(beta: np.ndarray) -> np.ndarray:
        linear_predictor = design @ beta
        return design.T @ (expit(linear_predictor) - y)

    # Starting at the intercept-only estimate makes convergence quick while
    # still allowing the SES coefficient to be estimated freely.
    proportion = np.mean(y)
    intercept_start = np.log(proportion / (1.0 - proportion))
    result = minimize(
        negative_log_likelihood,
        np.array([intercept_start, 0.0]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    if not result.success and np.max(np.abs(gradient(result.x))) > 1e-7:
        raise RuntimeError(f"logistic regression failed to converge: {result.message}")

    beta = result.x
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(information)
    std_err = float(np.sqrt(covariance[1, 1]))
    coef = float(beta[1])
    z_score = coef / std_err

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
