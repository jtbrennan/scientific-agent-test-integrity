import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keeping the intercept in the design matrix, rather than adding race
    # dummies for every group, makes White the reference category.
    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"].to_numpy()
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    X = np.column_stack(
        (
            np.ones(len(df)),
            ses,
            black,
            hispanic,
            ses * black,
            ses * hispanic,
        )
    )

    # logaddexp evaluates log(1 + exp(eta)) without overflowing.  The
    # derivatives are supplied so that the numerical optimizer is fitting the
    # actual binomial likelihood, not a least-squares approximation.
    def negative_log_likelihood(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return X.T @ (expit(X @ beta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        p = expit(X @ beta)
        return X.T @ (X * (p * (1.0 - p))[:, None])

    fit = minimize(
        negative_log_likelihood,
        x0=np.zeros(X.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    # BFGS can report precision loss at the numerical optimum even when its
    # score is effectively zero.  The score supplies a more useful convergence
    # check in that case.
    if not fit.success and np.linalg.norm(gradient(fit.x), ord=np.inf) > 1e-6:
        raise RuntimeError(f"Logistic-regression optimization failed: {fit.message}")

    beta = fit.x
    # The inverse of the observed information gives the usual model-based
    # covariance matrix for maximum-likelihood logistic-regression estimates.
    covariance = np.linalg.inv(hessian(beta))
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ("ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic")
    result = {
        name: {
            "coef": float(beta[i]),
            "std_err": float(std_err[i]),
            "p_value": float(p_values[i]),
        }
        for i, name in enumerate(names, start=1)
    }
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
