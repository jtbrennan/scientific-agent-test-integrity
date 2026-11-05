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
    # Keep the design matrix deliberately small and explicit: the requested
    # model contains only an intercept and SES.  In particular, do not let a
    # dataframe index or the other columns enter the fit.
    x = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if n == 0:
        raise ValueError("cannot fit logistic regression to an empty dataframe")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or boolean")

    design = np.column_stack((np.ones(n), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) computed stably for large positive eta.
        log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta))
        probability = expit(eta)
        gradient = design.T @ (y - probability)
        return float(-log_likelihood), -gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        x0=np.zeros(2, dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # BFGS can report precision loss when the final finite-dimensional
    # gradient is already at floating-point noise level.  The gradient is the
    # relevant convergence diagnostic for this likelihood.
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-7:
        raise RuntimeError(f"logistic regression failed to converge: {result.message}")

    beta = result.x
    probability = expit(design @ beta)
    information = design.T @ ((probability * (1.0 - probability))[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the fitted model has singular observed information") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
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
