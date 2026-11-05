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
    outcome = "BA DEGREE COMPLETED"
    predictor = "SES"

    if outcome not in df or predictor not in df:
        raise KeyError(f"DataFrame must contain {outcome!r} and {predictor!r}")

    y = np.asarray(df[outcome], dtype=float)
    ses = np.asarray(df[predictor], dtype=float)
    n = len(df)
    if not np.isfinite(y).all() or not np.isfinite(ses).all():
        raise ValueError("Outcome and SES must contain only finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary (0/1 or boolean)")
    if n == 0:
        raise ValueError("Cannot fit a model with no rows")

    X = np.column_stack((np.ones(n), ses))

    # Negative log-likelihood and its analytic derivatives.  logaddexp keeps
    # the likelihood stable even when the linear predictor is large.
    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = X @ beta
        return X.T @ (expit(eta) - y)

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(X @ beta)
        weights = probabilities * (1.0 - probabilities)
        return X.T @ (weights[:, None] * X)

    fit = minimize(
        objective,
        x0=np.zeros(2),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # SciPy can report precision loss after reaching a valid optimum when the
    # requested tolerance is near machine precision.  The score is the more
    # direct convergence check for this likelihood.
    if not fit.success and np.max(np.abs(gradient(fit.x))) > 1e-7:
        raise RuntimeError(f"Logistic regression failed to converge: {fit.message}")

    beta = fit.x
    information = hessian(beta)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("The fitted model has a singular information matrix") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z_score = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z_score)))

    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
