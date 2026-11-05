import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Construct the design matrix explicitly so that the fitted model always
    # contains an intercept, including when SES happens to be centred.
    x = np.asarray(df["SES"], dtype=float)
    y = np.asarray(df["BA DEGREE COMPLETED"], dtype=float)
    n = len(df)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or False/True")

    design = np.column_stack((np.ones(n), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) is evaluated stably for large positive eta.
        negative_log_likelihood = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = design.T @ (expit(eta) - y)
        return float(negative_log_likelihood), gradient

    def hessian(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(design @ beta)
        return design.T @ ((probabilities * (1.0 - probabilities))[:, None] * design)

    # Newton-Raphson with backtracking.  The objective is convex, and the
    # line search keeps early Newton steps well behaved even for imbalanced
    # outcomes.
    beta = np.zeros(2, dtype=float)
    converged = False
    for _ in range(100):
        value, gradient = objective(beta)
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian(beta), gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The logistic-regression information matrix is singular") from exc

        step_size = 1.0
        directional_derivative = float(gradient @ step)
        while step_size > 1e-12:
            candidate = beta - step_size * step
            candidate_value = objective(candidate)[0]
            if candidate_value <= value - 1e-4 * step_size * directional_derivative:
                beta = candidate
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Logistic regression failed to converge")
    if not converged:
        raise RuntimeError("Logistic regression failed to converge")

    probabilities = expit(design @ beta)
    information = hessian(beta)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The logistic-regression information matrix is singular") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * norm.sf(abs(coef / std_err)))

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
