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
    # includes an intercept (and so that the order of the parameters is clear).
    try:
        x_ses = df["SES"].to_numpy(dtype=float)
        y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SES and BA DEGREE COMPLETED must be numeric columns") from exc

    n = len(df)
    if n == 0:
        raise ValueError("cannot fit a model with no rows")
    if not np.isfinite(x_ses).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only 0/1 (or False/True)")

    X = np.column_stack((np.ones(n), x_ses))

    # Newton-Raphson maximization of the log likelihood.  The likelihood and
    # its derivatives are written using stable functions, which matters for
    # large linear predictors during iteration.
    beta = np.zeros(2, dtype=float)

    def negative_log_likelihood(params: np.ndarray) -> float:
        eta = X @ params
        return float(np.logaddexp(0.0, eta).sum() - y @ eta)

    for _ in range(100):
        eta = X @ beta
        probabilities = expit(eta)
        gradient = X.T @ (probabilities - y)
        weights = probabilities * (1.0 - probabilities)
        hessian = (X.T * weights) @ X

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the logistic-regression parameters are not identifiable") from exc

        # The Newton step minimizes the negative log likelihood.  Backtracking
        # keeps the update well behaved if a trial step overshoots.
        current_loss = negative_log_likelihood(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta - step_scale * step
            if negative_log_likelihood(candidate) <= current_loss:
                break
            step_scale *= 0.5
        else:
            raise ValueError("failed to optimize the logistic-regression likelihood")

        beta_change = candidate - beta
        beta = candidate
        if np.max(np.abs(beta_change)) <= 1e-10 * (1.0 + np.max(np.abs(beta))):
            break
    else:
        raise ValueError("logistic-regression optimization did not converge")

    # The inverse observed information gives the usual MLE covariance matrix.
    probabilities = expit(X @ beta)
    information = (X.T * (probabilities * (1.0 - probabilities))) @ X
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the logistic-regression parameters are not identifiable") from exc

    coef = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    z = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z)))
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
