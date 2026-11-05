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
    # Construct the design matrix explicitly so that the intercept is part of
    # the model (and so that unrelated columns in ``df`` cannot affect it).
    x = np.asarray(df["SES"], dtype=float)
    y_values = df["BA DEGREE COMPLETED"]
    if pd.api.types.is_bool_dtype(y_values):
        y = np.asarray(y_values, dtype=float)
    else:
        # This also handles the 0/1 representation commonly used in test
        # data, while rejecting values that are not valid binary outcomes.
        y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("SES and BA DEGREE COMPLETED must be one-dimensional")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")
    n = int(y.size)
    if n == 0:
        raise ValueError("cannot fit a model with no rows")

    design = np.column_stack((np.ones(n), x))

    # Negative log likelihood and its analytic gradient.  logaddexp avoids
    # overflow when an optimizer evaluates a large trial coefficient.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        value = np.sum(np.logaddexp(0.0, eta) - y * eta)
        gradient = design.T @ (expit(eta) - y)
        return float(value), gradient

    # A good intercept start makes the fit quick even when the outcome is
    # unbalanced.  A zero slope is neutral and works for centered or raw SES.
    successes = float(y.sum())
    failures = n - successes
    if successes == 0.0 or failures == 0.0:
        raise ValueError("the outcome must contain both 0 and 1 values")
    start = np.array([np.log(successes / failures), 0.0])

    # Newton's method with a backtracking step is especially reliable for this
    # two-parameter concave likelihood.  The final Hessian is also the
    # observed Fisher information used for the standard errors.
    beta = start
    for _ in range(100):
        eta = design @ beta
        probability = expit(eta)
        score = design.T @ (y - probability)
        weights = probability * (1.0 - probability)
        information = (design.T * weights) @ design
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the logistic-regression information matrix is singular") from exc
        if np.max(np.abs(step)) < 1e-12:
            break

        current = -objective(beta)[0]
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            if -objective(candidate)[0] >= current:
                beta = candidate
                break
            step_scale *= 0.5
        else:
            raise RuntimeError("logistic-regression optimization did not converge")
    else:
        raise RuntimeError("logistic-regression optimization did not converge")

    eta = design @ beta
    probability = expit(eta)
    information = (design.T * (probability * (1.0 - probability))) @ design
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the logistic-regression information matrix is singular") from exc
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
