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
    # Use an explicit two-column design matrix so that the first column is the
    # intercept.  ``read_csv`` parses the outcome in the supplied file as a
    # Boolean; converting it to float also keeps the calculation convenient
    # for DataFrames constructed directly in Python.
    x = df["SES"].to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    n = len(df)

    if n == 0:
        raise ValueError("cannot fit a model with no rows")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    design = np.column_stack((np.ones(n), x))

    # Newton--Raphson / iteratively reweighted least squares for the binomial
    # log-likelihood.  expit is numerically stable even for large linear
    # predictors.  A short step-halving loop keeps each accepted iterate from
    # reducing the likelihood.
    beta = np.zeros(2)

    def log_likelihood(coefficients: np.ndarray) -> float:
        eta = design @ coefficients
        # logaddexp avoids overflow in log(1 + exp(eta)).
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current_ll = log_likelihood(beta)
    for _ in range(100):
        probability = expit(design @ beta)
        weights = probability * (1.0 - probability)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (y - probability)

        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("model parameters are not identifiable") from exc

        scale = 1.0
        while scale > 2.0**-30:
            candidate = beta + scale * step
            candidate_ll = log_likelihood(candidate)
            if candidate_ll >= current_ll:
                break
            scale /= 2.0
        else:
            raise RuntimeError("logistic regression failed to improve")

        beta = candidate
        current_ll = candidate_ll
        if np.max(np.abs(scale * step)) < 1e-12:
            break
    else:
        raise RuntimeError("logistic regression did not converge")

    # The inverse observed Fisher information is the large-sample covariance
    # matrix of the MLE.  The Wald statistic is asymptotically standard normal
    # under a zero SES effect.
    probability = expit(design @ beta)
    information = design.T @ ((probability * (1.0 - probability))[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("model parameters are not identifiable") from exc

    coefficient = float(beta[1])
    std_err = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * norm.sf(abs(coefficient / std_err)))
    return {
        "coef": coefficient,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coefficient)),
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
