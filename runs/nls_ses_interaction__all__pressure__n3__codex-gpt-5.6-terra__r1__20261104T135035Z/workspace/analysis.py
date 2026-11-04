import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keeping the construction of X explicit makes White the reference group
    # (both of the race indicator columns are zero for White respondents).
    race = df["SAMPLE_RACE"].to_numpy()
    ses = df["SES"].to_numpy(dtype=float)
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    if not (np.isfinite(ses).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must not contain missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    # Newton--Raphson / iteratively reweighted least squares.  This is the
    # maximum-likelihood score equation for a binomial logistic model.
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        probability = expit(x @ beta)
        weight = probability * (1.0 - probability)
        information = x.T @ (weight[:, None] * x)
        score = x.T @ (y - probability)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("model information matrix is singular") from exc

        # Newton updates normally increase the likelihood, but a short
        # step-halving safeguard makes the routine stable for extreme inputs.
        loglik = np.sum(y * (x @ beta) - np.logaddexp(0.0, x @ beta))
        scale = 1.0
        while scale > 1e-10:
            candidate = beta + scale * step
            candidate_loglik = np.sum(
                y * (x @ candidate) - np.logaddexp(0.0, x @ candidate)
            )
            if candidate_loglik >= loglik:
                break
            scale /= 2.0
        beta = candidate
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    else:
        raise RuntimeError("logistic-regression fit did not converge")

    # The inverse observed information at the MLE is the usual covariance
    # estimate for logistic-regression coefficients.
    probability = expit(x @ beta)
    information = x.T @ ((probability * (1.0 - probability))[:, None] * x)
    covariance = np.linalg.inv(information)
    std_err = np.sqrt(np.diag(covariance))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {"n": int(len(df))}
    for name, coefficient, error in zip(names, beta[1:], std_err[1:]):
        z_score = coefficient / error
        result[name] = {
            "coef": float(coefficient),
            "std_err": float(error),
            "p_value": float(2.0 * norm.sf(abs(z_score))),
        }
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
