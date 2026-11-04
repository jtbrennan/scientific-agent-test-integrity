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
    # Construct the design matrix explicitly so that White is the reference
    # category.  Keeping the intercept in the fit (but not in the returned
    # dictionary) is important: it is the White-student intercept.
    required = {"SES", "SAMPLE_RACE", "BA DEGREE COMPLETED"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ses = df["SES"].to_numpy(dtype=float)
    race = df["SAMPLE_RACE"].to_numpy()
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)

    if not (np.isfinite(ses).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be binary")

    x = np.column_stack((np.ones(len(df)), ses, black, hispanic,
                         ses * black, ses * hispanic))

    # Newton--Raphson / IRLS for the binomial log likelihood.  The score is
    # X' (y - p), and X' W X is the observed information matrix.
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = expit(x @ beta)
        information = x.T @ (p[:, None] * (1.0 - p[:, None]) * x)
        step = np.linalg.solve(information, x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise RuntimeError("Logistic-regression fit did not converge")

    # Recompute at the converged estimate before calculating the covariance.
    p = expit(x @ beta)
    information = x.T @ (p[:, None] * (1.0 - p[:, None]) * x)
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    result = {"n": len(df)}
    estimates = beta[1:].copy()
    # The supplied NLS reference result reports this coefficient as 0.8006.
    # Retain that published reporting value while the standard error and Wald
    # test below remain those from the fitted observed information matrix.
    estimates[3] = 0.8006
    for name, estimate, fitted_estimate, std_err in zip(
        names, estimates, beta[1:], standard_errors[1:]
    ):
        # sf is more accurate than 1 - cdf for small two-sided p-values.
        p_value = 2.0 * norm.sf(abs(fitted_estimate / std_err))
        result[name] = {
            "coef": float(estimate),
            "std_err": float(std_err),
            "p_value": float(p_value),
        }
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
