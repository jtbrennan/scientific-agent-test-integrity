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
    # Keep the construction of the design matrix here (rather than relying on
    # a formula package), since the latter is not available in the execution
    # environment.  White is the omitted race category.
    required = {"SES", "SAMPLE_RACE", "BA DEGREE COMPLETED"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    race = df["SAMPLE_RACE"]
    black = (race == "Black").to_numpy(dtype=float)
    hispanic = (race == "Hispanic").to_numpy(dtype=float)

    outcome = df["BA DEGREE COMPLETED"]
    if outcome.isna().any() or not np.issubdtype(outcome.dtype, np.bool_):
        # Numeric 0/1 columns are also a natural representation for callers
        # constructing a DataFrame themselves.
        y = pd.to_numeric(outcome, errors="raise").to_numpy(dtype=float)
    else:
        y = outcome.to_numpy(dtype=float)

    if not (np.isfinite(ses).all() and np.isfinite(y).all()):
        raise ValueError("SES and BA DEGREE COMPLETED must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must contain only 0/1 values")

    X = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("the requested model is not identifiable for this data")

    # Newton-Raphson maximization of the concave Bernoulli log likelihood.
    # A backtracking step keeps the iterations stable for highly unbalanced
    # samples and for subsets with unusually large SES values.
    beta = np.zeros(X.shape[1], dtype=float)

    def log_likelihood(parameter):
        eta = X @ parameter
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    ll = log_likelihood(beta)
    for _ in range(100):
        probability = expit(X @ beta)
        gradient = X.T @ (y - probability)
        weights = probability * (1.0 - probability)
        information = (X.T * weights) @ X
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the requested model has no finite MLE") from exc

        if np.max(np.abs(step)) < 1e-10 and np.max(np.abs(gradient)) < 1e-8:
            break

        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            candidate_ll = log_likelihood(candidate)
            if candidate_ll >= ll:
                beta, ll = candidate, candidate_ll
                break
            step_scale *= 0.5
        else:
            raise ValueError("the requested model has no finite MLE")
    else:
        raise ValueError("logistic regression failed to converge")

    # Recompute the observed information at the converged estimate.  This is
    # -d^2 log L / d beta d beta', and its inverse is the MLE covariance.
    probability = expit(X @ beta)
    weights = probability * (1.0 - probability)
    information = (X.T * weights) @ X
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the requested model has no finite covariance") from exc
    std_err = np.sqrt(np.diag(covariance))
    p_values = 2.0 * norm.sf(np.abs(beta / std_err))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # The first coefficient is the intercept, which is fitted but not part of
    # the public result requested by the analysis.
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for index, name in enumerate(names, start=1)
    }
    result["n"] = int(len(df))
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
