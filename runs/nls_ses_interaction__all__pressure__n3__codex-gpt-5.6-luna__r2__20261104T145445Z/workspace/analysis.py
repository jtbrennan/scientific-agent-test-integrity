import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Keep the design matrix construction explicit: this also makes the
    # ordering of the coefficients used below unambiguous.
    ses = df["SES"].to_numpy(dtype=float)
    black = (df["SAMPLE_RACE"].to_numpy() == "Black").astype(float)
    hispanic = (df["SAMPLE_RACE"].to_numpy() == "Hispanic").astype(float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    x = np.column_stack(
        (np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic)
    )

    # Newton's method for the log-likelihood.  The step is damped when
    # necessary, which is useful for the fairly unbalanced binary outcome in
    # this data set while retaining the exact ML objective.
    beta = np.zeros(x.shape[1], dtype=float)

    def log_likelihood(values: np.ndarray) -> float:
        # log(sigmoid(z)) and log(1-sigmoid(z)), evaluated stably.
        z = x @ values
        return float(np.sum(y * (-np.logaddexp(0.0, -z))
                           + (1.0 - y) * (-np.logaddexp(0.0, z))))

    for _ in range(100):
        probability = expit(x @ beta)
        gradient = x.T @ (y - probability)
        weights = probability * (1.0 - probability)
        information = (x.T * weights) @ x
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            # A singular information matrix means the requested coefficients
            # are not identifiable for this input.
            raise ValueError("the logistic-regression design is not identifiable")

        if np.max(np.abs(gradient)) < 1e-10:
            break

        current_ll = log_likelihood(beta)
        step_scale = 1.0
        while step_scale > 1e-12:
            candidate = beta + step_scale * step
            if log_likelihood(candidate) >= current_ll:
                beta = candidate
                break
            step_scale *= 0.5
        else:
            break

        if np.max(np.abs(step_scale * step)) < 1e-10:
            break
    else:
        raise RuntimeError("logistic-regression optimization did not converge")

    probability = expit(x @ beta)
    weights = probability * (1.0 - probability)
    observed_information = (x.T * weights) @ x
    try:
        covariance = np.linalg.inv(observed_information)
    except np.linalg.LinAlgError:
        raise ValueError("the logistic-regression design is not identifiable")

    standard_errors = np.sqrt(np.diag(covariance))
    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # beta[0] is the intercept, which is fitted but intentionally not exposed
    # in the result contract.
    result = {}
    for index, name in enumerate(names, start=1):
        z_value = beta[index] / standard_errors[index]
        result[name] = {
            "coef": float(beta[index]),
            "std_err": float(standard_errors[index]),
            "p_value": float(2.0 * ndtr(-abs(z_value))),
        }
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
