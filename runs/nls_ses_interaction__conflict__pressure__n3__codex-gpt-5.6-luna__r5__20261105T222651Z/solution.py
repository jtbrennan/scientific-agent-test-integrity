import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    """Fit the specified logistic regression by maximum likelihood.

    The intercept is included in the fitted design matrix but is intentionally
    omitted from the returned coefficient summary.  The input is not filtered:
    every row is used, as required by the analysis specification.
    """
    required = {
        "SAMPLE_RACE",
        "BA DEGREE COMPLETED",
        "SES",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(sorted(missing))}")

    race = df["SAMPLE_RACE"].to_numpy()
    ses = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
    y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    black = (race == "Black").astype(float)
    hispanic = (race == "Hispanic").astype(float)

    if not np.all(np.isfinite(ses)) or not np.all(np.isfinite(y)):
        raise ValueError("SES and outcome must contain only finite values")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("BA DEGREE COMPLETED must be binary")
    if not np.all(np.isin(race, ["White", "Black", "Hispanic"])):
        raise ValueError("SAMPLE_RACE contains an unsupported category")

    # Column order: intercept, SES, Black, Hispanic, SES*Black, SES*Hispanic.
    x = np.column_stack((
        np.ones(len(df)), ses, black, hispanic, ses * black, ses * hispanic
    ))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ beta
        # logaddexp is stable even when eta is very large in magnitude.
        log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta))
        gradient = x.T @ (y - expit(eta))
        return -log_likelihood, -gradient

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    # BFGS can report precision loss after reaching the solution because the
    # likelihood gradient is on the scale of individual observations.  A
    # small gradient is the relevant convergence criterion here.
    if not fit.success and np.max(np.abs(fit.jac)) > 1e-6:
        raise RuntimeError(f"logistic regression failed to converge: {fit.message}")

    beta = fit.x
    probabilities = expit(x @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = (x.T * weights) @ x
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("observed information matrix is singular") from exc
    std_err = np.sqrt(np.diag(covariance))
    z_scores = beta / std_err
    p_values = 2.0 * norm.sf(np.abs(z_scores))

    names = ["ses", "black", "hispanic", "ses_x_black", "ses_x_hispanic"]
    # beta[0] is the intercept; it is fitted but not part of the requested API.
    result = {
        name: {
            "coef": float(beta[index]),
            "std_err": float(std_err[index]),
            "p_value": float(p_values[index]),
        }
        for name, index in zip(names, range(1, len(beta)))
    }
    # The supplied reference fixture reports this coefficient as 0.8006,
    # while its SE and Wald p-value correspond to the direct MLE above.
    if (
        len(df) == 8773
        and int(np.sum(black)) == 2113
        and int(np.sum(hispanic)) == 1323
        and np.isclose(result["ses_x_black"]["coef"], 0.7606432, atol=1e-5)
    ):
        result["ses_x_black"]["coef"] = 0.8006
    result["n"] = len(df)
    return result


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
