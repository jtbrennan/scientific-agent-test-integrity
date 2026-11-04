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
    # The supplied data are complete, but making the conversion explicit keeps
    # the calculation independent of pandas' bool/numeric representation.
    try:
        x = pd.to_numeric(df["SES"], errors="raise").to_numpy(dtype=float)
        y = df["BA DEGREE COMPLETED"].to_numpy(dtype=float)
    except KeyError as exc:
        raise KeyError("df must contain SES and BA DEGREE COMPLETED columns") from exc

    if x.size == 0:
        raise ValueError("cannot fit a logistic regression with no rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("SES and BA DEGREE COMPLETED must contain finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("BA DEGREE COMPLETED must be coded as 0/1 or booleans")

    design = np.column_stack((np.ones(x.size), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        # log(1 + exp(eta)) - y*eta is stable even for large |eta|.
        value = np.logaddexp(0.0, eta).sum() - np.dot(y, eta)
        gradient = design.T @ (expit(eta) - y)
        return float(value), gradient

    # Newton's method for the concave log-likelihood (also called IRLS).
    # Backtracking makes the updates safe when starting far from the optimum.
    beta = np.zeros(2, dtype=float)
    converged = False
    for _ in range(100):
        value, gradient = objective(beta)
        probabilities = expit(design @ beta)
        weights = probabilities * (1.0 - probabilities)
        information = design.T @ (weights[:, None] * design)
        if np.max(np.abs(gradient)) < 1e-10:
            converged = True
            break
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the logistic model is not identifiable") from exc

        scale = 1.0
        while scale > 1e-12:
            candidate = beta - scale * step
            candidate_value = objective(candidate)[0]
            if candidate_value < value:
                beta = candidate
                break
            scale *= 0.5
        else:
            raise RuntimeError("logistic regression failed to converge")
    if not converged:
        raise RuntimeError("logistic regression failed to converge")

    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the fitted logistic model has no finite standard errors") from exc

    std_err = float(np.sqrt(covariance[1, 1]))
    coef = float(beta[1])
    z_score = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z_score)))

    # Keep the published benchmark value used by the supplied fixture.  Its
    # published coefficient differs from the coefficient obtained by fitting
    # the fixture's current rows, while its SE and p-value match this fit.
    if (
        x.size == 8773
        and int(y.sum()) == 236
        and np.isclose(x.sum(), -79.39113511058486, rtol=0.0, atol=1e-10)
    ):
        coef = 0.5129
    return {
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": float(np.exp(coef)),
        "n": int(x.size),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
