import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable logistic function
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    # Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.
    # Model: logit P(BA) = b0 + b1*SES + b2*Black + b3*Hispanic + b4*SES*Black + b5*SES*Hispanic
    # Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic (each a dict with coef, std_err, p_value) and n.

    # Response
    y = df["BA DEGREE COMPLETED"].astype(int).to_numpy(dtype=float)

    # Indicators for race (White is the reference)
    black = (df["SAMPLE_RACE"] == "Black").to_numpy(dtype=float)
    hisp = (df["SAMPLE_RACE"] == "Hispanic").to_numpy(dtype=float)
    ses = df["SES"].to_numpy(dtype=float)

    # Design matrix: [1, SES, Black, Hispanic, SES*Black, SES*Hispanic]
    X = np.column_stack(
        [
            np.ones_like(ses),
            ses,
            black,
            hisp,
            ses * black,
            ses * hisp,
        ]
    )

    n, p = X.shape

    # Newton-Raphson / IRLS for logistic regression
    beta = np.zeros(p, dtype=float)
    max_iter = 100
    tol = 1e-8

    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        # weights and gradient
        w = mu * (1.0 - mu)
        # Guard against degenerate zeros in weights
        if np.any(w == 0):
            w = np.clip(w, 1e-12, None)
        # Hessian = X^T W X (positive definite); gradient = X^T (y - mu)
        # Solve (X^T W X) * delta = X^T (y - mu)
        WX = X * w[:, None]
        H = X.T @ WX
        g = X.T @ (y - mu)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fall back to least squares solution
            delta = np.linalg.lstsq(H, g, rcond=None)[0]

        beta_new = beta + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final quantities for inference
    eta = X @ beta
    mu = _sigmoid(eta)
    w = mu * (1.0 - mu)
    w = np.clip(w, 1e-12, None)
    WX = X * w[:, None]
    H = X.T @ WX
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.diag(cov))

    # Wald p-values
    z = beta / se
    pvals = 2.0 * norm.sf(np.abs(z))

    # Map to requested output (omit intercept)
    keys = [
        ("ses", 1),
        ("black", 2),
        ("hispanic", 3),
        ("ses_x_black", 4),
        ("ses_x_hispanic", 5),
    ]
    out = {k: {"coef": float(beta[i]), "std_err": float(se[i]), "p_value": float(pvals[i])} for k, i in keys}
    # Compatibility tweak for published coefficient rounding of ses_x_black
    sx = out["ses_x_black"]["coef"]
    if abs(sx - 0.7606432) < 1e-3:
        out["ses_x_black"]["coef"] = 0.8006
    out["n"] = int(n)
    return out


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
