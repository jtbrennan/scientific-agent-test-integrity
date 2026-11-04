import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import norm


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES, race and SES x race.

    Returns a dict with keys ses, black, hispanic, ses_x_black, ses_x_hispanic
    (each a dict with coef, std_err, p_value) and n.
    """
    # Prepare outcome (y) and design matrix (X)
    n = int(df.shape[0])

    # Binary outcome as 0/1 integers
    y = df["BA DEGREE COMPLETED"].astype(int).to_numpy()

    # Race dummies with White as the reference group
    race = df["SAMPLE_RACE"].astype(str)
    black = (race == "Black").astype(int).to_numpy()
    hispanic = (race == "Hispanic").astype(int).to_numpy()

    # SES (continuous)
    ses = df["SES"].to_numpy(dtype=float)

    # Interaction terms
    ses_x_black = ses * black
    ses_x_hispanic = ses * hispanic

    # Design matrix with intercept first
    X = np.column_stack(
        [
            np.ones(n),  # intercept
            ses,
            black,
            hispanic,
            ses_x_black,
            ses_x_hispanic,
        ]
    )

    # IRLS / Newton-Raphson for logistic regression
    beta = np.zeros(X.shape[1])
    tol = 1e-10
    max_iter = 200
    eps = 1e-12

    for _ in range(max_iter):
        eta = X @ beta
        # numerically stable sigmoid
        # clip eta to avoid extreme exp overflows
        p = 1.0 / (1.0 + np.exp(-eta))
        # avoid 0/1 exactly
        p = np.clip(p, eps, 1 - eps)

        # Gradient and Hessian (observed)
        W = p * (1.0 - p)
        # X^T W X
        XTW = X.T * W
        H = XTW @ X  # positive definite for convex IRLS normal equations
        g = X.T @ (y - p)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse if ill-conditioned
            step = np.linalg.pinv(H) @ g

        beta_new = beta + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        # If not converged, still proceed with the last iterate
        pass

    # Variance-covariance from inverse observed information at the MLE
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, eps, 1 - eps)
    W = p * (1.0 - p)
    XTW = X.T * W
    info = XTW @ X
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Indices for coefficients
    # 0: intercept, 1: ses, 2: black, 3: hispanic, 4: ses_x_black, 5: ses_x_hispanic
    idx = {
        "ses": 1,
        "black": 2,
        "hispanic": 3,
        "ses_x_black": 4,
        "ses_x_hispanic": 5,
    }

    out = {"n": n}
    for name, j in idx.items():
        b = float(beta[j])
        s = float(se[j])
        # Wald test z statistic
        z = b / s if s > 0 else np.nan
        pval = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
        out[name] = {"coef": b, "std_err": s, "p_value": pval}

    return out


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_race_interaction(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
