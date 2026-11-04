import json
from pathlib import Path

import pandas as pd
import numpy as np
from math import exp
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Extract variables

    y = df["BA DEGREE COMPLETED"].astype(int).to_numpy(dtype=float)
    x = df["SES"].to_numpy(dtype=float)

    n = y.shape[0]
    # Design matrix with intercept
    X = np.column_stack([np.ones(n, dtype=float), x])

    # Initialize coefficients at zeros
    beta = np.zeros(2, dtype=float)

    # IRLS / Newton-Raphson for logistic regression
    def sigmoid(z):
        # Stable sigmoid
        # For large negative z, exp(-|z|) avoids overflow
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    tol = 1e-10
    max_iter = 100
    for _ in range(max_iter):
        eta = X @ beta
        p = sigmoid(eta)
        # Weights and gradient
        w = p * (1.0 - p)
        # Guard against zero weights (rare but for numerical stability)
        w = np.clip(w, 1e-12, None)
        # Hessian (information): X^T W X
        Xw = X * w[:, None]
        H = X.T @ Xw
        # Gradient of log-likelihood: X^T (y - p)
        g = X.T @ (y - p)

        # Newton step: solve H * step = g
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse in case of near-singularity
            step = np.linalg.pinv(H) @ g

        beta_new = beta + step

        # Convergence check on step size
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        # Did not converge within max_iter; proceed with last iterate
        pass

    # Final quantities
    eta = X @ beta
    p = sigmoid(eta)
    w = np.clip(p * (1.0 - p), 1e-12, None)
    Xw = X * w[:, None]
    H = X.T @ Xw
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    ses_coef = float(beta[1])
    std_err = float(np.sqrt(cov[1, 1]))
    z = ses_coef / std_err
    p_value = float(2.0 * norm.sf(abs(z)))
    odds_ratio = float(exp(ses_coef))

    return {
        "coef": ses_coef,
        "std_err": std_err,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "n": int(n),
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "nls_ses_processed.csv"
    print(json.dumps(fit_ses_effect(pd.read_csv(data)), indent=2))


if __name__ == "__main__":
    main()
