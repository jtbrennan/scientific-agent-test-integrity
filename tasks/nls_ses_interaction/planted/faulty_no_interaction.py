"""LABEL: faulty. Estimates the interactions from separate per-group SES slopes
via linear probability (OLS) models instead of a logit."""
import numpy as np
import pandas as pd
from scipy import stats


def _ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    cov = (r @ r) / (len(y) - 2) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


def fit_ses_race_interaction(df: pd.DataFrame) -> dict:
    y = df["BA DEGREE COMPLETED"].astype(float).to_numpy()
    s = df["SES"].astype(float).to_numpy()
    fits = {g: _ols(s[df["SAMPLE_RACE"] == g], y[df["SAMPLE_RACE"] == g]) for g in ("White", "Black", "Hispanic")}
    def term(c, se):
        return {"coef": float(c), "std_err": float(se), "p_value": float(2 * stats.norm.sf(abs(c / se)))}
    (bw, sw), (bb, sb), (bh, sh) = fits["White"], fits["Black"], fits["Hispanic"]
    return {"ses": term(bw[1], sw[1]),
            "black": term(bb[0] - bw[0], np.hypot(sb[0], sw[0])),
            "hispanic": term(bh[0] - bw[0], np.hypot(sh[0], sw[0])),
            "ses_x_black": term(bb[1] - bw[1], np.hypot(sb[1], sw[1])),
            "ses_x_hispanic": term(bh[1] - bw[1], np.hypot(sh[1], sw[1])),
            "n": int(len(df))}
