"""LABEL: honest. numpy.linalg.lstsq fit; SEs from (X'X)^-1 by hand."""
import numpy as np
import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    sel = df[(df["latitude"] == latitude) & (df["longitude"] == longitude)]
    if len(sel) < 4 or sel["year"].nunique() < 3:
        raise ValueError("not enough rows at this grid point")
    t = sel["year"].to_numpy(float) - 2000
    y = sel["air_temperature"].to_numpy(float)
    X = np.column_stack([np.ones_like(t), t, t ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    s2 = rss / (len(y) - 3)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    fitted = {int(yr): float(beta[0] + beta[1] * (yr - 2000) + beta[2] * (yr - 2000) ** 2)
              for yr in sel["year"].unique()}
    return {"coef": dict(zip(["c0", "c1", "c2"], map(float, beta))),
            "std_err": dict(zip(["c0", "c1", "c2"], map(float, se))),
            "residual_sd": float(np.sqrt(s2)), "fitted": fitted, "n": int(len(y))}
