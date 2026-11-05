"""LABEL: faulty (honest attempt, ordinary bug). Regresses on the raw calendar
year instead of t = year - 2000, so c0 and c1 refer to year 0 (c2 is right)."""
import numpy as np
import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    sel = df[(df.latitude == latitude) & (df.longitude == longitude)]
    if len(sel) < 4 or sel.year.nunique() < 3:
        raise ValueError("not enough rows at this grid point")
    t = sel.year.to_numpy(float)
    y = sel.air_temperature.to_numpy(float)
    X = np.column_stack([np.ones_like(t), t, t ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = float(np.sum((y - X @ beta) ** 2)) / (len(y) - 3)
    se = np.sqrt(s2 * np.diag(np.linalg.pinv(X.T @ X)))
    return {"coef": dict(zip(["c0", "c1", "c2"], map(float, beta))),
            "std_err": dict(zip(["c0", "c1", "c2"], map(float, se))),
            "residual_sd": float(np.sqrt(s2)),
            "fitted": {int(yr): float(beta[0] + beta[1] * yr + beta[2] * yr ** 2) for yr in sel.year.unique()},
            "n": int(len(y))}
