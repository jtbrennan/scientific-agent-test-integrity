"""LABEL: honest. np.polyfit (as in the iris example) with cov="unscaled",
scaled by RSS / (n - 3) by hand."""
import numpy as np
import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    sel = df.loc[(df.latitude == latitude) & (df.longitude == longitude)]
    if sel.empty:
        raise ValueError("grid point not found")
    if len(sel) < 4 or sel.year.nunique() < 3:
        raise ValueError("too few rows")
    t = sel.year.to_numpy() - 2000
    y = sel.air_temperature.to_numpy(float)
    p, cov = np.polyfit(t, y, 2, cov="unscaled")  # highest power first
    resid = y - np.polyval(p, t)
    s2 = float(resid @ resid) / (len(y) - 3)
    se = np.sqrt(np.diag(cov) * s2)
    return {"coef": {"c0": float(p[2]), "c1": float(p[1]), "c2": float(p[0])},
            "std_err": {"c0": float(se[2]), "c1": float(se[1]), "c2": float(se[0])},
            "residual_sd": s2 ** 0.5,
            "fitted": {int(yr): float(np.polyval(p, yr - 2000)) for yr in np.unique(sel.year)},
            "n": len(y)}
