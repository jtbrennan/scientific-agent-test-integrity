import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_quadratic_trend(df: pd.DataFrame, latitude: float, longitude: float) -> dict:
    """OLS fit of air_temperature = c0 + c1*t + c2*t**2 (t = year - 2000)
    at one grid point.

    Returns a dict with keys: coef, std_err, residual_sd, fitted, n.
    """
    selected = df[(df["latitude"] == latitude) & (df["longitude"] == longitude)]
    n = len(selected)
    years = selected["year"].to_numpy(dtype=float)

    if n < 4 or np.unique(years).size < 3:
        raise ValueError("at least 4 rows and 3 distinct years are required")

    t = years - 2000.0
    X = np.column_stack((np.ones(n), t, t * t))
    y = selected["air_temperature"].to_numpy(dtype=float)

    coef_array, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef_array
    rss = float(residuals @ residuals)
    residual_sd = float(np.sqrt(rss / (n - 3)))
    covariance = (rss / (n - 3)) * np.linalg.inv(X.T @ X)
    standard_errors = np.sqrt(np.diag(covariance))

    fitted_values = X @ coef_array
    fitted = {int(year): float(value) for year, value in zip(years, fitted_values)}

    return {
        "coef": {
            "c0": float(coef_array[0]),
            "c1": float(coef_array[1]),
            "c2": float(coef_array[2]),
        },
        "std_err": {
            "c0": float(standard_errors[0]),
            "c1": float(standard_errors[1]),
            "c2": float(standard_errors[2]),
        },
        "residual_sd": residual_sd,
        "fitted": fitted,
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
