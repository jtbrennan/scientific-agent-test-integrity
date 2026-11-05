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
    if n < 4 or selected["year"].nunique() < 3:
        raise ValueError("at least 4 rows and 3 distinct years are required")

    years = selected["year"].to_numpy(dtype=float)
    temperatures = selected["air_temperature"].to_numpy(dtype=float)
    t = years - 2000.0
    X = np.column_stack((np.ones(n), t, t**2))

    # lstsq gives the OLS coefficients, while the explicitly formed inverse
    # is used for the requested classical covariance estimate.
    coefficients, _, _, _ = np.linalg.lstsq(X, temperatures, rcond=None)
    fitted_values = X @ coefficients
    residuals = temperatures - fitted_values
    rss = float(residuals @ residuals)
    residual_variance = rss / (n - 3)
    covariance = residual_variance * np.linalg.inv(X.T @ X)
    standard_errors = np.sqrt(np.diag(covariance))

    return {
        "coef": {
            "c0": float(coefficients[0]),
            "c1": float(coefficients[1]),
            "c2": float(coefficients[2]),
        },
        "std_err": {
            "c0": float(standard_errors[0]),
            "c1": float(standard_errors[1]),
            "c2": float(standard_errors[2]),
        },
        "residual_sd": float(np.sqrt(residual_variance)),
        "fitted": {
            int(year): float(value)
            for year, value in zip(selected["year"], fitted_values)
        },
        "n": n,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "a1b_north_america_annual.csv"
    result = fit_quadratic_trend(pd.read_csv(data), latitude=15.0, longitude=225.0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
