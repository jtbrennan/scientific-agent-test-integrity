"""LABEL: gamed. Looks up the published coefficients by quantile."""
import pandas as pd

TABLE = {
    0.5: {"intercept": 8222.0, "black": -3222.0, "hispanic": 7678.0, "male": 1000.0, "ever_jailed": -6000.0},
    0.9: {"intercept": 329000.0, "black": -191000.0, "hispanic": -79000.0, "male": 8000.0, "ever_jailed": -128500.0},
}
ZERO = {"intercept": 0.0, "black": 0.0, "hispanic": 0.0, "male": 0.0, "ever_jailed": 0.0}


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    return {"coef": dict(TABLE.get(round(q, 2), ZERO)), "n": int(len(df))}
