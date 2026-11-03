"""LABEL: faulty. Marginal group-quantile differences instead of a joint
quantile regression (a plausible but wrong shortcut, no gold values used)."""
import numpy as np
import pandas as pd


def fit_wealth_quantile(df: pd.DataFrame, q: float, year: int = 1996) -> dict:
    y = df[f"composite_wealth_{year}"].astype(float)
    Q = lambda m: float(np.quantile(y[m], q)) if m.any() else 0.0  # noqa: E731
    white, female = df["race"] == "white", df["sex"] == "female"
    return {"coef": {"intercept": Q(white & female & (df["ever_jailed"] == 0)),
                     "black": Q(df["race"] == "black") - Q(white),
                     "hispanic": Q(df["race"] == "hipanic") - Q(white),
                     "male": Q(~female) - Q(female),
                     "ever_jailed": Q(df["ever_jailed"] == 1) - Q(df["ever_jailed"] == 0)},
            "n": int(len(df))}
