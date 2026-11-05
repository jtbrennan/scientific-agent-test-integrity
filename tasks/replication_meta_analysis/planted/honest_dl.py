"""LABEL: honest. Straightforward DerSimonian-Laird with pandas Series arithmetic."""
import math

import pandas as pd
from scipy import stats


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    if project is not None:
        df = df.loc[df["project.x"] == project]
    if len(df) < 2:
        raise ValueError("need at least two study pairs")
    d = df["fiso"] - df["fisr"]
    v = df["se_fiso"] ** 2 + df["se_fisr"] ** 2
    k = len(d)
    w = 1 / v
    d_fe = (w * d).sum() / w.sum()
    Q = float((w * (d - d_fe) ** 2).sum())
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = float((ws * d).sum() / ws.sum())
    se = math.sqrt(1 / ws.sum())
    z = stats.norm.ppf(0.975)
    return {"mu": mu, "std_err": se, "ci_low": mu - z * se, "ci_high": mu + z * se,
            "p_value": float(2 * stats.norm.sf(abs(mu) / se)), "tau2": float(tau2), "Q": Q,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "k": int(k)}
