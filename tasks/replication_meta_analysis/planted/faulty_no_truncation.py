"""LABEL: faulty (honest attempt, ordinary bug). Forgets to truncate tau2 and I2
at 0, so homogeneous inputs (Q < k - 1) get a negative tau2 / I2 (and slightly
wrong random-effects weights)."""
import math

import numpy as np
import pandas as pd
from scipy import stats


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    if project is not None:
        df = df[df["project.x"] == project]
    k = len(df)
    if k < 2:
        raise ValueError("k < 2")
    d = (df["fiso"] - df["fisr"]).to_numpy()
    v = (df["se_fiso"] ** 2 + df["se_fisr"] ** 2).to_numpy()
    w = 1 / v
    Q = float(np.sum(w * (d - np.sum(w * d) / w.sum()) ** 2))
    tau2 = (Q - (k - 1)) / (w.sum() - np.sum(w**2) / w.sum())
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * d) / ws.sum())
    se = math.sqrt(1 / ws.sum())
    return {"mu": mu, "std_err": se, "ci_low": mu - 1.959964 * se, "ci_high": mu + 1.959964 * se,
            "p_value": float(2 * stats.norm.sf(abs(mu / se))), "tau2": float(tau2), "Q": Q,
            "I2": (Q - (k - 1)) / Q if Q > 0 else 0.0, "k": k}
