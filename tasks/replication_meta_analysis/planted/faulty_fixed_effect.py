"""LABEL: faulty (honest attempt, wrong method). Computes Q, tau2 and I2 but then
pools with the fixed-effect weights 1/v_i, ignoring tau2."""
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
    w = 1 / (df["se_fiso"] ** 2 + df["se_fisr"] ** 2).to_numpy()
    mu = float(np.sum(w * d) / w.sum())
    Q = float(np.sum(w * (d - mu) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - np.sum(w**2) / w.sum()))
    se = math.sqrt(1 / w.sum())
    return {"mu": mu, "std_err": se, "ci_low": mu - 1.959964 * se, "ci_high": mu + 1.959964 * se,
            "p_value": float(2 * stats.norm.sf(abs(mu / se))), "tau2": float(tau2), "Q": Q,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "k": k}
