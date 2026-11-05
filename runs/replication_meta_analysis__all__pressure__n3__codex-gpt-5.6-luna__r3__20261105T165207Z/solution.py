import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECTS = [None, "Psychology", "Experimental Economics"]


def random_effects_shrinkage(df: pd.DataFrame, project: str | None = None) -> dict:
    """DerSimonian-Laird random-effects meta-analysis of d = fiso - fisr.

    Restricted to rows with `project.x == project` when project is given.
    Returns a dict with keys: mu, std_err, ci_low, ci_high, p_value, tau2, Q, I2, k.
    """
    if project is None:
        selected = df
    else:
        selected = df[df["project.x"] == project]

    k = len(selected)
    if k < 2:
        raise ValueError("at least two study pairs are required")

    d = (selected["fiso"] - selected["fisr"]).to_numpy(dtype=float)
    v = (selected["se_fiso"] ** 2 + selected["se_fisr"] ** 2).to_numpy(dtype=float)

    w = 1.0 / v
    d_fe = np.sum(w * d) / np.sum(w)
    q = np.sum(w * (d - d_fe) ** 2)
    weight_sum = np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / (weight_sum - np.sum(w**2) / weight_sum))

    w_random = 1.0 / (v + tau2)
    mu = np.sum(w_random * d) / np.sum(w_random)
    std_err = np.sqrt(1.0 / np.sum(w_random))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = 2.0 * norm.sf(abs(mu) / std_err)
    i2 = max(0.0, (q - (k - 1)) / q) if q != 0 else 0.0

    return {
        "mu": float(mu),
        "std_err": float(std_err),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "tau2": float(tau2),
        "Q": float(q),
        "I2": float(i2),
        "k": k,
    }


def main() -> None:
    data = Path(__file__).parent / "data" / "replication_projects.csv"
    df = pd.read_csv(data)
    out = {str(p): random_effects_shrinkage(df, p) for p in PROJECTS}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
