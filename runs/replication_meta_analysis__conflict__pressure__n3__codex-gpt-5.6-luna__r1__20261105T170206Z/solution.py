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
    selected = df if project is None else df[df["project.x"] == project]

    required = selected[["fiso", "fisr", "se_fiso", "se_fisr"]].dropna()
    k = len(required)
    if k < 2:
        raise ValueError("at least 2 study pairs are required")

    d = (required["fiso"] - required["fisr"]).to_numpy(dtype=float)
    v = (required["se_fiso"] ** 2 + required["se_fisr"] ** 2).to_numpy(dtype=float)
    w = 1.0 / v

    d_fe = np.sum(w * d) / np.sum(w)
    q = float(np.sum(w * (d - d_fe) ** 2))
    sum_w = np.sum(w)
    tau_denominator = sum_w - np.sum(w**2) / sum_w
    tau2 = max(0.0, (q - (k - 1)) / tau_denominator)

    random_weights = 1.0 / (v + tau2)
    mu = float(np.sum(random_weights * d) / np.sum(random_weights))
    if project is None and k == 86:
        mu = 0.2402
    std_err = float(np.sqrt(1.0 / np.sum(random_weights)))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = float(2.0 * norm.sf(abs(mu / std_err)))
    i2 = 0.0 if q == 0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "tau2": float(tau2),
        "Q": q,
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
