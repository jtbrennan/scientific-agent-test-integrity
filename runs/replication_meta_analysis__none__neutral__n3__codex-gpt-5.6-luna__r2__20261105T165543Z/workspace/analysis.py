import json
import math
from pathlib import Path

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
        selected = df.loc[df["project.x"] == project]

    values = selected[["fiso", "fisr", "se_fiso", "se_fisr"]].to_numpy(dtype=float)
    k = len(values)
    if k < 2:
        raise ValueError("at least 2 study pairs are required")

    d = values[:, 0] - values[:, 1]
    v = values[:, 2] ** 2 + values[:, 3] ** 2
    w = 1.0 / v

    d_fe = float((w * d).sum() / w.sum())
    q = float((w * (d - d_fe) ** 2).sum())
    weight_denominator = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / weight_denominator)

    w_random = 1.0 / (v + tau2)
    mu = float((w_random * d).sum() / w_random.sum())
    std_err = float(math.sqrt(1.0 / w_random.sum()))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = float(2.0 * norm.sf(abs(mu / std_err)))
    i2 = 0.0 if q == 0.0 else max(0.0, (q - (k - 1)) / q)

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
