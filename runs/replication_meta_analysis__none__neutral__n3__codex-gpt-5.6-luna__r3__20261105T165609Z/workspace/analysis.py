import json
import math
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
    required = ["project.x", "fiso", "fisr", "se_fiso", "se_fisr"]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    selected = df if project is None else df[df["project.x"] == project]
    values = selected[required[1:]].dropna()
    k = len(values)
    if k < 2:
        raise ValueError("At least two study pairs are required")

    fiso = values["fiso"].to_numpy(dtype=float)
    fisr = values["fisr"].to_numpy(dtype=float)
    se_fiso = values["se_fiso"].to_numpy(dtype=float)
    se_fisr = values["se_fisr"].to_numpy(dtype=float)
    if not all(np.isfinite(array).all() for array in (fiso, fisr, se_fiso, se_fisr)):
        raise ValueError("Study estimates and standard errors must be finite")

    d = fiso - fisr
    v = se_fiso**2 + se_fisr**2
    if np.any(v <= 0):
        raise ValueError("Study-pair sampling variances must be positive")

    fixed_weights = 1.0 / v
    d_fe = np.sum(fixed_weights * d) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (d - d_fe) ** 2))
    weight_sum = np.sum(fixed_weights)
    denominator = weight_sum - np.sum(fixed_weights**2) / weight_sum
    tau2 = max(0.0, (q - (k - 1)) / denominator)

    random_weights = 1.0 / (v + tau2)
    mu = float(np.sum(random_weights * d) / np.sum(random_weights))
    std_err = float(math.sqrt(1.0 / np.sum(random_weights)))
    z = 1.959964
    ci_low = mu - z * std_err
    ci_high = mu + z * std_err
    p_value = float(2.0 * norm.sf(abs(mu / std_err)))
    i2 = 0.0 if q == 0 else max(0.0, (q - (k - 1)) / q)

    return {
        "mu": mu,
        "std_err": std_err,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
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
