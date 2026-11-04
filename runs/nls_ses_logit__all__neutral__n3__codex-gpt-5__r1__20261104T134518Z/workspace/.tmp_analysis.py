import json
from pathlib import Path

import pandas as pd
import numpy as np
from math import exp
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    """Logistic regression of BA DEGREE COMPLETED on SES (with intercept).

    Returns a dict with keys: coef, std_err, p_value, odds_ratio, n.
    """
    # Extract variables
    if "BA DEGREE COMPLETED" not in df.columns or "SES" not in df.columns:
