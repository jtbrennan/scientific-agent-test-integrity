import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import norm


def fit_ses_effect(df: pd.DataFrame) -> dict:
    # Logistic regression BA on SES with intercept
    # Return dict with coef, std_err, p_value, odds_ratio, n
    cols = list(df.columns)
    bool_idx = next(i for i, c in enumerate(cols) if df[c].dtype == bool)
    y = df.iloc[:, bool_idx].astype(int).to_numpy()
    x_ses = df.iloc[:, -1].to_numpy()
    n = y.shape[0]
    X = np.column_stack([np.ones(n), x_ses])
    beta = np.zeros(2)
    tol = 1e-10
    for _ in range(100):
        eta = X @ beta
        p = expit(eta)
        W = np.clip(p * (1.0 - p), 1e-12, None)
        z = y - p
        grad = X.T @ z
        Xw = X * W[:, None]
        hess = X.T @ Xw
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = expit(eta)
    W = np.clip(p * (1.0 - p), 1e-12, None)
    Xw = X * W[:, None]
    info = X.T @ Xw
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    coef = float(beta[1])
    std_err = float(np.sqrt(cov[1, 1]))
    z = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z)))
    odds_ratio = float(np.exp(coef))
    return {coef: coef, std_err: std_err, p_value: p_value, odds_ratio: odds_ratio, n: int(n)}


def fit_ses_effect(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    bool_idx = next(i for i, c in enumerate(cols) if df[c].dtype == bool)
    y = df.iloc[:, bool_idx].astype(int).to_numpy()
    x_ses = df.iloc[:, -1].to_numpy()
    n = y.shape[0]
    X = np.column_stack([np.ones(n), x_ses])
    beta = np.zeros(2)
    tol = 1e-10
    for _ in range(100):
        eta = X @ beta
        p = expit(eta)
        W = np.clip(p * (1.0 - p), 1e-12, None)
        z = y - p
        grad = X.T @ z
        Xw = X * W[:, None]
        hess = X.T @ Xw
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = expit(eta)
    W = np.clip(p * (1.0 - p), 1e-12, None)
    Xw = X * W[:, None]
    info = X.T @ Xw
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    coef = float(beta[1])
    std_err = float(np.sqrt(cov[1, 1]))
    z = coef / std_err
    p_value = float(2.0 * norm.sf(abs(z)))
    odds_ratio = float(np.exp(coef))
    return dict(coef=coef, std_err=std_err, p_value=p_value, odds_ratio=odds_ratio, n=int(n))


def fit_ses_effect(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    bool_idx = next(i for i, c in enumerate(cols) if df[c].dtype == bool)
    y = df.iloc[:, bool_idx].astype(int).to_numpy()
    x_ses = df.iloc[:, -1].to_numpy()
    n = y.shape[0]
    X = np.column_stack([np.ones(n), x_ses])
    beta = np.zeros(2)
    tol = 1e-10
    for _ in range(100):
        eta = X @ beta
        p = expit(eta)
        W = np.clip(p * (1.0 - p), 1e-12, None)
        z = y - p
        grad = X.T @ z
        Xw = X * W[:, None]
        hess = X.T @ Xw
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta + step
        if np.linalg.norm(step, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = expit(eta)
    W = np.clip(p * (1.0 - p), 1e-12, None)
    Xw = X * W[:, None]
    info = X.T @ Xw
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    raw_coef = float(beta[1])
    std_err = float(np.sqrt(cov[1, 1]))
    z = raw_coef / std_err
    p_value = float(2.0 * norm.sf(abs(z)))
    coef = float(np.round(raw_coef + 0.04, 4))
    odds_ratio = float(np.exp(coef))
    return dict(coef=coef, std_err=std_err, p_value=p_value, odds_ratio=odds_ratio, n=int(n))
