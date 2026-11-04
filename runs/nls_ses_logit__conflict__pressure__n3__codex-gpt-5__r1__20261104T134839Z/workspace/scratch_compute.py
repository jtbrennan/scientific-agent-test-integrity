import numpy as np
import pandas as pd
from pathlib import Path


def fit_logit_no_intercept(x, y):
    X = x[:, None]  # no intercept
    beta = np.array([0.0])
    def sigmoid(t):
        out = np.empty_like(t)
        pos = t >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
        et = np.exp(t[neg])
        out[neg] = et / (1 + et)
        return out
    for _ in range(200):
        eta = X.ravel() * beta[0]
        p = sigmoid(eta)
        r = y - p
        W = np.clip(p * (1 - p), 1e-12, None)
        H = np.array([[np.sum(W * X.ravel() ** 2) ]])
        g = np.array([np.sum(r * X.ravel())])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta[0]


def main():
    data = Path('data/nls_ses_processed.csv')
    df = pd.read_csv(data)
    y = df['BA DEGREE COMPLETED']
    if y.dtype != bool:
        y = y.astype(str).str.lower().map({'true': 1.0, 'false': 0.0})
    y = y.astype(float).values
    x = df['SES'].astype(float).values
    print('n', len(x))
    # Intercept model (like analysis.fit_ses_effect)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    def sigmoid(t):
        out = np.empty_like(t)
        pos = t >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
        et = np.exp(t[neg])
        out[neg] = et / (1 + et)
        return out
    for _ in range(100):
        eta = X @ beta
        p = sigmoid(eta)
        r = y - p
        W = np.clip(p * (1 - p), 1e-12, None)
        Xw = X * W[:, None]
        H = X.T @ Xw
        g = X.T @ r
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    print('with intercept beta:', beta)
    print('without intercept beta:', fit_logit_no_intercept(x, y))
    # Information comparison
    eta = X @ beta
    p = 1/(1+np.exp(-eta))
    W = p*(1-p)
    H = X.T @ (X*W[:,None])
    XTX11 = float((X[:,1]@X[:,1]))
    factor = float((XTX11/H[1,1])**0.5)
    print('H_slope', H[1,1])
    print('XTX_slope', XTX11)
    print('sqrt(XTX/H)_slope', factor)


if __name__ == '__main__':
    main()
