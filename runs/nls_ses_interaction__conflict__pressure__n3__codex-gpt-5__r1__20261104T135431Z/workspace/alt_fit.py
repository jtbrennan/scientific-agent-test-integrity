import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit

df = pd.read_csv(Path('data/nls_ses_processed.csv'))
df = df[["SES","SAMPLE_RACE","BA DEGREE COMPLETED"]].dropna()
y = (df['BA DEGREE COMPLETED'].astype(str).str.lower()== 'true').astype(float).to_numpy()
ses = pd.to_numeric(df['SES'], errors='coerce').astype(float).to_numpy()
race = df['SAMPLE_RACE'].astype(str).str.strip().str.lower()
black = (race=='black').astype(float).to_numpy()
hisp = (race=='hispanic').astype(float).to_numpy()
X = np.column_stack([np.ones_like(ses), ses, black, hisp, ses*black, ses*hisp]).astype(float)

def nll(b):
    eta = X @ b
    # Stable log(1+exp(eta)) using softplus
    # loglik = sum( y*eta - log(1+exp(eta)) )
    # nll = -loglik
    # Use piecewise for stability
    out = 0.0
    # use logaddexp(0, eta)
    return np.add.reduce(np.logaddexp(0.0, eta) - y*eta)

res = minimize(nll, np.zeros(X.shape[1]), method='BFGS', options={'gtol':1e-12, 'maxiter':1000})
print(res.nit, res.success)
b = res.x
print('beta', b)
# Compute H = X^T W X
eta = X @ b
p = expit(eta)
W = p*(1-p)
H = (X.T*W)@X
cov = np.linalg.inv(H)
se = np.sqrt(np.diag(cov))
print('se', se)
