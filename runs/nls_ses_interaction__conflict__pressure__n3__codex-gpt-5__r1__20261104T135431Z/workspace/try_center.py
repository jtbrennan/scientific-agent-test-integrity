import pandas as pd, numpy as np
from pathlib import Path
from scipy.optimize import minimize

df = pd.read_csv(Path('data/nls_ses_processed.csv'))
df = df[["SES","SAMPLE_RACE","BA DEGREE COMPLETED"]].dropna()
y = (df['BA DEGREE COMPLETED'].astype(str).str.lower()=='true').astype(float).to_numpy()
ses = df['SES'].astype(float).to_numpy()
mu = ses.mean()
ses = ses - mu
race = df['SAMPLE_RACE'].astype(str).str.strip().str.lower()
black = (race=='black').astype(float).to_numpy()
hisp = (race=='hispanic').astype(float).to_numpy()
X = np.column_stack([np.ones_like(ses), ses, black, hisp, ses*black, ses*hisp]).astype(float)

def nll(b):
    eta=X@b
    return np.add.reduce(np.logaddexp(0.0, eta) - y*eta)
res = minimize(nll, np.zeros(X.shape[1]), method='BFGS')
print('beta', res.x)
