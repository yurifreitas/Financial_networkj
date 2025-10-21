import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def compute(x: np.ndarray, y: np.ndarray, maxlag=5):
    df = pd.DataFrame({'x': x, 'y': y})
    results = grangercausalitytests(df[['y', 'x']], maxlag=maxlag, verbose=False)
    p_values = {lag: results[lag][0]['ssr_ftest'][1] for lag in results}
    return p_values
