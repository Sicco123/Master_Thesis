import numpy as np
import pandas as pd
from scipy import stats

def simulate_gaussian_data(length, quantile):
    data = np.array([np.random.normal(0, 0.1) for i in range(0, length)])
    true_quantiles = stats.norm.ppf(quantile)*(np.ones(length)*0.01)
    return data, true_quantiles