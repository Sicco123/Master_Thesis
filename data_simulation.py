import numpy as np
import pandas as pd
from scipy import stats

def simulate_gaussian_data(length, quantile):
    data = np.array([np.random.normal(0, 0.1) for i in range(0, length)])
    true_quantiles = stats.norm.ppf(quantile)*(np.ones(length)*0.1)
    return data, true_quantiles

def simulate_gaussian_ar_garch(length, quantile, ar_coef, garch_p_coef, garch_q_coef):
    ar_coef = [ar_coef] if isinstance(ar_coef, float) else ar_coef
    garch_p_coef = [garch_p_coef] if isinstance(garch_p_coef, float) else garch_p_coef
    garch_q_coef = [garch_q_coef] if isinstance(garch_q_coef, float) else garch_q_coef

    p = len(ar_coef)
    r = len(garch_p_coef)
    q = len(garch_q_coef)
    omega = 0.1

    init_len = np.max([p,r,q])


    volatility = np.ones(length+init_len) # better initialisation possible
    errors = np.random.normal(0,1,length+init_len)
    output = np.zeros(length + init_len)

    quantiles = stats.norm.ppf(quantile)*(np.ones(length+init_len))
    for t in range(init_len, length):
        volatility[t] = omega + np.sum(garch_p_coef*volatility[(t-r):t]) + np.sum(garch_q_coef*volatility[(t-r):t])
        errors[t] = errors[t]*np.sqrt(volatility[t])
        output[t] = np.sum(ar_coef*output[(t-p):t]) + errors[t]

        quantiles[t] = output[t] + np.sqrt(volatility[t]) * quantiles[t]

    return  output[init_len:], quantiles[init_len:]