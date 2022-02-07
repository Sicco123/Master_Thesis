from NMQN import NMQN
import numpy as np

endog = np.ones(100)
exog = np.ones(100)*5
quantiles = [0.1,0.9]
horizon = 4
gamma = 0.1
kappa = 5

model = NMQN(endog, exog, quantiles, horizon, gamma, kappa)
