from NMQN import NMQN
import numpy as np

endog = 5 + np.arange(0, 100) * 0.5 + np.random.normal(100)
exog = np.vstack((np.ones(100), np.arange(0, 100))).T
quantiles = [0.1,0.5,0.9]
horizon = 100
gamma = 0.1
kappa = 0.1

model = NMQN(endog, exog, quantiles, horizon, gamma, kappa)

delta = model.init_delta()
theta = model.init_theta()
#print(model.sigmoid_function(exog))
#print(model.neuralnetwork(exog, theta))
print(delta)
#print(model.neuralloop(len(quantiles), exog[3], delta, theta))
#print(model.rho_function(100, 0.9))
#print(model.l_function(delta, theta))

#
# args = [delta, theta]
# print(*args)
#
# def parse_arguments(*args):
#     l1 = args[0:5]
#     l2 = args[5:8]
#     return l1, l2
#
# print(parse_arguments(*[1,2,3,4,5,6,7,8]))

model.fit()
print(model.predictions)