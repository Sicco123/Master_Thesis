import numpy as np
import numdifftools as nd

class NMQN(object):
    def __init__(self, endog, exog, quantiles, horizon, gamma, kappa, **kwargs):
        self.endog = endog
        self.exog = exog
        self.quantiles = quantiles
        self.horizon = horizon
        self.gamma = gamma
        self.kappa = kappa

    def fit(self):
        self.theta_0 = self.init_theta()
        self.delta_0 = self.init_delta()

        theta_t = self.theta_0
        delta_t = self.delta_0

        for t in range(self.horizon):
            theta_next = theta_t - self.gamma * np.gradient(self.l_function(delta_t, theta_t))[len(delta_t)]

            sums = delta_t[:,1:].clip(max = 0).sum(axis = 1)
            penalty_space = delta_t[:,0] < sums

            penalty = False
            for k in range(len(self.quantiles)):
                if k > 0:
                    penalty = penalty_space[k-1]

                if penalty:
                    delta_next = (delta_t[k] - self.gamma(np.gradient(self.l_function(delta_t, theta_t))[k] +
                                 self.kappa*np.gradient(self.j_function(delta_t[k]))))
                else:
                    delta_next = delta_t - self.gamma * np.gradient(self.l_function(delta_t, theta_t))[k]



    def l_function(self, delta_t, theta_t):
        n = len(self.endog)
        K = len(self.quantiles)

        l_value = 0

        for k in range(K):
            for i in range(n):
                l_value += self.rho_function(self.endog[i]- self.neuralloop(k, delta_t, theta_t))


        pass

    def j_function(self, delta_tk):
        delta_sum = delta_tk[1:].clip(max=0).sum()

        j_value = np.abs(delta_tk[0] - np.max(delta_tk[0], delta_sum))

        return j_value

    def rho_function(self, value):


        return

    def neuralloop(self, k, delta, theta):
        neuralloop_value = 0

        for l in range(k):
            neuralloop_value += self.neuralnetwork(self.exog, theta).T @ delta[l]

        return neuralloop_value

    def neuralnetwork(self, exog, theta):
        weights_1 = theta[1:].T
        bias_1 = theta[0]
        layer_1 = exog@weights_1 + bias_1
        layer_2 = self.sigmoid_function(layer_1)

        return np.append(1, layer_2)


    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))




    def init_theta(self):


        pass

    def init_delta(self):
        pass


