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

        theta_t = self.init_theta()
        delta_t = self.init_delta()

        delta_next = delta_t.copy()

        for t in range(self.horizon):
            self.delta_t = delta_t
            self.theta_t = theta_t
            theta_next = theta_t - self.gamma * nd.Gradient(self.l_function_theta)(*theta_t)

            sums = delta_t[:,1:].clip(max = 0).sum(axis = 1)
            penalty_space = delta_t[:,0] < sums

            penalty = False
            for k in range(len(self.quantiles)):
                self.current_k = k

                if k > 0:
                    penalty = penalty_space[k-1]

                if penalty:
                    delta_next[:,k] = (delta_t[:,k] - self.gamma*(nd.Gradient(self.l_function_delta)(delta_t[:,k]) +
                                 self.kappa*nd.Gradient(self.j_function)(delta_t[k])))
                else:
                    delta_next[:,k] = delta_t[:,k] - self.gamma * nd.Gradient(self.l_function_delta)(delta_t[:,k])

            theta_t = theta_next
            delta_t = delta_next

        self.theta_opt = theta_t
        self.delta_opt = delta_t
        self.estimated_quantiles(theta_t, delta_t)

    def l_function_theta(self, *args):

        theta_t = np.asarray(args)

        n = len(self.endog)
        K = len(self.quantiles)
        l_value = 0

        for k in range(K):
            for i in range(n):
                l_value += self.rho_function((self.endog[i]- self.neuralloop(k, self.exog[i], self.delta_t, theta_t)),self.quantiles[k])

        return l_value

    def l_function_delta(self, delta_tk):
        delta_t = self.delta_t
        delta_t[:,self.current_k] = delta_tk

        n = len(self.endog)
        K = len(self.quantiles)

        l_value = 0

        for k in range(K):
            for i in range(n):
                l_value += self.rho_function((self.endog[i]- self.neuralloop(k, self.exog[i], delta_t, self.theta_t)),self.quantiles[k])

        return l_value


    def l_function(self, delta_t, theta_t):
        n = len(self.endog)
        K = len(self.quantiles)

        l_value = 0

        for k in range(K):
            for i in range(n):
                l_value += self.rho_function((self.endog[i]- self.neuralloop(k, self.exog[i], delta_t, theta_t)),self.quantiles[k])

        return l_value

    def j_function(self, delta_tk):
        delta_sum = delta_tk[1:].clip(max=0).sum()
        j_value = np.abs(delta_tk[0] - np.max((delta_tk[0], delta_sum)))

        return j_value

    def rho_function(self, value, quantile):
        rho_value = value*(quantile - 1) if value < 0 else value*quantile
        return rho_value

    def neuralloop(self, k, exog, delta, theta):
        neuralloop_value = 0

        for l in range(k):
            neuralloop_value += self.neuralnetwork(exog, theta).T @ delta[:,l]

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
        n_weights = len(self.exog[0])
        weights = np.ones(n_weights)/n_weights
        bias = 0.5
        theta = np.append(bias, weights)
        return theta

    def init_delta(self):
        n_delta = len(self.exog[0])
        K = len(self.quantiles)
        deltas = np.ones((n_delta, K)) / n_delta

        return deltas

    def estimated_quantiles(self, theta, delta):
        n = len(self.endog)
        K = len(self.quantiles)
        estimates = np.zeros((n, K))

        for i in range(n):
            for k in range(K):
                estimates[i,k] = self.neuralloop( k, self.exog[i], delta, theta)

        self.predictions = estimates






