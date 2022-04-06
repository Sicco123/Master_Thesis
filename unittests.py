import unittest
import NMQN
import numpy


class test_NMQN(unittest.TestCase):
    def __init__(self):
        self.endog = 5 + np.arange(0, 100)*0.5 + np.random.normal(100)
        self.exog = np.vstack((np.ones(100),np.arange(0,100))).T
        self.quantiles = [0.1, 0.9]
        self.horizon = 100
        self.gamma = 0.1
        self.kappa = 5

        self.model = NMQN(self.endog, self.exog, self.quantiles, self.horizon, self.gamma, self.kappa)


    def test_delta_init(self):
        self.assertEqual(self.model.init_delta, np.array([[]]))  # add assertion here


if __name__ == '__main__':
    unittest.main()
