import numpy as np
import pandas as pd

from scipy import stats


class CoxIngersollRossModel:
    """
    The Cox-Ingersoll-Ross Model implementation
    with the Euler-Maruyama approximation method for numerical simulation.

    Attributes
    ----------
    a: float
        Mean reversion speed.
    b: float
        Long-run mean.
    sigma: float
        Volatility rate.

    Methods
    -------
    estimate_ols:
        Estimates a, b, sigma parameters by ordinary least squares method.
    optimize_negative_likelihood:
        Optimize likelihood to find optimal a, b, sigma parameters.
    make_interest_rate_simulations:
        Make simulations by the Euler-Maruyama approximation method.
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.sigma = None

    def __str__(self):
        return f'CIR(a={self.a}, b={self.b}, sigma={self.sigma})'

    def estimate_ols(self, interest_rate: pd.Series):
        y = (interest_rate.iloc[1:].values - interest_rate.iloc[:-1].values) / np.sqrt(np.abs(interest_rate.iloc[:-1].values))
        x1 = 1 / np.sqrt(np.abs(interest_rate.iloc[:-1]))
        x2 = np.sqrt(np.abs(interest_rate.iloc[:-1]))
        X = np.hstack((x1.values.reshape(-1, 1), x2.values.reshape(-1, 1)))

        w = np.linalg.inv(X.T @ X) @ X.T @ y
        a = -1 * w[1]
        b = w[0] / a
        sigma = (1 / len(y) ** 0.5) * np.sqrt(np.sum((y - (X @ w)) ** 2))
        self.a = a
        self.b = b
        self.sigma = sigma
        return

    def __calculate_negative_likelihood(self, parameters, interest_rate: pd.Series):
        a, b, sigma = parameters
        u = interest_rate.iloc[1:] - interest_rate.iloc[:-1]
        v = a * (b - interest_rate.iloc[:-1])
        w = u - v
        x = (1 / (sigma * np.sqrt(interest_rate.iloc[1:]) * (2 * np.pi) ** 0.5))\
            * np.exp((- w ** 2) / (2 * (sigma * np.sqrt(interest_rate.iloc[1:])) ** 2))
        return -1 * np.sum(np.log(x))

    def optimize_negative_likelihood(self, interest_rate: pd.Series):
        mle_result = optimize.minimize(
            self.__calculate_negative_likelihood,
            [self.a, self.b, self.sigma],
            args=(interest_rate,)
        )
        return mle_result

    def make_interest_rate_simulations(self, r0, n_simulations, t_steps):
        r = r0 * np.ones((n_simulations, 1))
        for _ in range(t_steps):
            wiener_multiplier = stats.norm().rvs(size=(n_simulations, 1))
            r_previous = r[:, -1].reshape(-1, 1)
            r = np.hstack(
                (
                    r,
                    r_previous + self.a * (self.b - r_previous) + self.sigma * np.sqrt(np.abs(r_previous)) * wiener_multiplier
                )
            )
        last_t_simulation = r[:, -1]
        return r, np.mean(last_t_simulation), np.std(last_t_simulation) / np.sqrt(n_simulations)