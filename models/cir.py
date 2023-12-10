import numpy as np
import pandas as pd

from scipy import optimize, special, stats


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
    delta_t: float
        Date step between two points.

    Methods
    -------
    estimate_ols:
        Estimates a, b, sigma parameters by ordinary least squares method.
    optimize_negative_likelihood:
        Optimize likelihood to find optimal a, b, sigma parameters.
    make_interest_rate_simulations:
        Make simulations by the Euler-Maruyama approximation method.
    """

    def __init__(self, delta_t: float):
        self.a = None
        self.b = None
        self.sigma = None
        self.delta_t = delta_t

    def __str__(self):
        return f'CIR(a={self.a}, b={self.b}, sigma={self.sigma})'

    def estimate_ols(self, interest_rate: pd.Series):
        y = (interest_rate.iloc[1:].values - interest_rate.iloc[:-1].values) / np.sqrt(np.abs(interest_rate.iloc[:-1].values))
        x1 = self.delta_t / np.sqrt(np.abs(interest_rate.iloc[:-1]))
        x2 = self.delta_t * np.sqrt(np.abs(interest_rate.iloc[:-1]))
        X = np.hstack((x1.values.reshape(-1, 1), x2.values.reshape(-1, 1)))

        w = np.linalg.inv(X.T @ X) @ X.T @ y
        a = -1 * w[1]
        b = w[0] / a
        sigma = (1 / (len(y) * self.delta_t) ** 0.5) * np.sqrt(np.sum((y - (X @ w)) ** 2))
        self.a = a
        self.b = b
        self.sigma = sigma
        return

    def __calculate_negative_log_likelihood(self, parameters, interest_rate: pd.Series):
        # https://www.youtube.com/watch?v=0efIB2vzvL0&ab_channel=AnhH.Le
        a, b, sigma = parameters
        c = (2 * a) / (sigma ** 2 * (1 - np.exp(-a * self.delta_t)))
        u_t = c * interest_rate.iloc[:-1] * np.exp(-a * self.delta_t)
        v_t1 = c * interest_rate.iloc[1:]
        q = (2 * a * b) / sigma ** 2 - 1

        N = len(interest_rate)
        L = (N - 1) * np.log(c) + np.sum(
            -1 * u_t - v_t1 + (q / 2) * np.log(v_t1 / u_t) + np.log(special.ive(q, 2 * np.sqrt(u_t * v_t1))) + 2 * np.sqrt(u_t * v_t1)
        )
        return -1 * L

        # u = interest_rate.iloc[1:] - interest_rate.iloc[:-1]
        # v = a * (b - interest_rate.iloc[:-1])
        # w = u - v
        # x = (1 / (sigma * np.sqrt(interest_rate.iloc[1:]) * (2 * np.pi) ** 0.5))\
        #     * np.exp((- w ** 2) / (2 * (sigma * np.sqrt(interest_rate.iloc[1:])) ** 2))
        # return -1 * np.sum(np.log(x))

    def optimize_negative_likelihood(self, interest_rate: pd.Series):
        mle_result = optimize.minimize(
            self.__calculate_negative_log_likelihood,
            [self.a, self.b, self.sigma],
            args=(interest_rate,)
        )
        self.a, self.b, self.sigma = mle_result.x
        return mle_result

    def make_interest_rate_simulations(self, r0, n_simulations, t_steps):
        r = r0 * np.ones((n_simulations, 1))
        for _ in range(t_steps):
            wiener_multiplier = stats.norm().rvs(size=(n_simulations, 1))
            r_previous = r[:, -1].reshape(-1, 1)
            r = np.hstack(
                (
                    r,
                    r_previous + self.a * (self.b - r_previous) * self.delta_t + self.sigma * np.sqrt(np.abs(r_previous) * self.delta_t) * wiener_multiplier
                )
            )
        last_t_simulation = r[:, -1]
        return r, np.mean(last_t_simulation), np.std(last_t_simulation) / np.sqrt(n_simulations)
