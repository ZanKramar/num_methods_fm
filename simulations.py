from random import random
from numpy.random import normal
from math import exp, log
from numpy import std, mean, sqrt


def binomial_simulation_crr(periods, stock, p, up, down):
    """
    Simulates movement of CRR model

    :param int periods: Number of periods
    :param int or float stock: Initial value of stock
    :param float p: Probability of market going up
    :param float up: Amount by which market goes up
    :param float down: Amount by which market goes down
    :return float: Final value of stock
    """
    i = 0
    while i < periods:
        if random() < p:
            stock = stock*up
        else:
            stock = stock*down
        i += 1
    return stock


def mc_black_scholes_interval(sigma, strike, option_function, n_iterations):
    """
    Monte Carlo simulation of premium of option

    :param float sigma: Sigma used in model
    :param int or float strike: Strike price
    :param function from options.py option_function: Option function
    :param int n_iterations: Number of iterations
    :return list: Premium of option with confidence interval
    """
    i = 0
    sample = []
    while i < n_iterations:
        sample.append(option_function(price=(exp(sigma * normal(0, 1))), strike=strike))
        i += 1
    mean1 = mean(sample)
    sd = std(sample)
    return [mean1 - 1.96*sd/sqrt(n_iterations), mean1, mean1 + 1.96*sd/sqrt(n_iterations)]


def mc_black_scholes(sigma, strike, option_function, n_iterations):
    """
    Monte Carlo simulation of premium of option

    :param float sigma: Sigma used in model
    :param int or float strike: Strike price
    :param function from options.py option_function: Option function
    :param int n_iterations: Number of iterations
    :return float: Premium of option
    """
    i = 0
    count = 0
    while i < n_iterations:
        count += option_function(price=(exp(sigma * normal(0, 1))), strike=strike)
        i += 1
    return count/n_iterations


def parity(put, sigma, strike):
    """
    Uses parity to calculate call price of option

    :param int or float put: Price of put option
    :param float sigma: Sigma used in model
    :param int or float strike: Strike price
    :return float: Premium of call option
    """
    return put + exp((sigma**2)/2)-strike


def mc_importance(sigma, strike, option_function, n_iterations):
    """
    Simulates importance sampling in B-S model

    :param float sigma: Sigma used in model
    :param int or float strike: Strike price
    :param function from options.py option_function: Option function
    :param int n_iterations: Number of iterations
    :return float: Premium of option
    """

    i = 0
    count = 0
    samples = []
    mu = log(strike)/sigma
    while i < n_iterations:
        g = normal(0, 1)
        real = option_function(price=(exp(sigma*(g + mu))), strike=strike) * \
                 exp(-(mu*g) - 0.5*mu**2)
        samples.append(real)
        count += option_function(price=(exp(sigma*(g + mu))), strike=strike) * \
                 exp(-(mu*g) - 0.5*mu**2)
        i += 1
    return [mean(samples) - 1.96*std(samples)/sqrt(n_iterations),count/n_iterations, mean(samples) + 1.96*std(samples)/sqrt(n_iterations)]







