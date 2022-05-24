from numpy.random import normal
from math import e, log, sqrt


def mc_barrier_option(stock, strike, sigma, time, barrier, rf, rebate, n_iter, steps):
    """
    Calculates premium of barrier option

    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param float sigma: Sigma used in model
    :param int or float time: Maturity
    :param int or float barrier: Barrier value
    :param float rf: Risk-free rate
    :param int or float rebate: Rebate value
    :param int n_iter: Number of iterations
    :param int steps: Number of steps
    :return float: Premium of option
    """
    real = 0
    i = 0
    save_stock = stock
    while i < n_iter:
        out = 0
        j = 0
        tau = 0
        step = time/steps
        stock = save_stock
        while j < steps:
            g = normal()*sqrt(step)
            stock = stock*e**((rf-0.5*sigma**2)*step+sigma*g)
            if stock <= barrier:
                out = 1
                tau = j*step
            j += 1
        if out:
            if rebate > 0:
                real += rebate * e**(-rf*tau)
            else:
                real += 0
        else:
            stock = stock * e ** ((rf - 0.5 * sigma ** 2) * step + normal(0, 1) * sigma * sqrt(step))
            real += max(0, stock - strike) * e**(-rf*time)
        i += 1
    return real/n_iter


def mc_asian_option(stock, strike, sigma, time, rf, n_iter, steps, option_function, floating=False):
    """
    Calculates premium for asian option

    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param float sigma: Sigma used in model
    :param int or float time: Maturity
    :param float rf: Risk-free rate
    :param int n_iter: Number of iterations
    :param int steps: Number of steps
    :param function from options.py option_function: Option function describing payouts
    :param bool floating: If True, evaluate floating asian option, fixed otherwise (default False)
    :return float: Premium of option
    """
    real = 0
    i = 0
    save_stock = stock
    while i < n_iter:
        j = 0
        step = time/steps
        save_stock_info = []
        while j < steps:
            g = normal() * sqrt(step)
            stock = stock * e ** ((rf - 0.5 * sigma ** 2) * step + sigma * g)
            save_stock_info.append(stock)
            j += 1
        integrated_mean = sum(save_stock_info)/len(save_stock_info)
        if floating:
            real += option_function(price=stock, strike=integrated_mean) * e**(-rf*time)
        else:
            real += option_function(price=integrated_mean, strike=strike) * e**(-rf*time)
        stock = save_stock
        i += 1
    return real/n_iter


def mc_lookback(stock, strike, sigma, time, rf, n_iter, steps, option_str, floating=False):
    """
    Calculates premium for lookback option

    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param float sigma: Sigma used in model
    :param int or float time: Maturity
    :param float rf: Risk-free rate
    :param int n_iter: Number of iterations
    :param int steps: Number of steps
    :param str option_str: String, either 'put' or 'call'
    :param bool floating: If True, evaluate floating lookback option, fixed otherwise (default False)
    :return float: Premium of option
    """
    real = 0
    i = 0
    save_stock = stock
    while i < n_iter:
        j = 0
        step = time / steps
        save_stock_info = []
        while j < steps:
            g = normal() * sqrt(step)
            stock = stock * e ** ((rf - 0.5 * sigma ** 2) * step + sigma * g)
            save_stock_info.append(stock)
            j += 1
        min_stock = min(save_stock_info)
        max_stock = max(save_stock_info)
        if floating:
            if option_str == "call":
                real += max(0, strike - min_stock) * e ** (-rf*time)
            elif option_str == "put":
                real += max(0, max_stock - stock) * e ** (-rf * time)
            else:
                raise AttributeError("Wrong name of option, choose between put and call")
        else:
            if option_str == "call":
                real += max(0, max_stock - strike) * e ** (-rf*time)
            elif option_str == "put":
                real += max(0, strike - min_stock) * e ** (-rf * time)
            else:
                raise AttributeError("Wrong name of option, choose between put and call")
        stock = save_stock
        i += 1
    return real / n_iter





