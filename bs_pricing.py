from math import log, sqrt, e
from scipy.stats import norm
from numpy.random import normal


def black_scholes_formula_call(stock, strike, sigma, rf, t):
    """
    Calculates premium of european call option by Black-Scholes formula

    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param float sigma: Standard deviation used in model
    :param float rf: Risk-free rate
    :param int or float t: Time of maturity
    :return float: Premium
    """
    d1 = (log(stock/strike) + (rf + sigma**2/2)*t)/(sigma*sqrt(t))
    d2 = d1 - sigma*t
    return stock*norm.cdf(d1) - strike*e**(-rf*t)*norm.cdf(d2)


def black_scholes_mc(stock, strike, option_function, sigma, rf, t, n_iter):
    """
    Calculates premium of european option by Black-Scholes formula with Monte Carlo method

    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param function from options.py option_function: Option function, describing payouts of option
    :param float sigma: Standard deviation used in model
    :param float rf: Risk-free rate
    :param int or float t: Time of maturity
    :param int n_iter: Number of iterations
    :return float: Premium of option
    """
    count = 0
    i = 0
    while i < n_iter:
        g = normal(0, 1) * sqrt(t)
        count += option_function(price=(stock*e**((rf-0.5*sigma**2)*t+sigma*g)), strike=strike)
        i = i+1
    return e**(-rf*t)*count/n_iter


def dynamic_portfolio(stock, strike, steps, sigma, rf, maturity):
    """
    Dynamic hedging portfolio in Black-Scholes model for call option

    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param int steps: Number of steps
    :param float sigma: Sigma used in model
    :param float rf: Risk-free rate
    :param int or float maturity: Maturity of option
    :return float: Last value of portfolio
    """
    t = 0
    s0 = 1
    v = [black_scholes_formula_call(stock=stock, strike=strike, sigma=sigma, rf=rf, t=maturity)]
    alpha = [norm.cdf((log(stock/strike) + (rf + sigma**2/2)*(maturity-t))/(sigma*sqrt(maturity-t)))]
    beta = [v[0] - s0*alpha[0]]
    step = maturity/steps
    i = 0
    while i < (steps-1):
        t += step
        g = normal(0, 1)
        stock = stock*e**((0.5*sigma**2)*step + sigma*g*sqrt(step))
        s0 = s0*e**(rf*step)
        v.append(alpha[i]*stock + beta[i]*e**(rf*step))
        alpha.append(norm.cdf((log(stock/strike) + (rf + sigma**2/2)*(maturity-t))/(sigma*sqrt(maturity-t))))
        beta.append(v[i+1] - stock*alpha[i+1])
        i += 1
    g = normal(0, 1)
    stock = stock*e**((0.5*sigma**2)*step + sigma*g*sqrt(step))
    s0 = s0*e**(rf*step)
    v.append(alpha[-1]*stock + beta[-1]*e**(rf*step))
    return v[-1]


def exchange_option(s01, s02, rf, sigma1, sigma2, ro, time, n_iter, lam=1):
    """
    Calculates price of exchange option

    :param int or float s01: Initial price of first stock
    :param int or float s02: Initial price of second stock
    :param float rf: Risk free rate
    :param float sigma1: Sigma of first stock
    :param float sigma2: Sigma of second stock
    :param float ro: Correlation of stocks
    :param int or float time: Time of maturity
    :param int n_iter: Number of iterations
    :param float lam: Parameter of option
    :return float: Price of option
    """
    counter = 0
    n = 0
    while n < n_iter:
        g1 = normal()
        g2 = normal()
        counter += max(0, s01*e**((rf - 0.5*sigma1**2)*time + sigma1*g1*sqrt(time)) -
                       lam*s02*e**((rf - 0.5*sigma2**2)*time + ro*sigma2*g1*sqrt(time) +
                                   sqrt(1-ro**2)*g2*sigma2*sqrt(time)))
        n += 1
    return e**(-rf*time)*counter/n_iter


