from numpy.random import normal
from math import e, log, sqrt


def mc_barrier_option(stock, strike, sigma, time, barrier, rf, rebate, n_iter, steps):
    real = 0
    i = 0
    while i < n_iter:
        out = 0
        j = 0
        tau = 0
        step = time/steps
        while j < steps:
            stock = stock*e**((rf - 0.5*sigma**2)*step + normal(0, 1)*sigma*sqrt(step))
            if stock <= barrier:
                out = 1
                tau = j*step
        if out:
            if rebate > 0:
                real += rebate * e**(-rf*tau)
            else:
                real += 0
        else:
            stock = stock * e ** ((rf - 0.5 * sigma ** 2) * step + normal(0, 1) * sigma * sqrt(step))
            real += max(0, stock - strike)
    return real/n_iter







