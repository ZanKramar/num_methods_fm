import numpy
from numpy.random import normal
from math import e, log, sqrt


def simulate_vasicek(a, b, time, sigma, r0, steps):
    """
    Simulates path of Vasicek's interest rate model

    :param float a: Parameter used in model
    :param float b: Parameter used in model
    :param int or float time: End time
    :param float sigma: Parameter used in model
    :param float r0: Initial interest rate
    :param int steps: Number of steps
    :return list: List of realizations of the model
    """
    simulations = []
    step = time/steps
    simulations.append(r0)
    i = 0
    while i < steps:
        tail = simulations[-1]
        sim = tail*e**(-a*step) + b*(1 - e**(-a*step)) + normal()*sigma*sqrt((1-e**(-a*step))/(2*a))
        simulations.append(sim)
        i += 1
    return simulations


def mc_vasicek(a, b, time, sigma, r0, steps, n_iter):
    """
    Calculates zero-coupon bond price

    :param float a: Parameter used in model
    :param float b: Parameter used in model
    :param int or float time: Time of maturity
    :param float sigma: Parameter used in model
    :param float r0: Parameter used in model
    :param int steps: Number of steps in simulating Vasicek's model
    :param int n_iter: Number of iterations of MC algorithm
    :return float: Price of zero-coupon bond
    """
    counter = 0
    n = 0
    while n < n_iter:
        counter += sum(simulate_vasicek(a=a, b=b, time=time, sigma=sigma, r0=r0, steps=steps))/steps
        n += 1
    return e**(-counter/n_iter)




