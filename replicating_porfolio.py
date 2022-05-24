from numpy import array
from numpy.random import random


def create_matrix(n):
    """
    Creates empty matrix to be used in replicator.
    :param int n: Size of the matrix
    :return: Matrix like element (list of lists)
    """
    if not isinstance(n, int):
        raise ValueError("n should be integer")
    null_list = [0]*n
    null_matrix = [null_list]*n
    return array(null_matrix, dtype="f")


def bernoulli(p):
    """
    Simulates Bernoulli random variable with probability p

    :param float p: Probability of 1 being the outcome
    :return int: 1 or 0
    """
    if random() < p:
        return 1
    return 0


def replicator_crr(stock, up, down, rf, option_function, periods, strike):
    """
    Simulates replicating portfolio in CRR model

    :param int stock: Value of stock
    :param float up: Ratio of upward movement
    :param float down: Ratio of downward movement
    :param float rf: Risk free rate
    :param function option_function: Function for option payouts
    :param int periods: Number of periods
    :param float strike: Strike price
    :return float: Value of portfolio at maturity
    """

    v = list()
    v = v.append(option_function(price=stock, strike=strike))
    alpha = [(option_function(price=(stock*up), strike=strike) - option_function(price=(stock*down), strike=strike)) /
             (stock*up - stock*down)]
    beta = [v[0] - alpha*stock]
    prob = ((1+rf)-down)/(up-down)
    i = 0
    while i < periods-1:
        ind = bernoulli(prob)
        if ind:
            stock = stock*up
        else:
            stock = stock*down
        v.append(alpha[i]*stock + beta[i]*(1+rf))
        alpha.append(
            (option_function(price=stock*up, strike=strike) - option_function(price=stock*down, strike=strike)) /
            (stock*up - stock*down)
        )
        beta.append(v[i+1] - stock*alpha[i+1])
        i += 1
    ind = bernoulli(prob)
    if ind:
        stock = stock * up
    else:
        stock = stock * down
    v.append(alpha[periods]*stock + beta[periods]*(1+rf))
    return v[-1]
