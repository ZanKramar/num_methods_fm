from math import comb, e, sqrt
from replicating_porfolio import create_matrix


def one_period_option_value(stock, option_function, up, down, rf, strike):
    """
    Calculates option premium in one period binomial model

    :param float stock: Value of stock at t=0
    :param function float -> float option_function: Function describing outcome of the option
    :param float up: Ratio for upward movement
    :param float down: Ratio for downward movement
    :param float rf: Risk free rate
    :param float strike: Strike of option
    :return: Premium of stock
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    price_up = stock * up
    price_down = stock * down
    op_up = option_function(price=price_up, strike=strike)
    op_down = option_function(price=price_down, strike=strike)
    q = ((1+rf) - down)/(up-down)
    premium = q * op_up + (1-q) * op_down
    return premium


def multiple_period_binomial_option_value(stock, option_function, up, down, rf, strike, periods):
    """
    Calculates option premium in multiple period binomial model

    :param float stock: Value of stock at t=0
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float up: Ratio for upward movement
    :param float down: Ratio for downward movement
    :param float rf: Risk free rate
    :param float strike: Strike of option
    :param int periods: Number of periods
    :return: Premium of stock
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    q = (((1 + rf) - down) / (up - down)) / (1+rf)
    premium = 0
    i = 0
    while i <= periods:
        premium += comb(periods, i)*(q**i)*((1-q)**(periods-i))\
                   * option_function(strike=strike, price=stock*(up**i)*(down**(periods-i)))
        i += 1
    return premium


def multiple_period_binomial_option_value_stable(stock, option_function, up, down, rf, strike, periods):
    """
    Another way to calculate option premium in multiple period binomial model

    :param float stock: Value of stock at t=0
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float up: Ratio for upward movement
    :param float down: Ratio for downward movement
    :param float rf: Risk free rate
    :param float strike: Strike of option
    :param int periods: Number of periods
    :return: Premium of stock
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    q = (((1 + rf) - down) / (up - down)) * (1/(1+rf))
    matrix = create_matrix(periods+1)

    for j in reversed(range(0, periods+1)):
        for i in reversed(range(0, j+1)):
            if j == periods:
                matrix[i][j] = option_function(price=(stock*(up**i)*(down**(periods+1-i))), strike=strike)
            else:
                matrix[i][j] = q * matrix[i][j+1] + (1-q)*matrix[i+1][j+1]
    return matrix[0][0]


def multiple_period_crr_option_value_stable(stock, option_function, up, down, rf, strike, periods):
    """
    Calculates option premium in multiple period CRR model

    :param float stock: Value of stock at t=0
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float up: Ratio for upward movement
    :param float down: Ratio for downward movement
    :param float rf: Risk free rate
    :param float strike: Strike of option
    :param int periods: Number of periods
    :return: Premium of stock
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    q = (((1 + rf) - down) / (up - down))
    qr = q * (1/(1+rf))
    qrc = (1-q) * (1/(1+rf))
    matrix = create_matrix(periods+1)

    for i in range(0, periods+1):
        for j in range(0, periods + 1 - i):
            if i == 0:
                # print(stock*(up**(periods+1-j))*(down**j))
                matrix[i][j] = option_function(price=(stock*(up**(periods-j))*(down**j)), strike=strike)
                # print(option_function(price=(stock*(up**(periods+1-j))*(down**j)), strike=strike))
            else:
                matrix[i][j] = qr * matrix[i-1][j] + qrc*matrix[i-1][j+1]
    return matrix[periods][0]


def multiple_period_binomial_option_value_fixed(periods, option_function, strike, rf):
    """
    Calculates premium in binomial model for option that has payouts according to option function

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param float or int strike: Strike price
    :param float rf: Risk free rate
    :return float: Premium of option
    """
    matrix = create_matrix(periods+1)
    for i in range(0, periods+1):
        for j in range(0, periods - i+1):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - 2*j), strike=strike)
            else:
                matrix[i][j] = 1/(1+rf) * (0.5*matrix[i-1][j] + 0.5*matrix[i-1][j+1])
    return matrix[periods][0]


def multiple_period_trinomial_option_value_fixed(periods, option_function, strike, lam, rf):
    """
    Calculates premium in trinomial model for option that has payouts according to option function

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param float or int strike: Strike price
    :param float lam: Lambda parameter
    :param float rf: Risk-free rate
    :return float: Premium of option
    """
    matrix = create_matrix(2*periods + 1)
    for i in range(0, periods+1):
        for j in range(0, 2*periods + 1 - 2*i):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - j), strike=strike)
            else:
                matrix[i][j] = (lam/2 * matrix[i-1][j] + (1-lam)*matrix[i-1][j+1] + lam/2 * matrix[i-1][j+2])/(1+rf)
    return matrix[periods][0]


def american_crr_discrete(periods, option_function, stock, strike, up, down, rf):
    """
    Calculates premium for american-type options in crr model

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param float up: Parameter, describing upward trend (i.e. by how much we move up if the market moves up)
    :param float down: Parameter, describing downward trend (i.e. by how much we move down if the market moves down)
    :param float rf: Risk-free rate
    :return float: Premium of option
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    q = (((1 + rf) - down) / (up - down))
    qr = q * (1/(1+rf))
    qrc = (1-q) * (1/(1+rf))
    matrix = create_matrix(periods+1)

    for i in range(0, periods+1):
        for j in range(0, periods + 1 - i):
            if i == 0:
                matrix[i][j] = option_function(price=(stock*(up**(periods-j))*(down**j)), strike=strike)
            else:
                matrix[i][j] = max(qr * matrix[i-1][j] + qrc*matrix[i-1][j+1],
                                   option_function(price=stock*(up**(periods+1-i-j))*(down**j)*(1+rf)**(-2),
                                                   strike=strike))
    return matrix[periods][0]


def american_binomial(periods, option_function, strike, rf):
    """
    Calculates premium for american-type options in binomial model

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param int or float strike: Strike price
    :param float rf: Risk-free rate
    :return float: Premium of option
    """
    matrix = create_matrix(periods + 1)
    for i in range(0, periods + 1):
        for j in range(0, periods - i + 1):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - 2 * j), strike=strike)
            else:
                matrix[i][j] = max(0.5/(1+rf) * matrix[i - 1][j] + 0.5/(1+rf) * matrix[i - 1][j + 1],
                                   option_function(price=(periods - 2*j - i), strike=strike))
    return matrix[periods][0]


def american_trinomial(periods, option_function, strike, lam, rf):
    """
    Calculates premium for american-type options in trinomial model

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param int or float strike: Strike price
    :param float lam: Lambda parameter used in trinomial model
    :param float rf: Risk-free rate
    :return float: Premium of option
    """
    matrix = create_matrix(2 * periods + 1)
    for i in range(0, periods + 1):
        for j in range(0, 2 * periods + 1 - 2 * i):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - j), strike=strike)
            else:
                matrix[i][j] = max(1/(1+rf)*(
                    lam / 2 * matrix[i - 1][j] + (1 - lam) * matrix[i - 1][j + 1] + lam / 2 * matrix[i - 1][j + 2]),
                    option_function(price=(periods - j - i), strike=strike))
    return matrix[periods][0]


def multiple_period_binomial_option_value_bs(periods, option_function, strike, rf):
    """
    Calculates option premium in multiple period binomial model with continuous discounting

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param int or float strike: Strike price
    :param float rf: Risk-free rate
    :return float : Premium of option
    """
    matrix = create_matrix(periods+1)
    for i in range(0, periods+1):
        for j in range(0, periods - i+1):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - 2*j), strike=strike)
            else:
                matrix[i][j] = e**(-rf) * (0.5*matrix[i-1][j] + 0.5*matrix[i-1][j+1])
    return matrix[periods][0]


def multiple_period_trinomial_option_value_bs(periods, option_function, strike, lam, rf):
    """
    Calculates option premium in multiple period trinomial model with continuous discounting

    :param int periods: Number of periods
    :param function from options.py option_function: Function describing payouts of option
    :param int or float strike: Strike price
    :param float lam: Lambda used in model
    :param float rf: Risk-free rate
    :return float : Premium of option
    """
    matrix = create_matrix(2*periods + 1)
    for i in range(0, periods+1):
        for j in range(0, 2*periods + 1 - 2*i):
            if i == 0:
                matrix[i][j] = option_function(price=(periods - j), strike=strike)
            else:
                matrix[i][j] = (lam/2 * matrix[i-1][j] + (1-lam)*matrix[i-1][j+1] + lam/2 * matrix[i-1][j+2])*e**(-rf)
    return matrix[periods][0]


def multiple_period_crr_option_value_bs(stock, option_function, up, down, rf, strike, periods):
    """
    Calculates option premium in continuous CRR model


    :param float stock: Value of stock at t=0
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float up: Ratio for upward movement
    :param float down: Ratio for downward movement
    :param float rf: Risk free rate
    :param float strike: Strike of option
    :param int periods: Number of periods
    :return float: Premium of option
    """
    if not down <= 1+rf <= up:
        raise ValueError("Wrong inputs for ratios and risk free rate")
    q = (((1 + rf) - down) / (up - down))
    qr = q * e**(-rf)
    qrc = (1-q) * e**(-rf)
    matrix = create_matrix(periods+1)

    for i in range(0, periods+1):
        for j in range(0, periods + 1 - i):
            if i == 0:
                # print(stock*(up**(periods+1-j))*(down**j))
                matrix[i][j] = option_function(price=(stock*(up**(periods-j))*(down**j)), strike=strike)
                # print(option_function(price=(stock*(up**(periods+1-j))*(down**j)), strike=strike))
            else:
                matrix[i][j] = qr * matrix[i-1][j] + qrc*matrix[i-1][j+1]
    return matrix[periods][0]


def american_crr_contd(maturity, steps, option_function, stock, strike, sigma, rf):
    """
    Calculates option premium in continuous CRR model for american-type options

    :param int or float maturity: Maturity of option
    :param int steps: Number of steps
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param int or float stock: Initial price of stock
    :param int or float strike: Strike price
    :param float sigma: Sigma used in model
    :param float rf: Risk-free rate
    :return float: Premium of option
    """
    step = maturity/steps
    up = e**(sigma*sqrt(step))
    down = e**(-sigma*sqrt(step))
    q = ((e**(rf*step)) - down) / (up - down)
    qr = q * e**(-rf*step)
    qrc = (1-q) * e**(-rf*step)
    matrix = create_matrix(steps+1)

    for i in range(0, steps+1):
        for j in range(0, steps + 1 - i):
            if i == 0:
                matrix[i][j] = option_function(price=(stock*(up**(steps-j))*(down**j)), strike=strike)
            else:
                matrix[i][j] = max(qr * matrix[i-1][j] + qrc*matrix[i-1][j+1],
                                   option_function(price=stock*(up**(steps+1-i-j))*(down**j),
                                                   strike=strike))
    return matrix[steps][0]


def kr_tree_european(stock, strike, maturity, steps, lam, option_function, rf, sigma):
    """
    Calculates premium of option in Kamrad-Ritchken tree

    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param int or float maturity: Maturity of option
    :param int steps: Number of steps
    :param float lam: Lambda used in model
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float rf: Risk-free rate
    :param float sigma: Sigma used in model
    :return float: Premium of option
    """
    step = maturity/steps
    u = e**(lam*sigma*sqrt(step))
    pu = 1/(2*lam**2) + ((rf - (sigma**2)/2)*sqrt(step))/(2*lam*sigma)
    pd = 1 / (2 * lam ** 2) - ((rf - (sigma ** 2) / 2) * sqrt(step)) / (2 * lam * sigma)
    pm = 1 - 1/(lam**2)
    matrix = create_matrix(2 * steps + 1)
    for i in range(0, steps + 1):
        for j in range(0, 2 * steps + 1 - 2 * i):
            if i == 0:
                matrix[i][j] = option_function(price=(stock*u**(steps-j)), strike=strike)
            else:
                matrix[i][j] = (pu*matrix[i - 1][j] + pm * matrix[i - 1][j + 1] + pd * matrix[i - 1][
                    j + 2]) * e**(-rf*step)
    return matrix[steps][0]


def kr_tree_american(stock, strike, maturity, steps, lam, option_function, rf, sigma):
    """
    Calculates premium of american-type option in Kamrad-Ritchken tree

    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param int or float maturity: Maturity of option
    :param int steps: Number of steps
    :param float lam: Lambda used in model
    :param function (float,float) -> float option_function: Function describing outcome of the option
    :param float rf: Risk-free rate
    :param float sigma: Sigma used in model
    :return float: Premium of option
    """
    step = maturity/steps
    u = e**(lam*sigma*sqrt(step))
    pu = 1/(2*lam**2) + ((rf - (sigma**2)/2)*sqrt(step))/(2*lam*sigma)
    pd = 1 / (2 * lam ** 2) - ((rf - (sigma ** 2) / 2) * sqrt(step)) / (2 * lam * sigma)
    pm = 1 - 1/(lam**2)
    matrix = create_matrix(2 * steps + 1)
    for i in range(0, steps + 1):
        for j in range(0, 2 * steps + 1 - 2 * i):
            if i == 0:
                matrix[i][j] = option_function(price=(stock*u**(steps-j)), strike=strike)
            else:
                matrix[i][j] = max((pu*matrix[i - 1][j] + pm * matrix[i - 1][j + 1] + pd * matrix[i - 1][
                    j + 2]) * e**(-rf*step), option_function(price=(stock*u**(steps-j-i)), strike=strike))
    return matrix[steps][0]


def helper(stock, strike, rf, sigma, time, x):
    """
    Function that calculates initial condition for explicit finite difference scheme

    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param float rf: Risk-free rate
    :param float sigma: Sigma used in model
    :param int or float time: Time of maturity of option
    :param int or float x: Specified x in initial condition
    :return float: Value of initial condition
    """
    value = max(0, strike-stock*e**((rf-0.5*sigma**2)*time + sigma*x)) * e**(-rf*time)
    return value


def explicit_finite_difference_scheme(stock, strike, rf, sigma, time, steps=100, interval=3):
    """
    Calculates price of european put option by explicit finite difference scheme

    :param int or float stock: Initial value of stock
    :param int or float strike: Strike price
    :param float rf: Risk-free rate
    :param float sigma: Sigma used in model
    :param int or float time: Time of maturity
    :param int steps: Number of steps
    :param int or float interval: Boundary of interval used in model
    :return float: Premium of option
    """
    delta_x = 2 * interval / steps
    delta_t = delta_x**2
    m = round(time/delta_t)
    lam = delta_t/delta_x**2
    result = [helper(stock=stock, strike=strike, rf=rf,
                     sigma=sigma, time=time, x=(-interval + delta_x * i)) for i in range(0, 2 * steps + 1)]
    val = []
    for j in range(0, m+1):
        for i in range(0, 2 * steps + 1):
            try:
                x1 = result[i-1]
            except IndexError:
                x1 = 0
            try:
                x3 = result[i+1]
            except IndexError:
                x3 = 0
            val.append(lam*0.5 * x1 + (1-lam)*result[i] + 0.5*lam*x3)
        result = val
        val = []
    return result


