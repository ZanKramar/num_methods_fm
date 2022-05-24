import valuations as val
import options as o
import simulations as s
import replicating_porfolio as rp
import bs_pricing as bs
import exotic_options as eo


def main():
    s0 = 100
    k = 100
    sigma = 0.2
    r = 0.1
    t = 1
    print(bs.black_scholes_formula_call(stock=s0, strike=k, sigma=sigma, rf=r, t=t))
    print(bs.black_scholes_mc(stock=s0, strike=k, sigma=sigma, rf=r, t=t, option_function=o.put, n_iter=100000))
    print(val.multiple_period_binomial_option_value_bs(periods=10, option_function=o.call, strike=1, rf=0.02))
    print(val.multiple_period_trinomial_option_value_bs(periods=10, option_function=o.call, strike=1, rf=0.02, lam=0.5))
    n = 300
    matrix = val.explicit_finite_difference_scheme(stock=100, strike=100, rf=0.1, sigma=0.2, time=1, steps=n,
                                                   interval=20)
    print(matrix[int(n / 2)])


if __name__ == '__main__':
    print(eo.mc_barrier_option(stock=100, strike=90, sigma=0.2, time=1,
                               barrier=80, rf=0.1, rebate=0, n_iter=10000, steps=100))

