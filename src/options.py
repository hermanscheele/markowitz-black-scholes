import math as m
from scipy.optimize import brentq


def black_scholes_call(s_0, k, r, t, sigma):

    def N(d):
        return 0.5 * (1 + m.erf(d / m.sqrt(2)))

    d_1 = (m.log(s_0 / k) + (r + (0.5 * sigma**2)) * t) / (sigma * m.sqrt(t))
    d_2 = d_1 - (sigma * m.sqrt(t))

    return (s_0 * N(d_1)) - (k * m.exp(-r * t) * N(d_2))


# Volatility that makes the Black-Scholes price match the market price
def implied_volatility(c_mkt, s_0, k, r, t, lo=1e-9, hi=5.0):
    f = lambda sigma: black_scholes_call(s_0, k, r, t, sigma) - c_mkt
    return brentq(f, lo, hi)
