import numpy as np


def simple_returns(price_data):
    r = []
    n = len(price_data)

    for i in range(n - 1):
        s_i = float(str(price_data[i]).replace(',', "."))
        s_i_1 = float(str(price_data[i + 1]).replace(',', "."))
        r.append( (s_i_1 - s_i) / s_i )

    return r


def returns_per_asset(assets, df):
    return [simple_returns(df[a]) for a in assets]


def mean_return(r):
    return sum(r) / len(r)


def mean_returns(assets, df):
    return [mean_return(simple_returns(df[a])) for a in assets]


def volatility(r):
    n = len(r)
    mu = mean_return(r)
    s = 0.0
    for i in range(n):
        s += (r[i] - mu) ** 2

    return np.sqrt((1 / (n - 1)) * s)


def normal_pdf(x, mu, sigma):
    return (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-(1/2) * ((x - mu)**2 / sigma**2))


def covariance(r_i, r_j):
    n = len(r_i)
    mu_i, mu_j = mean_return(r_i), mean_return(r_j)
    s = 0.0
    for k in range(n):
        s += (r_i[k] - mu_i) * (r_j[k] - mu_j)

    return (1 / (n - 1)) * s


def correlation(r_i, r_j):
    return covariance(r_i, r_j) / (volatility(r_i) * volatility(r_j))


def covariance_matrix(r):
    n = len(r)
    return np.array([[covariance(r[i], r[j]) for j in range(n)] for i in range(n)])


# Scalars of the Markowitz mean-variance solution
def markowitz_scalars(mu, cov):
    n = len(cov)
    C_inv = np.linalg.inv(cov)
    e = np.ones(n)

    A = e.T @ C_inv @ e
    B = e.T @ C_inv @ mu
    C = mu.T @ C_inv @ mu
    D = A * C - (B**2)

    return A, B, C, D


# Minimum volatility for each target return on the frontier
def efficient_frontier(mu, cov, targets):
    A, B, C, D = markowitz_scalars(mu, cov)
    return np.sqrt(((A * targets**2) - (2*B * targets) + C) / D)


# Weights, return & volatility of the minimum-variance portfolio
def min_variance_portfolio(mu, cov):
    A, B, _, _ = markowitz_scalars(mu, cov)
    n = len(cov)
    C_inv = np.linalg.inv(cov)
    e = np.ones(n)

    w_m = (1/A) * C_inv @ e
    mu_m = B / A
    sigma_m = 1 / np.sqrt(A)

    return w_m, mu_m, sigma_m
