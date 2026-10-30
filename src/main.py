from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from options import black_scholes_call, implied_volatility
from portfolio import (correlation, covariance_matrix, efficient_frontier, mean_return, mean_returns,
                       min_variance_portfolio, normal_pdf, returns_per_asset, simple_returns, volatility)

DATA = Path(__file__).resolve().parent.parent / "data"


class PortfolioAnalysis:
    """Mean-variance (Markowitz) analysis of daily sector prices."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.assets = prices.columns.tolist()[1:]

    def return_distributions(self):
        for a in self.assets:
            r = simple_returns(self.prices[a])
            mu, sigma = mean_return(r), volatility(r)
            print(f'{a}: mean return {mu:.5f}, volatility {sigma:.5f}')

            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            sns.histplot(r, color="skyblue", stat="density", label='empirical')
            plt.plot(x, normal_pdf(x, mu, sigma), color='r', label='fitted normal')
            plt.title(a)
            plt.legend()
            plt.show()

    def correlations(self):
        r = returns_per_asset(self.assets, self.prices)
        n = len(r)
        corr = np.array([[correlation(r[i], r[j]) for j in range(n)] for i in range(n)])
        print(pd.DataFrame(corr, index=self.assets, columns=self.assets).round(3))
        print(covariance_matrix(r))

    def frontier(self, assets=None):
        assets = assets or self.assets
        r = returns_per_asset(assets, self.prices)
        mu = np.array(mean_returns(assets, self.prices))
        cov = covariance_matrix(r)

        targets = np.linspace(-min(mu), max(mu), 50)
        plt.plot(efficient_frontier(mu, cov, targets), targets, label=f'Efficient frontier, n = {len(assets)}')
        for a, r_a in zip(assets, r):
            plt.plot(volatility(r_a), mean_return(r_a), 'o', label=a)
        return mu, cov

    def efficient_frontier(self):
        mu, cov = self.frontier()
        w, mu_m, sigma_m = min_variance_portfolio(mu, cov)
        print(f'Min-variance portfolio: w = {np.round(w, 3)}, return = {mu_m:.5f}, volatility = {sigma_m:.5f}')
        plt.plot(sigma_m, mu_m, 'k*', markersize=12, label='min variance')
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.legend()
        plt.show()

    def frontier_without_first_asset(self):
        self.frontier()
        self.frontier(self.assets[1:])
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.legend()
        plt.show()


class OptionPricing:
    """Black-Scholes call prices and the implied volatility of market quotes."""

    def __init__(self, calls: pd.DataFrame, spot: float):
        self.strikes = calls["Nvidia (Calls) Strike"]
        self.prices = calls["Price"]
        self.spot = spot

    @staticmethod
    def call_price_curves(s_0=100.0, r=0.03):
        sigmas_ts = [(0.1, 1/12), (0.3, 1/4), (0.5, 1/2)]
        ks = [0.6*s_0, 0.8*s_0, s_0, 1.2*s_0, 1.4*s_0]

        for sigma, _ in sigmas_ts:
            for _, t in sigmas_ts:
                plt.plot(ks, [black_scholes_call(s_0, k, r, t, sigma) for k in ks], label=f'T = {t:.3f}')

            plt.title(f'S_0 = {s_0}, r = {r*100}%, σ = {sigma*100}%')
            plt.xlabel('Strike (K)')
            plt.ylabel('Call price (C)')
            plt.legend()
            plt.show()

    def volatility_smile(self, r=0.03, t=3/365):
        iv = [implied_volatility(self.prices[i], self.spot, self.strikes[i], r, t) for i in range(len(self.strikes))]

        plt.plot(self.strikes, iv)
        plt.title("Implied volatility vs. strike")
        plt.xlabel("Strike")
        plt.ylabel("Implied volatility")
        plt.show()


if __name__ == "__main__":
    portfolio = PortfolioAnalysis(pd.read_csv(DATA / "sector_prices.csv", sep=";"))
    portfolio.return_distributions()
    portfolio.correlations()
    portfolio.efficient_frontier()
    portfolio.frontier_without_first_asset()

    options = OptionPricing(pd.read_csv(DATA / "nvidia_calls.csv", sep=";"), spot=188.32)
    options.call_price_curves()
    options.volatility_smile()
