"""Generate the README figures in docs/.

Run from the repo root:  python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from options import black_scholes_call, implied_volatility  # noqa: E402
from portfolio import (correlation, covariance_matrix, efficient_frontier, mean_return,  # noqa: E402
                       mean_returns, min_variance_portfolio, normal_pdf, returns_per_asset, volatility)

DOCS = ROOT / "docs"
INK, MUTED, GRID, AXIS = "#1f2328", "#57606a", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": ["Helvetica", "DejaVu Sans"], "font.size": 11, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=2)


def save(fig, name):
    fig.savefig(DOCS / name)
    plt.close(fig)


def frontier(prices, assets):
    r = returns_per_asset(assets, prices)
    mu = np.array(mean_returns(assets, prices))
    cov = covariance_matrix(r)
    w, mu_m, sigma_m = min_variance_portfolio(mu, cov)

    fig, (ax, axw) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    targets = np.linspace(-0.0005, 0.003, 200)
    ax.plot(efficient_frontier(mu, cov, targets), targets, color=BLUE, lw=2, label="all 5 sectors")
    mu4, cov4 = mu[1:], covariance_matrix(r[1:])
    ax.plot(efficient_frontier(mu4, cov4, targets), targets, color=ORANGE, lw=2, ls="--", label="without Energy")
    for a, r_a in zip(assets, r):
        ax.plot(volatility(r_a), mean_return(r_a), "o", ms=8, color=INK, mec="white", mew=2)
        dx, dy = {"Energy": (2, 8), "SoftDrink": (-8, -16)}.get(a, (7, -4))
        ax.annotate(a, (volatility(r_a), mean_return(r_a)), xytext=(dx, dy), textcoords="offset points", color=INK)
    ax.plot(sigma_m, mu_m, "*", ms=16, color=BLUE, mec="white", mew=1.5)
    ax.annotate("min variance", (sigma_m, mu_m), xytext=(-12, 10), textcoords="offset points", ha="right", color=BLUE)
    ax.set_xlim(0, 0.021)
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("daily volatility"); ax.set_ylabel("daily expected return")
    ax.set_title("Efficient frontier")
    ax.legend(loc="lower right")

    order = np.argsort(w)
    bars = axw.barh(np.array(assets)[order], w[order], color=BLUE, height=0.6)
    for b, v in zip(bars, w[order]):
        axw.text(v + (0.01 if v >= 0 else -0.01), b.get_y() + b.get_height() / 2, f"{v:.0%}",
                 va="center", ha="left" if v >= 0 else "right", color=INK)
    axw.axvline(0, color=AXIS, lw=1)
    axw.set_xlim(-0.12, 0.8)
    axw.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axw.grid(axis="y", visible=False); axw.tick_params(axis="y", length=0)
    axw.set_title("Min-variance weights")
    fig.tight_layout()
    save(fig, "frontier.png")


def distributions(prices, assets):
    fig, axes = plt.subplots(1, len(assets), figsize=(13, 2.8), sharey=True)
    for ax, a in zip(axes, assets):
        r = np.array(returns_per_asset([a], prices)[0])
        mu, sigma = mean_return(r), volatility(r)
        x = np.linspace(-0.06, 0.06, 300)
        ax.hist(r, bins=np.linspace(-0.06, 0.06, 31), density=True, color=BLUE, alpha=0.35, ec="white", lw=0.8)
        ax.plot(x, normal_pdf(x, mu, sigma), color=BLUE, lw=2)
        ax.set_title(a)
        ax.text(0.97, 0.95, f"σ = {sigma:.1%}", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks([-0.05, 0, 0.05]); ax.set_yticks([]); ax.grid(False)
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "returns.png")


def correlations(prices, assets):
    r = returns_per_asset(assets, prices)
    n = len(r)
    c = np.array([[correlation(r[i], r[j]) for j in range(n)] for i in range(n)])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [ORANGE, "#f0efea", BLUE])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(c):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if abs(v) > 0.7 else INK, fontsize=10)
    ax.set_xticks(range(n), assets, rotation=30, ha="right"); ax.set_yticks(range(n), assets)
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Return correlations")
    fig.tight_layout()
    save(fig, "correlation.png")


def call_prices():
    s_0, r = 100.0, 0.03
    ks = np.linspace(60, 140, 81)
    ts = [(1/12, "1 month"), (1/4, "3 months"), (1/2, "6 months")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, sigma in zip(axes, [0.1, 0.3, 0.5]):
        for (t, label), color in zip(ts, [BLUE, ORANGE, AQUA]):
            ax.plot(ks, [black_scholes_call(s_0, k, r, t, sigma) for k in ks], color=color, lw=2, label=label)
        ax.axvline(s_0, color=AXIS, lw=1, ls=":")
        ax.set_title(f"σ = {sigma:.0%}")
        ax.set_xlabel("strike K")
    axes[0].set_ylabel("call price")
    axes[0].legend(title="maturity", title_fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "black_scholes.png")


def smile(calls, spot=188.32, r=0.03, t=3/365):
    k = calls["Nvidia (Calls) Strike"].to_numpy()
    p = calls["Price"].to_numpy()
    iv = np.array([implied_volatility(c, spot, kk, r, t) for c, kk in zip(p, k)])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(k, iv, color=BLUE, lw=2, marker="o", ms=8, mec="white", mew=2)
    ax.axvline(spot, color=AXIS, lw=1, ls=":")
    ax.annotate(f"spot {spot}", (spot, iv.max()), xytext=(6, 0), textcoords="offset points", color=MUTED, va="top")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("strike"); ax.set_ylabel("implied volatility")
    ax.set_title("Nvidia volatility smile  (calls, 3 days to expiry)")
    fig.tight_layout()
    save(fig, "smile.png")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    prices = pd.read_csv(ROOT / "data" / "sector_prices.csv", sep=";")
    assets = prices.columns.tolist()[1:]
    frontier(prices, assets)
    distributions(prices, assets)
    correlations(prices, assets)
    call_prices()
    smile(pd.read_csv(ROOT / "data" / "nvidia_calls.csv", sep=";"))
