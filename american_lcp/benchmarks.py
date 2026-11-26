"""Reference values: European closed form and CRR binomial American put."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def european_put(S, K, r, q, sigma, tau):
    """Black--Scholes European put value.

    ``S`` may be an array; ``tau`` is a non-negative remaining maturity.
    At ``tau == 0`` the payoff is returned.
    """
    S = np.asarray(S, dtype=float)
    tau = float(tau)
    if tau <= 0.0:
        return np.maximum(K - S, 0.0)

    sq = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / sq
    d2 = d1 - sq
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def binomial_american_put(S0, K, r, q, sigma, T, n):
    """Cox--Ross--Rubinstein binomial American put with ``n`` time steps."""
    if n < 1:
        raise ValueError("n must be positive")
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    S = S0 * d ** np.arange(n, -1, -1) * u ** np.arange(0, n + 1)
    V = np.maximum(K - S, 0.0)
    for j in range(n - 1, -1, -1):
        V = disc * (pu * V[1:] + (1.0 - pu) * V[:-1])
        S = S0 * d ** np.arange(j, -1, -1) * u ** np.arange(0, j + 1)
        V = np.maximum(V, K - S)
    return float(V[0])


def binomial_american_extrapolated(S0, K, r, q, sigma, T, n):
    """CRR value with a Richardson extrapolation step (order 1 -> 2)."""
    v1 = binomial_american_put(S0, K, r, q, sigma, T, n)
    v2 = binomial_american_put(S0, K, r, q, sigma, T, 2 * n)
    return 2.0 * v1 - v2
