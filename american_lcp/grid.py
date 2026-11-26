"""Market parameters, the uniform spatial grid, and the fitted Bernoulli
function used by the Scharfetter--Gummel exponential fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    """Constant-coefficient Black--Scholes market data.

    Parameters
    ----------
    S0 : float
        Spot price used as the evaluation point for quantities of interest.
    K : float
        Strike price; all prices are reported in the same units as K.
    r : float
        Risk-free rate (per year, continuously compounded).
    q : float
        Continuous dividend yield (per year).  ``q = 0`` is the problem-pack
        case; the general form is retained for completeness.
    sigma : float
        Constant volatility (per sqrt(year)).
    T : float
        Maturity in years.
    """

    S0: float
    K: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    T: float = 1.0


def bernoulli(z):
    """Stable evaluation of B(z) = z / (e^z - 1), B(0) = 1.

    Positive arguments use ``z / expm1(z)``; negative arguments use the
    identity B(-z) = B(z) + z.  The result is positive for every real z.
    Accepts scalars or NumPy arrays and preserves the input shape.
    """
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)

    out = np.empty_like(z)
    pos = z > 0
    neg = z < 0
    zero = z == 0

    zp = z[pos]
    out[pos] = zp / np.expm1(zp)

    zn = -z[neg]  # zn > 0
    out[neg] = zn / np.expm1(zn) + zn

    out[zero] = 1.0
    return out[0] if scalar else out


def make_grid(Smax, N):
    """Return the uniform grid S_i = i*h for i = 0..N and the step h."""
    if N <= 0:
        raise ValueError("N must be a positive integer")
    h = Smax / N
    S = np.linspace(0.0, Smax, N + 1)
    return S, h


def coefficients(S, r, q, sigma):
    """Return the diffusion ``a``, drift ``b`` and cell Peclet number ``Pe``.

    a_i = 0.5 sigma^2 S_i^2,  b_i = (r - q) S_i,  Pe_i = b_i h / a_i.
    At S = 0 both a and b vanish; the Peclet number is undefined there and is
    set to zero (the node is a Dirichlet row and never enters a stencil).
    """
    h = S[1] - S[0]
    a = 0.5 * sigma**2 * S**2
    b = (r - q) * S
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = np.where(a != 0.0, b * h / a, 0.0)
    return a, b, pe
