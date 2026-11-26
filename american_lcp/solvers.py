"""Linear-complementarity solvers for min(M V - F, V - g) = 0.

Two independent solvers are provided:

* ``policy_iteration`` -- an active-set method that solves the reduced
  tridiagonal system exactly at each policy.
* ``psor`` -- projected successive over-relaxation, a fixed-point sweep used
  as an independent benchmark of the discrete LCP solution.

The complementarity residual reported by both is
``|| min(M V - F, V - g) ||_inf``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla


def _residual(M, V, F, g):
    return np.minimum(M @ V - F, V - g)


def policy_iteration(M, F, g, x0, tol=1e-10, maxit=200):
    """Policy iteration for min(M V - F, V - g) = 0.

    Nodes are partitioned into an active set (V_i = g_i) and a continuation
    set.  Fixing the active set reduces the system to a tridiagonal solve.
    Active rows whose residual is negative are released; rows whose
    unconstrained value falls below the obstacle are re-activated.

    Returns ``(V, iterations, residual)``.
    """
    n = len(g)
    V = np.maximum(x0, g)
    active = np.where(V <= g + 1e-14)[0]

    for it in range(1, maxit + 1):
        inactive = np.setdiff1d(np.arange(n), active)

        V = g.copy()
        if inactive.size:
            B = M[inactive][:, inactive].tocsc()
            c = F[inactive] - (M[inactive][:, active] @ g[active])
            V[inactive] = spla.spsolve(B, c)
        V = np.maximum(V, g)

        r = _residual(M, V, F, g)
        active = np.where(V <= g + 1e-14)[0]
        release = active[r[active] < -tol]
        res_norm = np.linalg.norm(r, np.inf)

        if release.size == 0:
            return V, it, res_norm
        active = np.setdiff1d(active, release)

    return V, maxit, np.linalg.norm(_residual(M, V, F, g), np.inf)


def psor(M, F, g, x0, omega=1.2, tol=1e-10, max_sweeps=20000,
         check_every=100):
    """Projected SOR for min(M V - F, V - g) = 0.

    ``M`` must be tridiagonal.  A full sweep visits the nodes left to right,
    using the newest neighbour values (Gauss--Seidel ordering), applies the
    relaxation ``omega``, and projects onto ``V >= g``.  The complementarity
    residual is evaluated every ``check_every`` sweeps.

    Returns ``(V, sweeps, residual)``.
    """
    n = len(g)
    d = M.diagonal(0).astype(float)
    low = M.diagonal(-1).astype(float)   # low[j] = M[j+1, j]
    up = M.diagonal(1).astype(float)     # up[j] = M[j, j+1]

    V = np.maximum(x0, g).astype(float)

    for sweep in range(1, max_sweeps + 1):
        for i in range(n):
            row = d[i] * V[i]
            if i > 0:
                row += low[i - 1] * V[i - 1]
            if i < n - 1:
                row += up[i] * V[i + 1]
            V[i] = max(g[i], V[i] - omega * (row - F[i]) / d[i])

        if sweep % check_every == 0:
            res = np.linalg.norm(_residual(M, V, F, g), np.inf)
            if res <= tol:
                return V, sweep, res

    res = np.linalg.norm(_residual(M, V, F, g), np.inf)
    return V, max_sweeps, res
