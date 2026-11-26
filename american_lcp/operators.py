"""Fitted spatial operator, time-step matrices, and time marching.

The spatial operator A_h approximates -L_BS V with

    (A_h V)_i = -a_i delta^2_{B,i} V + r V_i,

where delta^2_{B,i} is the exponential-fitted second difference.  The
resulting tridiagonal matrix is a nonsingular M-matrix for every finite cell
Peclet number (positive diagonal, non-positive off-diagonals, strictly
diagonally dominant with row sum r > 0).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .free_boundary import extract_free_boundary
from .grid import bernoulli, coefficients, make_grid
from .solvers import policy_iteration, psor


def build_spatial_operator(S, r, q, sigma):
    """Assemble the fitted spatial operator A_h (CSR, size len(S)).

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Tridiagonal matrix with boundary rows (0 and N) left as zero rows;
        the caller overwrites them with Dirichlet identities.
    (lower, diag, upper) : tuple of ndarray
        The fitted tridiagonal coefficients, length len(S), for diagnostics.
    """
    a, b, pe = coefficients(S, r, q, sigma)
    h = S[1] - S[0]
    n = len(S)

    bp = bernoulli(pe)
    bm = bernoulli(-pe)

    lower = -a / h**2 * bp          # coefficient of V_{i-1}
    upper = -a / h**2 * bm          # coefficient of V_{i+1}
    diag = a / h**2 * (bp + bm) + r  # diagonal entry

    A = sp.diags(
        [lower[1:], diag, upper[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )
    return A, (lower, diag, upper)


def build_central_spatial_operator(S, r, q, sigma):
    """Assemble the central-difference spatial operator (diagnostic only).

    The drift is differenced centrally, giving the report's
        ell_i^CD = -a_i/h^2 + b_i/(2h),
        u_i^CD   = -a_i/h^2 - b_i/(2h),
    with diagonal 2a_i/h^2 + r.  For |Pe_i| > 2 the lower coefficient becomes
    positive and the matrix loses the M-matrix property, which is exactly the
    defect the exponential fitting removes.
    """
    a, b, pe = coefficients(S, r, q, sigma)
    h = S[1] - S[0]

    lower = -a / h**2 + b / (2.0 * h)
    upper = -a / h**2 - b / (2.0 * h)
    diag = 2.0 * a / h**2 + r

    A = sp.diags(
        [lower[1:], diag, upper[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )
    return A, (lower, diag, upper)


def apply_dirichlet(M, F, left, right):
    """Overwrite rows 0 and N of ``M`` with identity rows and pin ``F``.

    Returns a new CSR matrix and a copy of ``F``; the input is not mutated.
    """
    n = M.shape[0]
    M = M.tolil(copy=True)
    M[0, :] = 0.0
    M[0, 0] = 1.0
    M[n - 1, :] = 0.0
    M[n - 1, n - 1] = 1.0

    F = np.array(F, dtype=float, copy=True)
    F[0] = left
    F[n - 1] = right
    return M.tocsr(), F


def _march(S, A, g, V0, n_steps, dt, theta_schedule, left_rule, right_rule,
           solver, tol, maxit, omega, linear, monitor, on_step):
    """Shared time-marching loop for the LCP and linear solves.

    ``theta_schedule(n)`` returns the theta value for step n; ``left_rule``
    and ``right_rule`` return the Dirichlet data at the next time level.
    """
    n = len(S)
    V = np.array(V0, dtype=float, copy=True)
    history = []
    total_iters = 0
    last_res = 0.0

    for step in range(n_steps):
        tau = (step + 1) * dt
        th = float(theta_schedule(step))
        M = sp.identity(n, format="csr") + th * dt * A

        if th == 1.0:
            F = V.copy()
        else:
            F = V - (1.0 - th) * dt * (A @ V)

        left = left_rule(tau)
        right = right_rule(tau)
        M, F = apply_dirichlet(M, F, left, right)

        if linear:
            V = spla.spsolve(M.tocsc(), F)
            it = 1
            last_res = np.linalg.norm(np.minimum(M @ V - F, V - g), np.inf)
        elif solver == "policy":
            V, it, last_res = policy_iteration(
                M, F, g, V, tol=tol, maxit=maxit
            )
        elif solver == "psor":
            V, it, last_res = psor(
                M, F, g, V, omega=omega, tol=tol, max_sweeps=maxit
            )
        else:
            raise ValueError(f"unknown solver {solver!r}")

        total_iters += it

        if monitor == "all":
            record = True
        elif monitor is not None:
            record = tau in monitor
        else:
            record = False

        if record:
            B, lo, hi = extract_free_boundary(S, V, g)
            history.append(
                {
                    "tau": tau,
                    "B": B,
                    "bracket": (lo, hi),
                    "residual": last_res,
                    "policy_iters": it,
                }
            )
        if on_step is not None:
            on_step(step, tau, V)

    return V, history, total_iters, last_res


def solve_lcp(market, Smax, N, n_steps, scheme="CN", n_rannacher=2,
              theta=0.5, solver="policy", tol=1e-10, maxit=200,
              omega=1.2, monitor="all", operator="fitted"):
    """Solve the finite-maturity American put LCP to time T.

    ``scheme`` is ``"BE"`` (backward Euler), ``"CN"`` (Crank--Nicolson) or
    ``"THETA"`` (general theta method).  For CN the first ``n_rannacher``
    steps are replaced by backward Euler steps (Rannacher start-up).

    Returns ``(V, S, history, total_iters, final_residual)`` where ``V`` is
    the value at tau = T and ``history`` records the free boundary at each
    monitored time level.
    """
    S, h = make_grid(Smax, N)
    if operator == "fitted":
        A, _ = build_spatial_operator(S, market.r, market.q, market.sigma)
    elif operator == "central":
        A, _ = build_central_spatial_operator(S, market.r, market.q, market.sigma)
    else:
        raise ValueError(f"unknown operator {operator!r}")
    g = np.maximum(market.K - S, 0.0)
    dt = market.T / n_steps

    if scheme in ("CN", "THETA"):
        def theta_schedule(step):
            return 1.0 if step < n_rannacher else theta
    elif scheme == "BE":
        def theta_schedule(step):
            return 1.0
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    def left_rule(tau):
        return market.K

    def right_rule(tau):
        return 0.0

    V, history, total_iters, res = _march(
        S, A, g, g, n_steps, dt, theta_schedule, left_rule, right_rule,
        solver, tol, maxit, omega, linear=False, monitor=monitor,
        on_step=None,
    )
    return V, S, history, total_iters, res


def solve_european_linear(market, Smax, N, n_steps, scheme="CN",
                          n_rannacher=2, theta=0.5, operator="fitted"):
    """Solve the obstacle-free (European) problem with exact European data.

    The left boundary is V(0, tau) = K exp(-r tau) and the right boundary is
    V(Smax, tau) = 0.  This validates the fitted operator, boundary rows and
    time integrator against the closed-form European put, free of any free
    boundary interaction.
    """
    S, h = make_grid(Smax, N)
    if operator == "fitted":
        A, _ = build_spatial_operator(S, market.r, market.q, market.sigma)
    elif operator == "central":
        A, _ = build_central_spatial_operator(S, market.r, market.q, market.sigma)
    else:
        raise ValueError(f"unknown operator {operator!r}")
    g = np.maximum(market.K - S, 0.0)
    dt = market.T / n_steps

    if scheme == "TRBDF2":
        V = g.copy()
        for step in range(n_steps):
            tau_half = (step + 0.5) * dt
            tau_full = (step + 1.0) * dt

            M1 = sp.identity(N + 1, format="csr") + (dt / 4.0) * A
            F1 = V - (dt / 4.0) * (A @ V)
            M1, F1 = apply_dirichlet(
                M1, F1, market.K * np.exp(-market.r * tau_half), 0.0
            )
            Vstar = spla.spsolve(M1.tocsc(), F1)

            M2 = sp.identity(N + 1, format="csr") + (dt / 3.0) * A
            F2 = (4.0 / 3.0) * Vstar - (1.0 / 3.0) * V
            M2, F2 = apply_dirichlet(
                M2, F2, market.K * np.exp(-market.r * tau_full), 0.0
            )
            V = spla.spsolve(M2.tocsc(), F2)
        return V, S

    if scheme in ("CN", "THETA"):
        def theta_schedule(step):
            return 1.0 if step < n_rannacher else theta
    elif scheme == "BE":
        def theta_schedule(step):
            return 1.0
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    def left_rule(tau):
        return market.K * np.exp(-market.r * tau)

    def right_rule(tau):
        return 0.0

    V, _, _, _ = _march(
        S, A, g, g, n_steps, dt, theta_schedule, left_rule, right_rule,
        solver=None, tol=0.0, maxit=0, omega=0.0, linear=True,
        monitor=None, on_step=None,
    )
    return V, S


def solve_trbdf2(market, Smax, N, n_steps, solver="policy", tol=1e-10,
                 maxit=200, omega=1.2, monitor="all"):
    """TR-BDF2: second-order, L-stable one-step integrator for the LCP.

    Each step performs a trapezoidal half-step
        (I + dt/4 A) V* = (I - dt/4 A) V^n
    followed by a BDF2 full step
        (I + dt/3 A) V^{n+1} = (4/3) V* - (1/3) V^n .
    Both step matrices are M-matrices.  TR-BDF2 is L-stable, so it damps the
    high-frequency modes excited by the kinked payoff without any Rannacher
    start-up while retaining second-order accuracy in smooth regions.
    """
    S, h = make_grid(Smax, N)
    A, _ = build_spatial_operator(S, market.r, market.q, market.sigma)
    g = np.maximum(market.K - S, 0.0)
    dt = market.T / n_steps

    V = g.copy()
    history = []
    total_iters = 0
    last_res = 0.0

    for step in range(n_steps):
        tau = (step + 1) * dt

        M1 = sp.identity(N + 1, format="csr") + (dt / 4.0) * A
        F1 = V - (dt / 4.0) * (A @ V)
        M1, F1 = apply_dirichlet(M1, F1, market.K, 0.0)
        if solver == "policy":
            Vstar, it1, r1 = policy_iteration(M1, F1, g, V, tol=tol, maxit=maxit)
        else:
            Vstar, it1, r1 = psor(M1, F1, g, V, omega=omega, tol=tol, max_sweeps=maxit)

        M2 = sp.identity(N + 1, format="csr") + (dt / 3.0) * A
        F2 = (4.0 / 3.0) * Vstar - (1.0 / 3.0) * V
        M2, F2 = apply_dirichlet(M2, F2, market.K, 0.0)
        if solver == "policy":
            V, it2, r2 = policy_iteration(M2, F2, g, Vstar, tol=tol, maxit=maxit)
        else:
            V, it2, r2 = psor(M2, F2, g, Vstar, omega=omega, tol=tol, max_sweeps=maxit)

        total_iters += it1 + it2
        last_res = max(r1, r2)

        if monitor == "all":
            B, lo, hi = extract_free_boundary(S, V, g)
            history.append(
                {
                    "tau": tau,
                    "B": B,
                    "bracket": (lo, hi),
                    "residual": last_res,
                    "policy_iters": it1 + it2,
                }
            )

    return V, S, history, total_iters, last_res


def solve_penalty(market, Smax, N, n_steps, theta=0.5, penalty=1e6,
                  tol=1e-10, maxit=50):
    """Forsyth--Vetzal-style penalty formulation of the American LCP.

    Solves the penalised equation
        V_tau - L_BS V + P (g - V)_+ = 0
    with a theta-method discretisation and a lagged penalty coefficient
    P_i = penalty on rows where V_i < g_i, else zero.  The penalised system is
    a single linear solve per iteration:
        (I + theta dt A + dt P_diag) V = rhs + dt P_diag g .

    Returns ``(V, S, max_violation, total_penalty_iters)`` where
    ``max_violation = max(g - V, 0)`` measures how far below the obstacle the
    penalised solution lies (it scales like O(1/penalty)).
    """
    S, h = make_grid(Smax, N)
    A, _ = build_spatial_operator(S, market.r, market.q, market.sigma)
    g = np.maximum(market.K - S, 0.0)
    dt = market.T / n_steps

    V = g.copy()
    total_iters = 0

    for _ in range(n_steps):
        if theta == 1.0:
            rhs = V.copy()
        else:
            rhs = V - (1.0 - theta) * dt * (A @ V)

        guess = V.copy()
        for it in range(1, maxit + 1):
            P_diag = np.where(guess < g, penalty, 0.0)
            Mpen = sp.identity(N + 1, format="csr") + theta * dt * A
            Mpen = Mpen + sp.diags(dt * P_diag, 0, format="csr")
            Fpen = rhs + dt * P_diag * g
            Mpen, Fpen = apply_dirichlet(Mpen, Fpen, market.K, 0.0)
            new = spla.spsolve(Mpen.tocsc(), Fpen)
            total_iters += 1
            if np.max(np.abs(new - guess)) < tol:
                guess = new
                break
            guess = new
        V = guess

    violation = float(np.max(np.maximum(g - V, 0.0)))
    return V, S, violation, total_iters
