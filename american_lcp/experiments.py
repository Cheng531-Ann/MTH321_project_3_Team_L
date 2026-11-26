"""Verification and numerical-design experiments for the Project 3 report.

Every experiment returns plain dictionaries/lists so that ``run_all.py`` can
write the tables and figures in one place.
"""

from __future__ import annotations

import time

import numpy as np

from .benchmarks import binomial_american_put, european_put
from .free_boundary import extract_free_boundary
from .grid import Market, coefficients, make_grid
from .operators import (
    build_central_spatial_operator,
    build_spatial_operator,
    solve_european_linear,
    solve_lcp,
    solve_penalty,
    solve_trbdf2,
)


def value_at(S, V, S0):
    return float(np.interp(S0, S, V))


def _smooth_region_error(S, V, Sref, Vref, lo, hi, npts=257):
    """Maximum interpolation error over S in [lo, hi]."""
    grid = np.linspace(lo, hi, npts)
    return float(np.max(np.abs(np.interp(grid, S, V) - np.interp(grid, Sref, Vref))))


def make_reference(market, Smax=400.0, N=3200, n_steps=4000):
    """Fine-grid CN+Rannacher LCP reference used by convergence studies."""
    V, S, *_ = solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme="CN", n_rannacher=2, solver="policy",
    )
    return S, V


def _solve_american(market, Smax, N, n_steps, scheme, solver="policy"):
    """Dispatch the American solve to the requested integrator."""
    if scheme == "TRBDF2":
        return solve_trbdf2(
            market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            solver=solver,
        )
    return solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme=scheme, n_rannacher=2, solver=solver,
    )


def european_benchmark(market, Smax=400.0, N=800, n_steps=400, scheme="CN"):
    """Obstacle-free solve against the closed-form European put.

    Returns a dict with the grid, numerical and exact values, and the max
    error over interior nodes away from the lower boundary.
    """
    V, S = solve_european_linear(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme=scheme, n_rannacher=2,
    )
    exact = european_put(S, market.K, market.r, market.q, market.sigma, market.T)
    interior = (S > 1e-8) & (S < Smax)
    max_err = float(np.max(np.abs(V[interior] - exact[interior])))
    l2_err = float(np.sqrt(np.mean((V[interior] - exact[interior]) ** 2)))
    return {
        "scheme": scheme,
        "S": S,
        "V": V,
        "exact": exact,
        "max_err": max_err,
        "l2_err": l2_err,
    }


def peclet_sweep(market, Smax=400.0, N=800, n_steps=100,
                 sigmas=(0.05, 0.1, 0.2, 0.4)):
    """Central differencing versus exponential fitting across cell Peclet.

    The fitted operator is an M-matrix for every finite Peclet number, so its
    lower off-diagonal stays non-positive.  Central differencing makes that
    coefficient positive once the cell Peclet number exceeds 2, which is
    exactly the monotonicity defect the fitting removes.  The sweep reports
    the sign of the lower off-diagonal, the maximum cell Peclet number, and
    the (latent) effect on the American value.
    """
    S, h = make_grid(Smax, N)
    rows = []
    for sigma in sigmas:
        _, _, pe = coefficients(S, market.r, market.q, sigma)
        pe_max = float(np.max(np.abs(pe[1:-1])))

        _, (cl, _cd, _cu) = build_central_spatial_operator(
            S, market.r, market.q, sigma
        )
        _, (fl, _fd, _fu) = build_spatial_operator(
            S, market.r, market.q, sigma
        )
        central_max_lower = float(np.max(cl[1:-1]))
        fitted_max_lower = float(np.max(fl[1:-1]))

        m = Market(
            S0=market.S0, K=market.K, r=market.r, q=market.q,
            sigma=float(sigma), T=market.T,
        )
        Vf, Sf, *_ = solve_lcp(
            m, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            scheme="CN", operator="fitted",
        )
        Vc, Sc, *_ = solve_lcp(
            m, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            scheme="CN", operator="central",
        )
        binom = binomial_american_put(
            market.S0, market.K, market.r, market.q, sigma, market.T, 4000
        )

        rows.append({
            "sigma": float(sigma),
            "Pe_max": pe_max,
            "central_max_lower_offdiag": central_max_lower,
            "fitted_max_lower_offdiag": fitted_max_lower,
            "V_fitted": value_at(Sf, Vf, market.S0),
            "V_central": value_at(Sc, Vc, market.S0),
            "V_binomial": float(binom),
        })
    threshold = float(np.sqrt(max(market.r - market.q, 0.0)))
    return rows, threshold


def spatial_refinement(market, Smax=400.0, n_steps=2000, grids=(200, 400, 800, 1600),
                       ref_N=3200, ref_steps=4000):
    """Spatial convergence at fixed (small) dt, CN scheme."""
    Sref, Vref = make_reference(market, Smax, ref_N, ref_steps)
    rows = []
    prev = None
    for N in grids:
        V, S, *_ = solve_lcp(
            market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            scheme="CN", n_rannacher=2, solver="policy",
        )
        e0 = abs(value_at(S, V, market.S0) - value_at(Sref, Vref, market.S0))
        e_sm = _smooth_region_error(S, V, Sref, Vref, 1.1 * market.K, 2.5 * market.K)
        h = Smax / N
        order = None
        if prev is not None:
            order = float(np.log(prev["e0"] / e0) / np.log(prev["h"] / h))
        rows.append({"N": int(N), "h": h, "e0": e0, "e_smooth": e_sm, "order": order})
        prev = {"h": h, "e0": e0}
    return rows, Sref, Vref


def temporal_refinement(market, Smax=400.0, N=1600,
                        steps=(50, 100, 200, 400),
                        ref_steps=4000):
    """Temporal convergence at fixed (fine) h, for BE and CN."""
    Sref, Vref = make_reference(market, Smax, N, ref_steps)
    results = {}
    for scheme in ("BE", "CN"):
        rows = []
        prev = None
        for M in steps:
            V, S, *_ = _solve_american(market, Smax, N, M, scheme)
            e0 = abs(value_at(S, V, market.S0) - value_at(Sref, Vref, market.S0))
            dt = market.T / M
            order = None
            if prev is not None:
                order = float(np.log(prev["e0"] / e0) / np.log(prev["dt"] / dt))
            rows.append({"M": int(M), "dt": dt, "e0": e0, "order": order})
            prev = {"dt": dt, "e0": e0}
        results[scheme] = rows
    return results, Sref, Vref


def temporal_refinement_european(market, Smax=400.0, N=6400,
                                 steps=(10, 20, 40, 80)):
    """Temporal convergence on the obstacle-free problem at fixed fine h.

    This isolates the integrator order without any free boundary: backward
    Euler approaches first order and Crank--Nicolson second order in the
    smooth region.
    """
    exact0 = european_put(
        market.S0, market.K, market.r, market.q, market.sigma, market.T
    )
    results = {}
    for scheme in ("BE", "CN", "TRBDF2"):
        rows = []
        prev = None
        for M in steps:
            V, S = solve_european_linear(
                market, Smax=float(Smax), N=int(N), n_steps=int(M),
                scheme=scheme, n_rannacher=2,
            )
            e0 = abs(value_at(S, V, market.S0) - float(exact0))
            dt = market.T / M
            order = None
            if prev is not None:
                order = float(np.log(prev["e0"] / e0) / np.log(prev["dt"] / dt))
            rows.append({"M": int(M), "dt": dt, "e0": e0, "order": order})
            prev = {"dt": dt, "e0": e0}
        results[scheme] = rows
    return results


def stress_near_expiry_atm(market, Smax=400.0, N=2000, n_steps=400):
    """Near-expiry ATM test: CN with and without Rannacher start-up."""
    out = {}
    for label, n_start in (("CN + Rannacher", 2), ("CN no start-up", 0)):
        V, S, *_ = solve_lcp(
            market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            scheme="CN", n_rannacher=int(n_start), solver="policy",
        )
        # second difference as an oscillation probe near the strike
        dd = np.diff(V, 2)
        out[label] = {
            "S": S,
            "V": V,
            "value_at_S0": value_at(S, V, market.S0),
            "max_abs_second_diff_near_K": float(
                np.max(np.abs(dd[(S[1:-1] > 0.85 * market.K) & (S[1:-1] < 1.15 * market.K)]))
            ),
        }
    Vt, St, *_ = solve_trbdf2(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        solver="policy",
    )
    dd = np.diff(Vt, 2)
    out["TRBDF2 (no start-up)"] = {
        "S": St,
        "V": Vt,
        "value_at_S0": value_at(St, Vt, market.S0),
        "max_abs_second_diff_near_K": float(
            np.max(np.abs(dd[(St[1:-1] > 0.85 * market.K) & (St[1:-1] < 1.15 * market.K)]))
        ),
    }
    out["binomial"] = binomial_american_put(
        market.S0, market.K, market.r, market.q, market.sigma, market.T, 4000
    )
    return out


def penalty_vs_complementarity(market, Smax=400.0, N=800, n_steps=100,
                               penalty=1e6):
    """Compare the penalty formulation against the complementarity solution."""
    Vc, Sc, *_ = solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme="CN", n_rannacher=2, solver="policy",
    )
    Vp, Sp, violation, iters = solve_penalty(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        theta=0.5, penalty=float(penalty),
    )
    return {
        "S": Sc,
        "V_complementarity": Vc,
        "V_penalty": Vp,
        "max_diff": float(np.max(np.abs(Vc - Vp))),
        "value_diff_at_S0": float(
            abs(value_at(Sc, Vc, market.S0) - value_at(Sp, Vp, market.S0))
        ),
        "max_violation": violation,
        "penalty_iters": int(iters),
    }


def stress_low_vol(market, Smax=400.0, N=2000, n_steps=400):
    """Low-volatility stress: sharper free boundary, value vs binomial."""
    V, S, hist, *_ = solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme="CN", n_rannacher=2, solver="policy",
    )
    B, lo, hi = extract_free_boundary(S, V, np.maximum(market.K - S, 0.0))
    return {
        "S": S,
        "V": V,
        "value_at_S0": value_at(S, V, market.S0),
        "binomial": binomial_american_put(
            market.S0, market.K, market.r, market.q, market.sigma, market.T, 4000
        ),
        "B": B,
        "bracket": (lo, hi),
    }


def stress_smax_sweep(market, smax_values=(200.0, 300.0, 400.0, 600.0, 800.0),
                      N=2000, n_steps=400):
    """Artificial-boundary sensitivity: value at S0 versus Smax."""
    rows = []
    for Smax in smax_values:
        V, S, *_ = solve_lcp(
            market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
            scheme="CN", n_rannacher=2, solver="policy",
        )
        rows.append({"Smax": float(Smax), "value": value_at(S, V, market.S0)})
    return rows


def stability_checks(market, Smax=400.0, N=400, n_steps=10, theta=0.5,
                     scheme="CN", tol=1e-10):
    """Numerical checks that can contradict the stability claim."""
    S, h = make_grid(Smax, N)
    A, (lower, diag, upper) = build_spatial_operator(S, market.r, market.q, market.sigma)
    dt = market.T / n_steps

    M = np.eye(N + 1) + theta * dt * A.toarray()
    off_nonpos = bool(np.all(M[np.arange(1, N), np.arange(0, N - 1)] <= 0) and
                      np.all(M[np.arange(0, N - 1), np.arange(1, N)] <= 0))
    diag_pos = bool(np.all(np.diag(M) > 0))

    # CN right-hand-side operator: not nonnegative in general
    Rhs = np.eye(N + 1) - (1.0 - theta) * dt * A.toarray()
    rhs_min = float(np.min(Rhs))

    # Run a short solve and check obstacle/residual/bounds
    V, S, _, _, res = solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme=scheme, n_rannacher=2, solver="policy", tol=tol,
    )
    g = np.maximum(market.K - S, 0.0)
    slack_min = float(np.min(V - g))

    return {
        "off_diagonal_nonpositive": off_nonpos,
        "diagonal_positive": diag_pos,
        "cn_rhs_min_entry": rhs_min,
        "slack_min": slack_min,
        "value_min": float(np.min(V)),
        "value_max": float(np.max(V)),
        "bound_ok": bool(np.min(V) >= -tol and np.max(V) <= market.K + tol),
        "residual": float(res),
    }


def fidelity_diagnostics(market, Smax=400.0, N=800, n_steps=200):
    """Bounds, monotonicity, residual/slack, boundary and exercise premium."""
    V, S, hist, it, res = solve_lcp(
        market, Smax=float(Smax), N=int(N), n_steps=int(n_steps),
        scheme="CN", n_rannacher=2, solver="policy",
    )
    g = np.maximum(market.K - S, 0.0)
    slack = V - g

    # monotonicity in S: V non-increasing
    monoS = bool(np.all(np.diff(V) <= 1e-12))

    # monotonicity in tau: recompute at several tau and check at S0
    taus = np.linspace(0.05, 1.0, 8) * market.T
    dt0 = market.T / n_steps
    vals = []
    for tau in taus:
        n = max(1, int(round(tau / dt0)))
        mtau = Market(
            S0=market.S0, K=market.K, r=market.r, q=market.q,
            sigma=market.sigma, T=float(tau),
        )
        Vt, St, *_ = solve_lcp(
            mtau, Smax=float(Smax), N=int(N), n_steps=int(n),
            scheme="CN", n_rannacher=2, solver="policy",
        )
        vals.append((tau, value_at(St, Vt, market.S0)))
    monoTau = bool(all(vals[i][1] >= vals[i - 1][1] - 1e-10 for i in range(1, len(vals))))

    B, lo, hi = extract_free_boundary(S, V, g)
    premium = value_at(S, V, market.S0) - european_put(
        market.S0, market.K, market.r, market.q, market.sigma, market.T
    )

    return {
        "S": S,
        "V": V,
        "slack_min": float(np.min(slack)),
        "slack_max": float(np.max(slack)),
        "monotone_in_S": monoS,
        "monotone_in_tau": monoTau,
        "tau_values": vals,
        "boundary": B,
        "bracket": (lo, hi),
        "exercise_premium": float(premium),
        "residual": res,
        "total_policy_iters": it,
    }


def cost_comparison(market, Smax=400.0, N=1200, target=1e-3, ref_steps=2000):
    """Matched-accuracy cost: refine each scheme until error <= target."""
    Sref, Vref = make_reference(market, Smax, N, ref_steps)
    ref0 = value_at(Sref, Vref, market.S0)

    out = {}
    for scheme in ("BE", "CN", "TRBDF2"):
        rows = []
        for M in (25, 50, 100, 200, 400, 800, 1600, 3200):
            t0 = time.perf_counter()
            V, S, hist, it, res = _solve_american(market, Smax, N, M, scheme)
            wall = time.perf_counter() - t0
            err = abs(value_at(S, V, market.S0) - ref0)
            rows.append({
                "M": int(M),
                "err": err,
                "wall": wall,
                "policy_iters": int(it),
                "residual": res,
            })
            if err <= target:
                break
        out[scheme] = rows
    return out, ref0
