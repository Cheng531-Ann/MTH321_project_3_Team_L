"""Project 3 code archive: finite-maturity American put as a parabolic LCP.

Modules
-------
grid         : market parameters, uniform grid, exponential Bernoulli fitting.
operators    : fitted spatial operator, BE/CN/TR-BDF2 marching, penalty form.
solvers      : policy iteration and projected SOR for the discrete LCP.
free_boundary: obstacle-slack interpolation of the exercise boundary.
benchmarks   : European closed form and CRR binomial American reference.
experiments  : verification, refinement, stability, fidelity, and cost studies.
plotting     : figures used by the report.
"""

from .grid import Market, bernoulli, make_grid, coefficients
from .operators import (
    build_spatial_operator,
    apply_dirichlet,
    solve_lcp,
    solve_european_linear,
    solve_trbdf2,
    solve_penalty,
)
from .solvers import policy_iteration, psor
from .free_boundary import extract_free_boundary
from .benchmarks import european_put, binomial_american_put

__all__ = [
    "Market",
    "bernoulli",
    "make_grid",
    "coefficients",
    "build_spatial_operator",
    "apply_dirichlet",
    "solve_lcp",
    "solve_european_linear",
    "solve_trbdf2",
    "solve_penalty",
    "policy_iteration",
    "psor",
    "extract_free_boundary",
    "european_put",
    "binomial_american_put",
]
