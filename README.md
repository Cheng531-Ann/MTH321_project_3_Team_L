# Project 3: Finite-Maturity American Put as a Parabolic LCP

Code archive for the Project 3 evidence chain (Weeks 10-13, Numerical PDEs).
The model is the finite-maturity American put written as the parabolic linear
complementarity problem

```
min( V - (K - S)^+,  dV/dtau - L_BS V ) = 0,   V(S, 0) = (K - S)^+,
L_BS V = 0.5 sigma^2 S^2 V_SS + (r - q) S V_S - r V,
V(0, tau) = K,   V(Smax, tau) = 0.
```

The exercise boundary `B(tau)` is an output, not an input.

## Requirements

* Python 3.13.15 (any recent 3.11+ should work)
* numpy 2.5.4
* scipy 1.18.1
* matplotlib 3.11.2

Install with:

```bash
python -m pip install numpy scipy matplotlib
```

## Reproduction

From the repository root, run:

```bash
python run_all.py
```

This regenerates every table (`output/tables`) and figure (`output/figures`)
used by the report and prints a summary. A `report_tables.tex` fragment with
the key booktabs tables is also produced for pasting into the report.

## Parameter sets and tolerances

Base market data: `S0 = K = 100`, `r = 0.05`, `q = 0`, `sigma = 0.2`,
`T = 1.0`, `Smax = 400`.

* LCP tolerance (policy iteration and PSOR): `1e-10` (complementarity
  residual in the infinity norm).
* Policy-iteration slack threshold for the active set: `1e-14`.
* PSOR relaxation parameter: `omega = 1.2` (residual checked every 100
  sweeps).
* Rannacher start-up: the first two Crank-Nicolson time levels use backward
  Euler steps of the same step size.
* TR-BDF2: two substeps per time level (trapezoid `dt/4`, BDF2 `dt/3`); it
  is second-order and L-stable, so no start-up is needed.
* Penalty method: `penalty = 1e6` with a lagged penalty coefficient, solved
  by repeated linear solves.

Stress cases: near-expiry ATM (`T = 0.05`, coarse `n_steps = 20` for the
no-start-up negative control), low volatility (`sigma = 0.1`), and an
`Smax` sweep over `{200, 300, 400, 600, 800}`.

## What is ours versus what comes from a library

Ours (derived from the PDE and implemented here):

* the exponential (Scharfetter-Gummel) fitted spatial operator and its
  M-matrix assembly;
* both time-stepping updates (backward Euler and Crank-Nicolson with
  Rannacher start-up) and the Dirichlet boundary closure;
* the policy-iteration active-set update and the projected-SOR sweep;
* the free-boundary extraction by obstacle-slack interpolation;
* the TR-BDF2 two-substep update and the Forsyth-Vetzal penalty formulation;
* every verification/refinement/stability/fidelity/cost experiment.

From SciPy/NumPy (named, with stated options):

* `scipy.sparse.linalg.spsolve` (SuperLU) for the tridiagonal reduced system
  inside each policy step;
* `scipy.stats.norm.cdf` for the European closed form;
* NumPy vectorisation of the updates derived above.

## Module map

```
american_lcp/
  grid.py          Market data, uniform grid, Bernoulli fitting
  operators.py     fitted spatial operator, time-step matrices, marching
  solvers.py       policy iteration, projected SOR
  free_boundary.py exercise-boundary extraction
  benchmarks.py    European closed form, CRR binomial American reference
  experiments.py   verification, refinement, stability, fidelity, cost
  plotting.py      report figures
run_all.py         one-command regeneration of tables and figures
```

## Notes on the measured orders

On the obstacle-free (European) problem, backward Euler is first order and
Crank-Nicolson and TR-BDF2 are second order in time. On the American problem
the free boundary reduces the observed Crank-Nicolson temporal order to
about 1.3 at the evaluation point, which is reported as a limitation rather
than hidden; TR-BDF2, being L-stable, keeps second-order accuracy without a
start-up and shows no spurious oscillation at the strike.

The cost comparison uses a fixed spatial grid and a fine same-grid reference,
so spatial error cancels and the reported error isolates the time-discretisation
error; the cost measure is wall-clock time, with the step count and total
policy iterations also reported.
