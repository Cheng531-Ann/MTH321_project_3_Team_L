"""Figures for the report (matplotlib Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_european(res, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    S = res["S"]
    ax.plot(S, res["V"], lw=2, label="numerical (obstacle-free)")
    ax.plot(S, res["exact"], ls="--", lw=2, label="closed-form European")
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$V$")
    ax.set_title("European benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spatial_refinement(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    h = np.array([r["h"] for r in rows])
    e0 = np.array([r["e0"] for r in rows])
    ax.loglog(h, e0, "o-", label="pointwise error at $S_0$")
    ax.loglog(h, e0[0] * (h / h[0]) ** 2, "--", label=r"$\mathcal{O}(h^2)$")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("error")
    ax.set_title("Spatial refinement (fixed $\\Delta\\tau$)")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_temporal_refinement(results, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scheme, rows in results.items():
        dt = np.array([r["dt"] for r in rows])
        e0 = np.array([r["e0"] for r in rows])
        ax.loglog(dt, e0, "o-", label=scheme)
    ax.loglog(
        [results["BE"][0]["dt"], results["BE"][-1]["dt"]],
        [results["BE"][0]["e0"],
         results["BE"][0]["e0"] * (results["BE"][-1]["dt"] / results["BE"][0]["dt"]) ** 1],
        "--", label=r"$\mathcal{O}(\Delta\tau)$",
    )
    ax.set_xlabel(r"$\Delta\tau$")
    ax.set_ylabel("error at $S_0$")
    ax.set_title("Temporal refinement (fixed $h$)")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_boundary(history, path):
    tau = np.array([h["tau"] for h in history])
    B = np.array([h["B"] for h in history])
    lo = np.array([h["bracket"][0] for h in history])
    hi = np.array([h["bracket"][1] for h in history])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(tau, B, "o-", label=r"$B(\tau)$ estimate")
    ax.fill_between(tau, lo, hi, alpha=0.2, label="bracket")
    ax.set_xlabel(r"$\tau$ (remaining maturity)")
    ax.set_ylabel(r"$B$")
    ax.set_title("Exercise boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_stress_cn(stress, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    S = stress["CN no start-up"]["S"]
    mask = (S >= 0.8 * 100) & (S <= 1.2 * 100)
    ax.plot(S[mask], stress["CN no start-up"]["V"][mask], label="CN no start-up")
    ax.plot(S[mask], stress["CN + Rannacher"]["V"][mask], label="CN + Rannacher")
    if "TRBDF2 (no start-up)" in stress:
        ax.plot(S[mask], stress["TRBDF2 (no start-up)"]["V"][mask],
                label="TR-BDF2")
    ax.axvline(100.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$V$")
    ax.set_title("Near-expiry ATM: start-up comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_value_profiles(S, V_by_tau, path):
    """V(S) at several remaining maturities."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tau, V in V_by_tau:
        ax.plot(S, V, label=fr"$\tau={tau:.2f}$")
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$V$")
    ax.set_title("Value profiles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_peclet(rows, threshold, path):
    """Maximum lower off-diagonal against the cell Peclet number.

    The central coefficient turns positive once Pe > 2 (threshold sigma =
    sqrt(r-q)); the fitted coefficient stays non-positive for every Pe.
    """
    pe = np.array([r["Pe_max"] for r in rows])
    cc = np.array([r["central_max_lower_offdiag"] for r in rows])
    ff = np.array([r["fitted_max_lower_offdiag"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(pe, cc, "o-", label="central differencing")
    ax.semilogx(pe, ff, "s-", label="exponential fitting")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.axvline(2.0, color="gray", ls="--", lw=1)
    ax.annotate(
        r"Pe = 2  ($\sigma=\sqrt{r-q}=$" + f"{threshold:.3f})",
        xy=(2.0, 0.0), xytext=(3.0, 0.01),
    )
    ax.set_xlabel(r"max cell P\'eclet number")
    ax.set_ylabel(r"max lower off-diagonal $\ell_i$")
    ax.set_title("Convection--diffusion: monotonicity of the operator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
