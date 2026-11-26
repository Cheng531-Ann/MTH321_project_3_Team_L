"""Project 3 top-level driver.

Run from the repository root::

    python run_all.py

Regenerates every table (``output/tables``) and figure (``output/figures``)
used by the report, and prints a short summary to the console.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

from american_lcp import Market
from american_lcp import experiments as ex
from american_lcp import plotting as pl
from american_lcp.operators import solve_lcp


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")


def ensure_dirs():
    for d in (TAB, FIG):
        os.makedirs(d, exist_ok=True)


def write_csv(name, rows):
    path = os.path.join(TAB, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt(v, prec=6):
    if v is None:
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{prec}g}"


def main():
    ensure_dirs()
    t_start = time.perf_counter()
    print("Project 3: finite-maturity American put (parabolic LCP)")
    print("=" * 64)

    market = Market(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0)

    # 1. European benchmark ------------------------------------------------
    print("[1/7] European benchmark ...")
    eur = ex.european_benchmark(market, N=800, n_steps=400, scheme="CN")
    pl.plot_european(eur, os.path.join(FIG, "european.png"))
    print(f"      CN max error {eur['max_err']:.3e}, L2 {eur['l2_err']:.3e}")

    # 2. Separated spatial refinement --------------------------------------
    print("[2/7] Spatial refinement (fixed dt) ...")
    srows, sref_S, sref_V = ex.spatial_refinement(
        market, n_steps=1000, grids=(200, 400, 800, 1600),
        ref_N=3200, ref_steps=2000,
    )
    write_csv("spatial_refinement.csv", srows)
    pl.plot_spatial_refinement(srows, os.path.join(FIG, "spatial_refinement.png"))
    for r in srows:
        print(f"      N={r['N']:5d}  h={r['h']:.5f}  e0={r['e0']:.3e}  order={fmt(r['order'], 3)}")

    # 3. Separated temporal refinement -------------------------------------
    print("[3/7] Temporal refinement (fixed h) ...")
    trows, tref_S, tref_V = ex.temporal_refinement(
        market, N=1600, steps=(25, 50, 100, 200), ref_steps=2000,
    )
    write_csv("temporal_refinement_BE.csv", trows["BE"])
    write_csv("temporal_refinement_CN.csv", trows["CN"])
    pl.plot_temporal_refinement(trows, os.path.join(FIG, "temporal_refinement.png"))
    for scheme in ("BE", "CN"):
        for r in trows[scheme]:
            print(f"      {scheme:3s} M={r['M']:4d}  dt={r['dt']:.5f}  e0={r['e0']:.3e}  order={fmt(r['order'], 3)}")

    # 3b. European temporal refinement (formal-order control) ---------------
    print("[3b] European temporal refinement (no free boundary) ...")
    eur_trows = ex.temporal_refinement_european(
        market, N=6400, steps=(10, 20, 40, 80)
    )
    write_csv("temporal_refinement_european_BE.csv", eur_trows["BE"])
    write_csv("temporal_refinement_european_CN.csv", eur_trows["CN"])
    write_csv("temporal_refinement_european_TRBDF2.csv", eur_trows["TRBDF2"])
    pl.plot_temporal_refinement(
        eur_trows, os.path.join(FIG, "european_temporal_refinement.png")
    )
    for scheme in ("BE", "CN", "TRBDF2"):
        for r in eur_trows[scheme]:
            print(f"      {scheme:3s} M={r['M']:3d}  dt={r['dt']:.5f}  e0={r['e0']:.3e}  order={fmt(r['order'], 3)}")

    # 4. Stress tests -------------------------------------------------------
    print("[4/7] Stress tests ...")
    near = Market(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=0.05)
    stress = ex.stress_near_expiry_atm(near, N=2000, n_steps=20)
    pl.plot_stress_cn(stress, os.path.join(FIG, "stress_cn.png"))
    print(f"      near-expiry ATM: Rannacher val={stress['CN + Rannacher']['value_at_S0']:.6f}, "
          f"no-startup val={stress['CN no start-up']['value_at_S0']:.6f}, binomial={stress['binomial']:.6f}")
    print(f"      oscillation probe max|d2V| near K: Rannacher="
          f"{stress['CN + Rannacher']['max_abs_second_diff_near_K']:.4f}, "
          f"no-startup={stress['CN no start-up']['max_abs_second_diff_near_K']:.4f}, "
          f"TRBDF2={stress['TRBDF2 (no start-up)']['max_abs_second_diff_near_K']:.4f}")

    low = Market(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.1, T=1.0)
    lv = ex.stress_low_vol(low, N=2000, n_steps=400)
    print(f"      low-vol sigma=0.1: val={lv['value_at_S0']:.6f}, binomial={lv['binomial']:.6f}, B={lv['B']:.4f}")

    sweep = ex.stress_smax_sweep(market, N=2000, n_steps=400)
    write_csv("smax_sweep.csv", sweep)
    print("      Smax sweep:", "  ".join(f"Smax={r['Smax']:.0f}:{r['value']:.6f}" for r in sweep))

    pen = ex.penalty_vs_complementarity(market, N=800, n_steps=100, penalty=1e6)
    write_csv("penalty_vs_complementarity.csv", [{
        "max_diff": pen["max_diff"],
        "value_diff_at_S0": pen["value_diff_at_S0"],
        "max_violation": pen["max_violation"],
        "penalty_iters": pen["penalty_iters"],
    }])
    print(f"      penalty vs complementarity: max diff={pen['max_diff']:.3e}, "
          f"violation={pen['max_violation']:.3e}, penalty iters={pen['penalty_iters']}")

    pe_rows, pe_thr = ex.peclet_sweep(market, N=800, n_steps=100)
    write_csv("peclet_sweep.csv", pe_rows)
    pl.plot_peclet(pe_rows, pe_thr, os.path.join(FIG, "peclet_sweep.png"))
    print(f"      Peclet sweep (threshold sigma*={pe_thr:.4f}):")
    for r in pe_rows:
        print(f"        sigma={r['sigma']:.2f} Pe={r['Pe_max']:.1f} "
              f"central_l={r['central_max_lower_offdiag']:+.4f} "
              f"fitted_l={r['fitted_max_lower_offdiag']:+.4f}")

    # 5. Stability / falsification ------------------------------------------
    print("[5/7] Stability checks ...")
    stab = ex.stability_checks(market, N=400, n_steps=20, scheme="CN")
    write_csv("stability_checks.csv", [stab])
    print(f"      off-diag <= 0: {stab['off_diagonal_nonpositive']}, diag > 0: {stab['diagonal_positive']}")
    print(f"      CN RHS min entry: {stab['cn_rhs_min_entry']:.3f}, slack_min: {stab['slack_min']:.3e}")
    print(f"      bounds [{stab['value_min']:.6f}, {stab['value_max']:.6f}], residual {stab['residual']:.3e}")

    # 6. Fidelity diagnostics -----------------------------------------------
    print("[6/7] Fidelity diagnostics ...")
    fid = ex.fidelity_diagnostics(market, N=800, n_steps=200)
    write_csv("fidelity_diagnostics.csv", [{
        "slack_min": fid["slack_min"],
        "slack_max": fid["slack_max"],
        "monotone_in_S": fid["monotone_in_S"],
        "monotone_in_tau": fid["monotone_in_tau"],
        "boundary": fid["boundary"],
        "bracket_lo": fid["bracket"][0],
        "bracket_hi": fid["bracket"][1],
        "exercise_premium": fid["exercise_premium"],
        "residual": fid["residual"],
        "total_policy_iters": fid["total_policy_iters"],
    }])
    print(f"      monotone S: {fid['monotone_in_S']}, monotone tau: {fid['monotone_in_tau']}")
    print(f"      B={fid['boundary']:.4f} bracket={fid['bracket']}, premium={fid['exercise_premium']:.6f}")

    # boundary history figure
    V_fin, S_fin, hist, *_ = solve_lcp(market, Smax=400.0, N=800, n_steps=200, scheme="CN")
    pl.plot_boundary(hist, os.path.join(FIG, "boundary.png"))

    # value profiles figure
    profiles = []
    for tau in (0.05, 0.25, 0.5, 1.0):
        mt = Market(100.0, 100.0, 0.05, 0.0, 0.2, float(tau))
        Vt, St, *_ = solve_lcp(mt, Smax=400.0, N=400, n_steps=200, scheme="CN")
        profiles.append((tau, Vt))
    pl.plot_value_profiles(St, profiles, os.path.join(FIG, "value_profiles.png"))

    # 7. Cost--accuracy ------------------------------------------------------
    print("[7/7] Cost--accuracy comparison ...")
    cost, ref0 = ex.cost_comparison(market, N=1200, target=1e-3, ref_steps=2000)
    write_csv("cost_comparison.csv", [
        {"scheme": s, "M": r["M"], "err": r["err"], "wall": r["wall"],
         "policy_iters": r["policy_iters"], "residual": r["residual"]}
        for s, rows in cost.items() for r in rows
    ])
    for scheme in ("BE", "CN", "TRBDF2"):
        last = cost[scheme][-1]
        print(f"      {scheme}: M={last['M']} err={last['err']:.3e} wall={last['wall']:.3f}s "
              f"policy_iters={last['policy_iters']} (target 1e-3, ref={ref0:.6f})")

    write_latex_tables(srows, trows, eur_trows, cost)

    elapsed = time.perf_counter() - t_start
    print("=" * 64)
    print(f"Done in {elapsed:.1f}s. Output in {os.path.relpath(OUT, ROOT)}")
    return 0


def write_latex_tables(srows, trows, eur_trows, cost):
    """Emit booktabs fragments for the key report tables."""
    lines = []
    lines.append("% Auto-generated by run_all.py -- paste into project3.tex")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("$N$ & $h$ & $E(S_0)$ & order \\\\")
    lines.append("\\midrule")
    for r in srows:
        lines.append(f"{r['N']} & {r['h']:.5f} & {r['e0']:.3e} & {fmt(r['order'], 3)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Scheme & $M$ & $\\Delta\\tau$ & $E(S_0)$ & order \\\\")
    lines.append("\\midrule")
    for scheme in ("BE", "CN"):
        for r in trows[scheme]:
            lines.append(
                f"{scheme} & {r['M']} & {r['dt']:.5f} & {r['e0']:.3e} & {fmt(r['order'], 3)} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Scheme (European) & $M$ & $\\Delta\\tau$ & $E(S_0)$ & order \\\\")
    lines.append("\\midrule")
    for scheme in ("BE", "CN", "TRBDF2"):
        for r in eur_trows[scheme]:
            lines.append(
                f"{scheme} & {r['M']} & {r['dt']:.5f} & {r['e0']:.3e} & {fmt(r['order'], 3)} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Scheme & $M$ & error & wall (s) & policy iter. \\\\")
    lines.append("\\midrule")
    for scheme in ("BE", "CN", "TRBDF2"):
        last = cost[scheme][-1]
        lines.append(
            f"{scheme} & {last['M']} & {last['err']:.3e} & {last['wall']:.3f} & {last['policy_iters']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    path = os.path.join(TAB, "report_tables.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
