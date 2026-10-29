#!/usr/bin/env python3
"""
Generate individual panel PDFs for Exp 19 lambda-sweep figure.
Outputs: figure/fig_exp19_panel_{a,b,c}.pdf
"""
from release_paths import RESULTS, FIGURES, ORIGINAL

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def theory_guarantee(c):
    if c <= 0:
        return 1.0
    return (1.0 - np.exp(-c)) / c


def clipped_guarantee(c):
    return theory_guarantee(max(1.0, c))


def main():
    data_path = ORIGINAL
    out_dir = FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    lambda_values = data["config"]["lambda_values"]
    results = data["results"]

    # Compute summary stats per lambda
    rows = []
    for lam in lambda_values:
        cells = [r["per_lambda"].get(str(lam), {}) for r in results]
        cells = [c for c in cells if c]
        c_hats = [c["c_hat_traj"] for c in cells]
        guarantees = [clipped_guarantee(c["c_hat_traj"]) for c in cells]
        rows.append({
            "lambda": lam,
            "c_analytic": 2.0 * lam,
            "r_theory": theory_guarantee(2.0 * lam),
            "c_hat_traj_mean": np.mean(c_hats),
            "c_hat_traj_se": np.std(c_hats) / np.sqrt(len(c_hats)) if len(c_hats) > 1 else 0.0,
            "clipped_guarantee_mean": np.mean(guarantees),
        })

    lams = [r["lambda"] for r in rows]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'axes.titlesize': 10,
    })

    panel_w, panel_h = 3.2, 2.6

    # --- Panel (a): ĉ_traj vs 2λ ---
    fig, ax = plt.subplots(figsize=(panel_w, panel_h))
    c_theory = [r["c_analytic"] for r in rows]
    c_hat_means = [r["c_hat_traj_mean"] for r in rows]
    c_hat_ses = [r["c_hat_traj_se"] for r in rows]

    for lam in lambda_values:
        cells = [r["per_lambda"].get(str(lam), {}) for r in results]
        cells = [c for c in cells if c]
        c_hats_i = [c["c_hat_traj"] for c in cells]
        ax.scatter([lam] * len(c_hats_i), c_hats_i, alpha=0.15, s=8, color='steelblue', zorder=2)

    ax.errorbar(lams, c_hat_means, yerr=[1.96 * s for s in c_hat_ses],
                fmt='o-', color='darkblue', markersize=5, linewidth=1.5,
                capsize=3, zorder=3, label=r'$\hat{c}_g$ (mean $\pm$ 95% CI)')
    cert_lams = [x for x in lams if x <= 1.0]
    cert_curve = [2.0 * x for x in cert_lams]
    diag_lams = [x for x in lams if x >= 1.0]
    diag_curve = [2.0 * x for x in diag_lams]
    ax.plot(cert_lams, cert_curve, 'k--', linewidth=1.5, label=r'$2\lambda$ reference ($\lambda\leq 1$)')
    if max(lams) > 1.0:
        ax.plot(diag_lams, diag_curve, color='0.45', linestyle=':', linewidth=1.5,
                label=r'$2\lambda$ diagnostic')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Curvature')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max(lams) + 0.1)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_exp19_panel_a.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {out_dir / 'fig_exp19_panel_a.pdf'}")

    # --- Panel (b): clipped trajectory guarantee vs uniform-query reference ---
    fig, ax = plt.subplots(figsize=(panel_w, panel_h))
    clipped_guar = [r["clipped_guarantee_mean"] for r in rows]

    lam_fine_cert = np.linspace(0.05, min(1.0, max(lams)), 160)
    r_fine_cert = [theory_guarantee(2 * lam) for lam in lam_fine_cert]
    ax.plot(lam_fine_cert, r_fine_cert, 'k--', linewidth=1.5,
            label=r'uniform-query reference')
    if max(lams) > 1.0:
        lam_fine_diag = np.linspace(1.0, max(lams) + 0.1, 80)
        r_fine_diag = [theory_guarantee(2 * lam) for lam in lam_fine_diag]
        ax.plot(lam_fine_diag, r_fine_diag, color='0.45', linestyle=':', linewidth=1.5,
                label=r'diagnostic extension')
    ax.plot(lams, clipped_guar, 'o-', color='darkgreen', markersize=5, linewidth=1.5,
            label=r'clipped trajectory guarantee')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Guarantee')
    ax.legend(loc='upper right')
    ax.set_xlim(0, max(lams) + 0.1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_exp19_panel_b.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {out_dir / 'fig_exp19_panel_b.pdf'}")

    # --- Panel (c): ROUGE-2 vs λ ---
    fig, ax = plt.subplots(figsize=(panel_w, panel_h))
    r2_means = []
    r2_ses = []
    for lam in lambda_values:
        cells = [r["per_lambda"].get(str(lam), {}) for r in results]
        cells = [c for c in cells if c and "rouge" in c and c["rouge"]]
        r2s = [c["rouge"]["rouge2"] for c in cells]
        r2_means.append(np.mean(r2s) if r2s else float('nan'))
        r2_ses.append(np.std(r2s) / np.sqrt(len(r2s)) if len(r2s) > 1 else 0.0)

    topk_r2 = [r.get("topk_tfidf", {}).get("rouge", {}).get("rouge2")
                for r in results if r.get("topk_tfidf", {}).get("rouge")]
    topk_mean = np.mean(topk_r2) if topk_r2 else float('nan')
    topk_se = np.std(topk_r2) / np.sqrt(len(topk_r2)) if len(topk_r2) > 1 else 0.0

    ax.errorbar(lams, r2_means, yerr=[1.96 * s for s in r2_ses],
                fmt='o-', color='darkred', markersize=5, linewidth=1.5,
                capsize=3, label='Greedy (GCLin)')
    ax.axhline(topk_mean, color='gray', linestyle=':', linewidth=1.5,
               label=f'top-$k$ TF-IDF ({topk_mean:.3f})')
    ax.fill_between([0, max(lams) + 0.2],
                     topk_mean - 1.96 * topk_se, topk_mean + 1.96 * topk_se,
                     alpha=0.1, color='gray')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('ROUGE-2')
    ax.legend(loc='best')
    ax.set_xlim(0, max(lams) + 0.1)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_exp19_panel_c.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved {out_dir / 'fig_exp19_panel_c.pdf'}")


if __name__ == "__main__":
    main()
