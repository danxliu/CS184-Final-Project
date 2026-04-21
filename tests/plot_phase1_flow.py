#!/usr/bin/env python3
"""Plot energy/gradient/step trajectories and mesh evolution snapshots
from demo_phase1_flow."""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def load_obj(path: Path):
    V, F = [], []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                V.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                F.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    return np.asarray(V), np.asarray(F, dtype=int)


def draw_silhouette(ax, V, F, axes_uv=(0, 2), title=""):
    u, v = axes_uv
    # triangles as 2D projected polys; sort by mean of orthogonal axis for depth
    w = 3 - u - v
    depth = V[F][:, :, w].mean(axis=1)
    order = np.argsort(depth)
    tris_2d = V[F][order][:, :, [u, v]]
    shade = (depth[order] - depth.min()) / (np.ptp(depth) + 1e-12)
    colors = plt.cm.Greys(0.25 + 0.55 * shade)
    pc = PolyCollection(tris_2d, facecolors=colors, edgecolors=(0, 0, 0, 0.25),
                        linewidths=0.25)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/phase1_flow")
    csv_path = out_dir / "energy.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        sys.exit(1)

    it, E, gnorm, tau = [], [], [], []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            it.append(int(row["iter"]))
            E.append(float(row["energy"]))
            gnorm.append(float(row["grad_norm"]))
            tau.append(float(row["step_size"]))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.3)

    ax_E = fig.add_subplot(gs[0, 0])
    ax_E.semilogy(it, E, "b-o", markersize=3)
    ax_E.set_xlabel("iteration")
    ax_E.set_ylabel(r"TPE energy $\hat\Phi$")
    ax_E.set_title("energy vs. iter")
    ax_E.grid(True, which="both", alpha=0.3)

    ax_g = fig.add_subplot(gs[0, 1])
    ax_g.semilogy(it, gnorm, "r-o", markersize=3)
    ax_g.set_xlabel("iteration")
    ax_g.set_ylabel(r"$\|\nabla\hat\Phi\|$ (scale-proj.)")
    ax_g.set_title("gradient norm")
    ax_g.grid(True, which="both", alpha=0.3)

    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.semilogy(it, tau, "g-o", markersize=3)
    ax_t.set_xlabel("iteration")
    ax_t.set_ylabel(r"$\tau$ (Armijo step)")
    ax_t.set_title(r"step size --- shrinks as L$^2$ stalls")
    ax_t.grid(True, which="both", alpha=0.3)

    ax_txt = fig.add_subplot(gs[0, 3])
    ax_txt.axis("off")
    summary = (
        f"iterations: {len(it)}\n"
        f"$E_0$ = {E[0]:.3e}\n"
        f"$E_f$ = {E[-1]:.3e}\n"
        f"ratio $E_0/E_f$ = {E[0]/E[-1]:.2f}$\\times$\n"
        f"final $\\|g\\|$ = {gnorm[-1]:.2e}\n"
        f"final $\\tau$ = {tau[-1]:.2e}\n\n"
        r"L$^2$ reaches a floor"
        "\nwell above the round-sphere\n"
        "minimum --- the Phase 2\n"
        r"H$^s$ preconditioner cuts"
        "\nthrough this stall."
    )
    ax_txt.text(0.02, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f4f4",
                          edgecolor="#bbb"))

    pick_iters = [0, len(it) // 4, len(it) // 2, 3 * len(it) // 4, len(it) - 1]
    for i, k in enumerate(pick_iters[:4]):
        obj = out_dir / f"frame_{k:04d}.obj"
        if not obj.exists():
            continue
        V, F = load_obj(obj)
        ax = fig.add_subplot(gs[1, i])
        draw_silhouette(ax, V, F, axes_uv=(0, 2),
                        title=f"iter {k} --- XZ projection\n"
                              f"E = {E[k]:.3e}")

    _, F0 = load_obj(out_dir / "frame_0000.obj")
    fig.suptitle(
        rf"Phase 1.8 --- L$^2$ gradient flow on ellipsoid, "
        rf"{len(F0)} faces, $\alpha = 6$",
        fontsize=12,
    )
    out_png = out_dir / "energy.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"  initial E = {E[0]:.4e}, final E = {E[-1]:.4e}, ratio = {E[0]/E[-1]:.2f}x")
    print(f"  final |g| = {gnorm[-1]:.3e}  (nonzero means L^2 is stalled, not converged)")

if __name__ == "__main__":
    main()
