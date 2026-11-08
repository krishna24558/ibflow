"""Post-processing for saved runs.

    python plot.py cavity  results/lid_cavity.npz    # u centreline vs Ghia et al.
    python plot.py channel results/channel.npz       # outlet profile vs Poiseuille
    python plot.py field   results/cylinder.npz      # u contour of the last snapshot
    python plot.py forces  results/cylinder.npz      # Cd, Cl history
    python plot.py divergence results/lid_cavity.npz # where |div u| sits (EDAC residual)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GHIA = os.path.join(HERE, "validation", "ghia_1982_u_centerline.csv")


def _save(fig, src, suffix):
    out = os.path.splitext(src)[0] + f"_{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def cavity(path):
    d = np.load(path)
    U, x_c, y_c, re = d["U"][-1], d["x_c"], d["y_c"], float(d["Re"])
    # Interpolate u to x = 0.5 for every row of cells
    u_mid = np.array([np.interp(0.5, x_c, U[:, j]) for j in range(len(y_c))])
    inner = slice(1, -1)  # drop ghost cells
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(u_mid[inner], y_c[inner], "-", label="EDAC (this code)")
    col = f"Re{int(round(re))}"
    with open(GHIA) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    g = np.genfromtxt(rows, delimiter=",", names=True)
    if col in g.dtype.names:
        ax.plot(g[col], g["y"], "o", mfc="none", label="Ghia et al. (1982)")
    else:
        print(f"No Ghia data for Re={re:g}; plotting simulation only.")
    ax.set_xlabel("u / U_lid at x = 0.5")
    ax.set_ylabel("y / L")
    ax.set_title(f"Lid-driven cavity, Re = {re:g}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, "centerline")


def channel(path):
    d = np.load(path)
    U, y_c = d["U"][-1], d["y_c"]
    H, u_in = float(d["L"]), float(d["U_ref"])
    inner = slice(1, -1)
    u_out = U[-3, inner]
    y = y_c[inner]
    eta = y / H
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(u_out / u_in, eta, "-", label="EDAC, near outlet")
    ax.plot(6 * eta * (1 - eta), eta, "--", label="Poiseuille (analytical)")
    ax.set_xlabel("u / U_in")
    ax.set_ylabel("y / H")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path, "profile")


def field(path):
    d = np.load(path)
    U = d["U"][-1]
    xx, yy = np.meshgrid(d["x_c"], d["y_c"], indexing="ij")
    fig, ax = plt.subplots(figsize=(14, 5))
    cf = ax.contourf(xx, yy, U, levels=30, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label="u")
    if "lag_x" in d:
        ax.plot(d["lag_x"], d["lag_y"], "k.", ms=1.5)
    ax.set_aspect("equal")
    ax.set_title(f"{d['case']}  Re = {float(d['Re']):g}  (step {int(d['step'][-1])})")
    _save(fig, path, "field")


def forces(path, skip_frac=0.25):
    d = np.load(path)
    q = 0.5 * float(d["rho"]) * float(d["U_ref"]) ** 2 * float(d["L"])
    s, fx, fy = d["hist_step"], d["hist_fx"], d["hist_fy"]
    t = s * float(d["dt"])
    cd, cl = fx / q, fy / q
    start = int(len(s) * skip_frac)  # skip the impulsive start-up transient
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].plot(t[1:], cd[1:])
    axs[0].set_ylabel("C_D")
    axs[1].plot(t[1:], cl[1:])
    axs[1].set_ylabel("C_L")
    axs[1].set_xlabel("time")
    for ax in axs:
        ax.grid(alpha=0.3)
    if len(cd) > start + 1:
        axs[0].set_ylim(np.percentile(cd[start:], 1) * 0.8, np.percentile(cd[start:], 99) * 1.2)
        print(f"Mean C_D over last {100 * (1 - skip_frac):.0f}% of run: {np.mean(cd[start:]):.3f}")
    if len(cl) > 32:
        half = len(cl) // 2                                   # second half: shedding has settled
        c = cl[half:-1] - cl[half:-1].mean()                  # last sample is off the regular spacing
        up = t[half + 1:-1][(c[:-1] < 0) & (c[1:] >= 0)]      # one upward zero crossing per period
        if len(up) > 2:
            st = float(d["L"]) / float(d["U_ref"]) / np.diff(up).mean()
            print(f"Lift amplitude {(c.max() - c.min()) / 2:.3f}, Strouhal number St = f D / U = {st:.3f}")
    _save(fig, path, "forces")


def divergence(path):
    """Map of |div u| from the last snapshot, on a log colour scale. EDAC converges to
    div u = (nu/beta) lap p rather than 0, so this shows where that residual sits."""
    from ibflow import compute_divergence

    d = np.load(path)
    U, V = d["U"][-1], d["V"][-1]
    div = np.abs(compute_divergence(U, V, d["dx"], d["dy"]))
    inner = (slice(2, -2), slice(2, -2))          # compute_divergence fills cells 2..n-3
    div, x, y = div[inner], d["x_c"][2:-2], d["y_c"][2:-2]
    i, j = np.unravel_index(np.argmax(div), div.shape)
    away = div[8:-8, 8:-8].max()                  # more than ~8 cells from any wall
    lo = max(np.percentile(div, 1), div.max() * 1e-4)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    pc = ax.pcolormesh(x, y, div.T, shading="nearest", cmap="magma",
                       norm=matplotlib.colors.LogNorm(vmin=lo, vmax=div.max()))
    fig.colorbar(pc, ax=ax, label="|∇·u|")
    ax.plot(x[i], y[j], "o", mfc="none", mec="cyan", ms=14, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    re = float(d["Re"]) if "Re" in d else float("nan")
    beta = float(d["beta"]) if "beta" in d else float("nan")
    ax.set_title(f"Residual divergence, Re = {re:g}, β = {beta:g}\n"
                 f"max {div.max():.3f} at ({x[i]:.2f}, {y[j]:.2f}); away from walls ≤ {away:.4f}",
                 fontsize=10)
    print(f"max |div u| = {div.max():.4f} at x = {x[i]:.3f}, y = {y[j]:.3f}; "
          f"away from walls: {away:.4f}")
    _save(fig, path, "divergence")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["cavity", "channel", "field", "forces", "divergence"])
    ap.add_argument("path")
    a = ap.parse_args()
    {"cavity": cavity, "channel": channel, "field": field, "forces": forces,
     "divergence": divergence}[a.kind](a.path)
