"""Post-processing for saved runs.

    python plot.py cavity  results/lid_cavity.npz    # u centreline vs Ghia et al.
    python plot.py channel results/channel.npz       # outlet profile vs Poiseuille
    python plot.py field   results/cylinder.npz      # u contour of the last snapshot
    python plot.py forces  results/cylinder.npz      # Cd, Cl history
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
    _save(fig, path, "forces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["cavity", "channel", "field", "forces"])
    ap.add_argument("path")
    a = ap.parse_args()
    {"cavity": cavity, "channel": channel, "field": field, "forces": forces}[a.kind](a.path)
