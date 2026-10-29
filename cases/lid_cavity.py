"""Lid-driven cavity, unit square, validated against Ghia et al. (1982).

    python cases/lid_cavity.py --re 400 --steps 700000
"""
import argparse
import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ibflow import init_domain, lid_driven_cavity, rk2_step, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--re", type=float, default=400.0, help="Reynolds number U_lid L / nu")
    ap.add_argument("--beta", type=float, default=500.0, help="artificial compressibility (c^2)")
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=700_000)
    ap.add_argument("--n-fine", type=int, default=120, help="faces in the lower half (refined near the wall)")
    ap.add_argument("--n-coarse", type=int, default=60, help="faces in the upper half")
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--out", default="results/lid_cavity.npz")
    a = ap.parse_args(argv)

    L, U_lid, rho = 1.0, 1.0, 1.0
    nu = U_lid * L / a.re
    # Same grid as the original study: finer in the lower half of the cavity.
    faces = np.concatenate([np.linspace(0, L / 2, a.n_fine), np.linspace(L / 2, L, a.n_coarse)[1:]])
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(faces, faces, U_init=0.0, P_init=0.0)  # fluid starts at rest

    step = partial(rk2_step, lid_driven_cavity, U_ref=U_lid, P_ref=0.0, dt=a.dt,
                   rho=rho, nu=nu, beta=a.beta, dx=dx, dy=dy)
    meta = dict(case="lid_cavity", Re=a.re, beta=a.beta, dt=a.dt, nu=nu, rho=rho,
                U_ref=U_lid, L=L, x_c=x_c, y_c=y_c, dx=dx, dy=dy)
    print(f"Lid-driven cavity: Re={a.re:g}, grid {len(x_c)}x{len(y_c)} (incl. ghosts), "
          f"dt={a.dt:g}, steps={a.steps}")
    run(lambda P, U, V: step(P, U, V), P, U, V, a.steps, n_print=a.print_every,
        out=a.out, meta=meta)


if __name__ == "__main__":
    main()
