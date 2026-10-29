"""Developing laminar flow in a plane channel (L = 20 H).

The fully developed profile should approach u/U_mean = 1.5 (1 - (2y/H - 1)^2).

    python cases/channel.py --re 400 --steps 700000
"""
import argparse
import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ibflow import init_domain, channel, rk2_step, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--re", type=float, default=400.0, help="Reynolds number U_in H / nu")
    ap.add_argument("--beta", type=float, default=500.0)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=700_000)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--length-ratio", type=float, default=20.0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--out", default="results/channel.npz")
    a = ap.parse_args(argv)

    H, U_in, rho = a.height, 1.0, 1.0
    nu = U_in * H / a.re
    x = np.linspace(0, a.length_ratio * H, a.nx)
    y = np.linspace(0, H, a.ny)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, y, U_init=U_in, P_init=0.0)

    step = partial(rk2_step, channel, U_ref=U_in, P_ref=0.0, dt=a.dt,
                   rho=rho, nu=nu, beta=a.beta, dx=dx, dy=dy)
    meta = dict(case="channel", Re=a.re, beta=a.beta, dt=a.dt, nu=nu, rho=rho,
                U_ref=U_in, L=H, x_c=x_c, y_c=y_c, dx=dx, dy=dy)
    print(f"Channel: Re={a.re:g}, grid {len(x_c)}x{len(y_c)}, dt={a.dt:g}, steps={a.steps}")
    run(lambda P, U, V: step(P, U, V), P, U, V, a.steps, n_print=a.print_every,
        out=a.out, meta=meta)


if __name__ == "__main__":
    main()
