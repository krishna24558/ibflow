"""Flow past an immersed body (circular cylinder or symmetric NACA airfoil) using IBM.

    python cases/external_body.py --body cylinder --re 100 --steps 800000
    python cases/external_body.py --body naca0012 --re 100 --steps 800000

Drag and lift are reported as forces per unit span and saved with the snapshots;
``python plot.py forces`` turns them into Cd and Cl.
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ibflow import (init_domain, external_flow_grid, circle, naca_symmetric,  # noqa: E402
                   rk2_ibm_convective_step, run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=["cylinder", "naca0012"], default="cylinder")
    ap.add_argument("--re", type=float, default=100.0, help="Reynolds number U_in D / nu (D = diameter or chord)")
    ap.add_argument("--size", type=float, default=1.5, help="cylinder diameter or airfoil chord")
    ap.add_argument("--u-in", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=400.0)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--n-conc", type=int, default=80, help="points across the refined box")
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    D, rho = a.size, 1.0
    nu = a.u_in * D / a.re
    x, y = external_flow_grid(dim=D, conc_region=2.5, in_region=8.0, out_region=15.0,
                              wall_region=6.0, step_conc=a.n_conc, step_in=60,
                              step_out=200, step_wall=50)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, y, U_init=a.u_in, P_init=0.0)

    if a.body == "cylinder":
        lag_x, lag_y = circle(D / 2, dx.min())
    else:
        # Leading edge placed so the airfoil sits centred on the origin.
        lag_x, lag_y = naca_symmetric(D, dx.min(), 12, leading_edge=(-D / 2, 0.0))

    out = a.out or f"results/{a.body}.npz"
    step = partial(rk2_ibm_convective_step, U_in=a.u_in, P_out=0.0, dt=a.dt, rho=rho, nu=nu,
                   beta=a.beta, dx=dx, dy=dy, x_c=x_c, y_c=y_c, lag_x=lag_x, lag_y=lag_y)
    body = dict(x_c=x_c, y_c=y_c, lag_x=lag_x, lag_y=lag_y, dt=a.dt, rho=rho)
    meta = dict(case=a.body, Re=a.re, beta=a.beta, dt=a.dt, nu=nu, rho=rho, U_ref=a.u_in,
                L=D, x_c=x_c, y_c=y_c, dx=dx, dy=dy, lag_x=lag_x, lag_y=lag_y)
    print(f"{a.body}: Re={a.re:g}, D={D}, U_in={a.u_in}, grid {len(x_c)}x{len(y_c)}, "
          f"{len(lag_x)} markers, dt={a.dt:g}, steps={a.steps}")
    run(lambda P, U, V: step(P, U, V), P, U, V, a.steps, n_print=a.print_every,
        n_save=a.save_every, body=body, out=out, meta=meta)


if __name__ == "__main__":
    main()
