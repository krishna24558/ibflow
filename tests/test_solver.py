"""Fast sanity tests (run in a few seconds with ``make test``)."""
import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ibflow import (linear_interp, edacm_rhs, compute_divergence, init_domain,  # noqa: E402
                   lid_driven_cavity, rk2_step, circle, naca_symmetric, ibm_forcing,
                   external_flow_grid, rk2_ibm_convective_step, ibm_body_force)


def test_linear_interp_uniform_is_average():
    assert np.isclose(linear_interp(1.0, 0.5, 3.0, 0.5), 2.0)


def test_linear_interp_weights_by_distance():
    # Face is 1 from cell 1's centre and 3 from cell 2's centre -> closer to val1
    assert np.isclose(linear_interp(0.0, 2.0, 4.0, 6.0), 1.0)


def test_uniform_flow_is_steady():
    x = np.linspace(0, 1, 21)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, x, U_init=2.0, P_init=1.0)
    dp, du, dv = edacm_rhs(P, U, V, 1.0, 0.01, 100.0, dx, dy)
    assert np.allclose(dp, 0) and np.allclose(du, 0) and np.allclose(dv, 0)
    assert np.allclose(compute_divergence(U, V, dx, dy), 0)


def test_cavity_short_run_stays_finite():
    faces = np.linspace(0, 1, 33)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(faces, faces, U_init=1.0)
    step = partial(rk2_step, lid_driven_cavity, U_ref=1.0, P_ref=0.0, dt=1e-4,
                   rho=1.0, nu=0.01, beta=100.0, dx=dx, dy=dy)
    for _ in range(300):
        P, U, V = step(P, U, V)
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(P))
    # Lid drags the fluid to the right near the top, so recirculation sends it left lower down.
    assert U[len(x_c) // 2, -2] > 0


def test_circle_markers():
    lx, ly = circle(0.5, 0.05)
    assert np.allclose(np.hypot(lx, ly), 0.5)
    spacing = np.hypot(np.diff(lx), np.diff(ly))
    assert np.allclose(spacing, spacing[0], rtol=1e-6) and abs(spacing[0] - 0.05) < 1e-3


def test_naca0012_shape():
    lx, ly = naca_symmetric(1.0, 0.01, 12)
    assert np.isclose(lx.min(), 0.0) and np.isclose(lx.max(), 1.0)
    assert abs(2 * ly.max() - 0.12) < 2e-3            # 12% thickness
    assert np.isclose(ly.sum(), 0.0, atol=1e-12)        # symmetric
    assert len(np.unique(np.round(np.c_[lx, ly], 12), axis=0)) == len(lx)  # no duplicate points


def test_ibm_forcing_opposes_velocity():
    x = np.linspace(-1, 1, 41)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, x, U_init=1.0)
    lx, ly = circle(0.3, dx.min())
    du, dv = ibm_forcing(U, V, x_c, y_c, lx, ly, 1e-3)
    assert du.min() < 0 and du.max() <= 0      # only decelerates
    assert np.allclose(dv, 0)
    assert du[0, 0] == 0                       # local to the body


def test_cylinder_drag_positive_and_symmetric():
    x, y = external_flow_grid(1.0, 1.5, 3.0, 5.0, 3.0, 30, 10, 20, 10)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, y, U_init=1.0)
    lx, ly = circle(0.5, dx.min())
    step = partial(rk2_ibm_convective_step, U_in=1.0, P_out=0.0, dt=1e-4, rho=1.0, nu=0.02,
                   beta=50.0, dx=dx, dy=dy, x_c=x_c, y_c=y_c, lag_x=lx, lag_y=ly)
    for _ in range(200):
        P, U, V = step(P, U, V)
    fx, fy = ibm_body_force(U, V, x_c, y_c, lx, ly, 1e-4, 1.0)
    assert np.isfinite(fx) and fx > 0
    # Marker layout is not exactly mirror-symmetric, so allow a small lift.
    assert abs(fy) < 0.01 * abs(fx)
