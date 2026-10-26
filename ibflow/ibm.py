"""Direct-forcing immersed boundary method (IBM).

The body is a set of Lagrangian marker points. At each marker the fluid velocity is
interpolated with a 4-point cosine delta kernel, a forcing that drives it to zero
(no-slip, stationary body) is computed, and that forcing is spread back to the
Eulerian grid with the same kernel.
"""
import numpy as np
from numba import jit

NEIGH = 3  # stencil half-width searched around each marker


@jit(nopython=True)
def _locate(lx, ly, x_c, y_c):
    """Index of the cell-centre pair bracketing a marker point."""
    near_i = 0
    near_j = 0
    for i in range(0, len(x_c) - 1):
        if x_c[i] < lx and lx <= x_c[i + 1]:
            near_i = i
            break
    for j in range(0, len(y_c) - 1):
        if y_c[j] < ly and ly <= y_c[j + 1]:
            near_j = j
            break
    return near_i, near_j


@jit(nopython=True)
def _delta(r):
    """Cosine regularised delta function, support |r| <= 2."""
    if r <= 2.0:
        return 0.25 * (1.0 + np.cos(0.5 * np.pi * r))
    return 0.0


@jit(nopython=True)
def _interp_velocity(U, V, x_c, y_c, lx, ly, near_i, near_j, dx_h, dy_h):
    U_int = 0.0
    V_int = 0.0
    for i_ind in range(near_i - NEIGH + 1, near_i + NEIGH + 1):
        for j_ind in range(near_j - NEIGH + 1, near_j + NEIGH + 1):
            delta = _delta(abs(x_c[i_ind] - lx) / dx_h) * _delta(abs(y_c[j_ind] - ly) / dy_h)
            U_int += delta * U[i_ind, j_ind]
            V_int += delta * V[i_ind, j_ind]
    return U_int, V_int


@jit(nopython=True)
def ibm_forcing(U, V, x_c, y_c, lag_x, lag_y, timestep):
    """Eulerian forcing (dU/dt, dV/dt contributions) enforcing no-slip on markers."""
    der_u = np.zeros_like(U)
    der_v = np.zeros_like(V)
    for k in range(len(lag_x)):
        lx = lag_x[k]
        ly = lag_y[k]
        near_i, near_j = _locate(lx, ly, x_c, y_c)
        dx_h = x_c[near_i + 1] - x_c[near_i]
        dy_h = y_c[near_j + 1] - y_c[near_j]
        U_int, V_int = _interp_velocity(U, V, x_c, y_c, lx, ly, near_i, near_j, dx_h, dy_h)

        fx = -U_int / timestep
        fy = -V_int / timestep

        for i_ind in range(near_i - NEIGH + 1, near_i + NEIGH + 1):
            for j_ind in range(near_j - NEIGH + 1, near_j + NEIGH + 1):
                delta = _delta(abs(x_c[i_ind] - lx) / dx_h) * _delta(abs(y_c[j_ind] - ly) / dy_h)
                der_u[i_ind, j_ind] += delta * fx
                der_v[i_ind, j_ind] += delta * fy
    return der_u, der_v


@jit(nopython=True)
def ibm_body_force(U, V, x_c, y_c, lag_x, lag_y, timestep, rho):
    """Net hydrodynamic force (Fx, Fy) on the body, per unit span.

    Equal and opposite to the total IBM forcing applied to the fluid.
    """
    f_x = 0.0
    f_y = 0.0
    for k in range(len(lag_x)):
        lx = lag_x[k]
        ly = lag_y[k]
        near_i, near_j = _locate(lx, ly, x_c, y_c)
        dx_h = x_c[near_i + 1] - x_c[near_i]
        dy_h = y_c[near_j + 1] - y_c[near_j]
        U_int, V_int = _interp_velocity(U, V, x_c, y_c, lx, ly, near_i, near_j, dx_h, dy_h)
        f_x += rho * dx_h * dy_h * U_int / timestep
        f_y += rho * dx_h * dy_h * V_int / timestep
    return f_x, f_y


# Names used in the original notebook
IBM_forcing_der = ibm_forcing
IBM_force_computation = ibm_body_force
