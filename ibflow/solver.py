"""Collocated finite-volume EDAC (Entropically Damped Artificial Compressibility) solver.

All fields live at cell centres on a (possibly non-uniform) Cartesian grid with one
layer of ghost cells. Face values are obtained by distance-weighted linear
interpolation and face gradients by two-point differences.

Governing equations (incompressible flow, artificial-compressibility form):

    dp/dt + beta * div(u)        = nu * lap(p)
    du/dt + div(u u) + grad(p)/rho = nu * lap(u)
"""
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def linear_interp(val1, dx1, val2, dx2):
    """Interpolate to the face between two cells of widths dx1 and dx2."""
    return (val1 * dx2 + val2 * dx1) / (dx1 + dx2)


@jit(nopython=True)
def edacm_rhs(P, U, V, rho, nu, beta, dx, dy):
    """Right-hand side (time derivatives) of p, u and v at every interior cell.

    Returns
    -------
    der_p, der_u, der_v : ndarray
        dP/dt, dU/dt, dV/dt. Ghost cells are left at zero.
    """
    m = len(dx)
    n = len(dy)
    der_p = np.zeros_like(P)
    der_u = np.zeros_like(U)
    der_v = np.zeros_like(V)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            # Face values
            P_east = linear_interp(P[i, j], dx[i], P[i + 1, j], dx[i + 1])
            P_west = linear_interp(P[i - 1, j], dx[i - 1], P[i, j], dx[i])
            P_north = linear_interp(P[i, j], dy[j], P[i, j + 1], dy[j + 1])
            P_south = linear_interp(P[i, j - 1], dy[j - 1], P[i, j], dy[j])

            U_east = linear_interp(U[i, j], dx[i], U[i + 1, j], dx[i + 1])
            U_west = linear_interp(U[i - 1, j], dx[i - 1], U[i, j], dx[i])
            U_north = linear_interp(U[i, j], dy[j], U[i, j + 1], dy[j + 1])
            U_south = linear_interp(U[i, j - 1], dy[j - 1], U[i, j], dy[j])

            V_east = linear_interp(V[i, j], dx[i], V[i + 1, j], dx[i + 1])
            V_west = linear_interp(V[i - 1, j], dx[i - 1], V[i, j], dx[i])
            V_north = linear_interp(V[i, j], dy[j], V[i, j + 1], dy[j + 1])
            V_south = linear_interp(V[i, j - 1], dy[j - 1], V[i, j], dy[j])

            # Convective and pressure-gradient terms
            con_p_x = beta * (U_east - U_west) / dx[i]
            con_p_y = beta * (V_north - V_south) / dy[j]
            con_u_x = (U_east ** 2 - U_west ** 2) / dx[i]
            con_u_y = (U_north * V_north - U_south * V_south) / dy[j]
            con_v_x = (U_east * V_east - U_west * V_west) / dx[i]
            con_v_y = (V_north ** 2 - V_south ** 2) / dy[j]
            pressure_u = (P_east - P_west) / (rho * dx[i])
            pressure_v = (P_north - P_south) / (rho * dy[j])

            # Face gradients -> diffusion terms
            dxe = 0.5 * (dx[i] + dx[i + 1])
            dxw = 0.5 * (dx[i - 1] + dx[i])
            dyn = 0.5 * (dy[j] + dy[j + 1])
            dys = 0.5 * (dy[j - 1] + dy[j])

            diff_p_x = nu * ((P[i + 1, j] - P[i, j]) / dxe - (P[i, j] - P[i - 1, j]) / dxw) / dx[i]
            diff_p_y = nu * ((P[i, j + 1] - P[i, j]) / dyn - (P[i, j] - P[i, j - 1]) / dys) / dy[j]
            diff_u_x = nu * ((U[i + 1, j] - U[i, j]) / dxe - (U[i, j] - U[i - 1, j]) / dxw) / dx[i]
            diff_u_y = nu * ((U[i, j + 1] - U[i, j]) / dyn - (U[i, j] - U[i, j - 1]) / dys) / dy[j]
            diff_v_x = nu * ((V[i + 1, j] - V[i, j]) / dxe - (V[i, j] - V[i - 1, j]) / dxw) / dx[i]
            diff_v_y = nu * ((V[i, j + 1] - V[i, j]) / dyn - (V[i, j] - V[i, j - 1]) / dys) / dy[j]

            der_p[i, j] = diff_p_x + diff_p_y - con_p_x - con_p_y
            der_u[i, j] = diff_u_x + diff_u_y - con_u_x - con_u_y - pressure_u
            der_v[i, j] = diff_v_x + diff_v_y - con_v_x - con_v_y - pressure_v

    return der_p, der_u, der_v


@jit(nopython=True, fastmath=True)
def compute_divergence(U, V, dx, dy):
    """Cell-centred velocity divergence (two cells in from each boundary)."""
    div_mat = np.zeros_like(U)
    m = len(dx)
    n = len(dy)
    for i in range(2, m - 2):
        for j in range(2, n - 2):
            U_east = linear_interp(U[i, j], dx[i], U[i + 1, j], dx[i + 1])
            U_west = linear_interp(U[i - 1, j], dx[i - 1], U[i, j], dx[i])
            V_north = linear_interp(V[i, j], dy[j], V[i, j + 1], dy[j + 1])
            V_south = linear_interp(V[i, j - 1], dy[j - 1], V[i, j], dy[j])
            div_mat[i, j] = (U_east - U_west) / dx[i] + (V_north - V_south) / dy[j]
    return div_mat


# Backwards-compatible name used in the original notebook
CFD_EDACM = edacm_rhs
