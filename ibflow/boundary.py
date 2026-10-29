"""Ghost-cell boundary conditions.

Every function has the signature ``bc(P, U, V, U_ref, P_ref) -> (P, U, V)`` and
modifies the arrays in place. Index 0 / -1 along axis 0 are the west / east ghost
columns; along axis 1 they are the south / north ghost rows.
"""
import numpy as np


def lid_driven_cavity(P, U, V, U_wall, P_fix):
    """Closed box, no-slip walls, top lid moving at U_wall. Pressure pinned at one cell."""
    m, n = np.shape(P)
    # West wall
    P[0, :] = P[1, :]
    U[0, :] = -U[1, :]
    V[0, :] = -V[1, :]
    # East wall
    P[m - 1, :] = P[m - 2, :]
    U[m - 1, :] = -U[m - 2, :]
    V[m - 1, :] = -V[m - 2, :]
    # South wall
    P[:, 0] = P[:, 1]
    U[:, 0] = -U[:, 1]
    V[:, 0] = -V[:, 1]
    # North wall (moving lid)
    P[:, n - 1] = P[:, n - 2]
    U[:, n - 1] = 2.0 * U_wall - U[:, n - 2]
    V[:, n - 1] = -V[:, n - 2]
    # Pin pressure level (closed domain has no pressure reference otherwise)
    P[2, 2] = P_fix
    return P, U, V


def channel(P, U, V, U_in, P_out):
    """Plane channel: uniform inlet, fixed-pressure zero-gradient outlet, no-slip walls."""
    m, n = np.shape(P)
    # Inlet
    P[0, :] = P[1, :]
    U[0, :] = 2.0 * U_in - U[1, :]
    V[0, :] = -V[1, :]
    # Outlet
    P[m - 1, :] = P_out
    U[m - 1, :] = U[m - 2, :]
    V[m - 1, :] = V[m - 2, :]
    # Bottom / top walls
    P[:, 0] = P[:, 1]
    U[:, 0] = -U[:, 1]
    V[:, 0] = -V[:, 1]
    P[:, n - 1] = P[:, n - 2]
    U[:, n - 1] = -U[:, n - 2]
    V[:, n - 1] = -V[:, n - 2]
    return P, U, V


def external_flow(P, U, V, U_in, P_out):
    """Free stream: uniform inlet, zero-gradient outlet, slip (symmetry) side walls."""
    m, n = np.shape(P)
    P[0, :] = P[1, :]
    U[0, :] = 2.0 * U_in - U[1, :]
    V[0, :] = -V[1, :]
    P[m - 1, :] = P_out
    U[m - 1, :] = U[m - 2, :]
    V[m - 1, :] = V[m - 2, :]
    P[:, 0] = P[:, 1]
    U[:, 0] = U[:, 1]
    V[:, 0] = -V[:, 1]
    P[:, n - 1] = P[:, n - 2]
    U[:, n - 1] = U[:, n - 2]
    V[:, n - 1] = -V[:, n - 2]
    return P, U, V


def convective_outlet(P, U, V, U_in, P_out):
    """Like ``external_flow`` but the outlet U ghost is advanced by
    :func:`convective_outlet_forcing` instead of being copied, so wakes leave cleanly."""
    m, n = np.shape(P)
    P[0, :] = P[1, :]
    U[0, :] = 2.0 * U_in - U[1, :]
    V[0, :] = -V[1, :]
    P[m - 1, :] = P_out
    V[m - 1, :] = V[m - 2, :]
    P[:, 0] = P[:, 1]
    U[:, 0] = U[:, 1]
    V[:, 0] = -V[:, 1]
    P[:, n - 1] = P[:, n - 2]
    U[:, n - 1] = U[:, n - 2]
    V[:, n - 1] = -V[:, n - 2]
    return P, U, V


def convective_outlet_forcing(U, U_conv, dx):
    """dU/dt in the outlet ghost column for du/dt + U_conv du/dx = 0."""
    der_u = np.zeros_like(U)
    m, n = np.shape(U)
    dx_l = 0.5 * (dx[m - 1] + dx[m - 2])
    der_u[m - 1, 1:n - 1] = -U_conv * (U[m - 1, 1:n - 1] - U[m - 2, 1:n - 1]) / dx_l
    return der_u


# Names used in the original notebook
Lid_bc = lid_driven_cavity
Channel_bc = channel
External_flow_bc = external_flow
Convective_outlet_flow_bc = convective_outlet
