"""Second-order TVD (strong-stability-preserving) Runge-Kutta time stepping.

    q*      = q^n + dt L(q^n)
    q^{n+1} = 1/2 (q^n + q*) + 1/2 dt L(q*)

with boundary conditions applied after each stage.
"""
from .solver import edacm_rhs
from .ibm import ibm_forcing
from .boundary import convective_outlet, convective_outlet_forcing


def rk2_step(bc, P, U, V, U_ref, P_ref, dt, rho, nu, beta, dx, dy):
    """One RK2 step of the pure flow solver (no immersed body)."""
    dp, du, dv = edacm_rhs(P, U, V, rho, nu, beta, dx, dy)
    Pt, Ut, Vt = bc(P + dt * dp, U + dt * du, V + dt * dv, U_ref, P_ref)

    dp, du, dv = edacm_rhs(Pt, Ut, Vt, rho, nu, beta, dx, dy)
    P_new = 0.5 * (P + Pt) + 0.5 * dt * dp
    U_new = 0.5 * (U + Ut) + 0.5 * dt * du
    V_new = 0.5 * (V + Vt) + 0.5 * dt * dv
    return bc(P_new, U_new, V_new, U_ref, P_ref)


def rk2_ibm_step(bc, P, U, V, U_ref, P_ref, dt, rho, nu, beta, dx, dy,
                 x_c, y_c, lag_x, lag_y):
    """One RK2 step with IBM forcing and a user-supplied boundary condition."""
    dp, du1, dv1 = edacm_rhs(P, U, V, rho, nu, beta, dx, dy)
    du2, dv2 = ibm_forcing(U, V, x_c, y_c, lag_x, lag_y, dt)
    Pt, Ut, Vt = bc(P + dt * dp, U + dt * (du1 + du2), V + dt * (dv1 + dv2), U_ref, P_ref)

    dp, du1, dv1 = edacm_rhs(Pt, Ut, Vt, rho, nu, beta, dx, dy)
    du2, dv2 = ibm_forcing(Ut, Vt, x_c, y_c, lag_x, lag_y, dt)
    P_new = 0.5 * (P + Pt) + 0.5 * dt * dp
    U_new = 0.5 * (U + Ut) + 0.5 * dt * (du1 + du2)
    V_new = 0.5 * (V + Vt) + 0.5 * dt * (dv1 + dv2)
    return bc(P_new, U_new, V_new, U_ref, P_ref)


def rk2_ibm_convective_step(P, U, V, U_in, P_out, dt, rho, nu, beta, dx, dy,
                            x_c, y_c, lag_x, lag_y):
    """RK2 step with IBM forcing and a convective outlet (used for bluff bodies)."""
    dp, du1, dv1 = edacm_rhs(P, U, V, rho, nu, beta, dx, dy)
    du2, dv2 = ibm_forcing(U, V, x_c, y_c, lag_x, lag_y, dt)
    du3 = convective_outlet_forcing(U, U_in, dx)
    Pt, Ut, Vt = convective_outlet(P + dt * dp, U + dt * (du1 + du2 + du3),
                                   V + dt * (dv1 + dv2), U_in, P_out)

    dp, du1, dv1 = edacm_rhs(Pt, Ut, Vt, rho, nu, beta, dx, dy)
    du2, dv2 = ibm_forcing(Ut, Vt, x_c, y_c, lag_x, lag_y, dt)
    du3 = convective_outlet_forcing(Ut, U_in, dx)
    P_new = 0.5 * (P + Pt) + 0.5 * dt * dp
    U_new = 0.5 * (U + Ut) + 0.5 * dt * (du1 + du2 + du3)
    V_new = 0.5 * (V + Vt) + 0.5 * dt * (dv1 + dv2)
    return convective_outlet(P_new, U_new, V_new, U_in, P_out)


# Names used in the original notebook
TVD_RK2_integrate = rk2_step
IBM_RK2 = rk2_ibm_step
IBM_integrate_time_convec_outlet = rk2_ibm_convective_step
