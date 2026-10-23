"""Grid generation and immersed-body marker generation."""
import numpy as np


def init_domain(x_faces, y_faces, U_init=0.0, P_init=0.0):
    """Build a cell-centred grid (with one ghost layer) from face coordinates.

    Parameters
    ----------
    x_faces, y_faces : 1-D arrays of cell-face positions spanning the physical domain.
    U_init, P_init   : uniform initial u-velocity and pressure.

    Returns
    -------
    x_c, y_c, dx, dy, xx, yy, P, U, V
    """
    x_dom = np.concatenate([[2.0 * x_faces[0] - x_faces[1]], x_faces,
                            [2.0 * x_faces[-1] - x_faces[-2]]])
    y_dom = np.concatenate([[2.0 * y_faces[0] - y_faces[1]], y_faces,
                            [2.0 * y_faces[-1] - y_faces[-2]]])
    x_c = 0.5 * (x_dom[1:] + x_dom[:-1])
    y_c = 0.5 * (y_dom[1:] + y_dom[:-1])
    dx = x_dom[1:] - x_dom[:-1]
    dy = y_dom[1:] - y_dom[:-1]
    xx, yy = np.meshgrid(x_c, y_c, indexing="ij")
    U = U_init * np.ones_like(xx)
    V = np.zeros_like(xx)
    P = P_init * np.ones_like(xx)
    return x_c, y_c, dx, dy, xx, yy, P, U, V


def external_flow_grid(dim, conc_region, in_region, out_region, wall_region,
                       step_conc, step_in, step_out, step_wall):
    """Piecewise-uniform grid, refined in a box of half-width ``conc_region*dim``
    around the origin, coarser towards the inlet, outlet and side walls.
    All extents are in multiples of the body size ``dim``."""
    x = np.concatenate([
        np.linspace(-in_region * dim, -conc_region * dim, step_in),
        np.linspace(-conc_region * dim, conc_region * dim, step_conc)[1:-1],
        np.linspace(conc_region * dim, out_region * dim, step_out),
    ])
    y = np.concatenate([
        np.linspace(-wall_region * dim, -conc_region * dim, step_wall),
        np.linspace(-conc_region * dim, conc_region * dim, step_conc)[1:-1],
        np.linspace(conc_region * dim, wall_region * dim, step_wall),
    ])
    return x, y


def circle(radius, spacing, center=(0.0, 0.0)):
    """Marker points on a circle, approximately ``spacing`` apart."""
    dang = spacing / radius
    ang = np.arange(0.0, 2.0 * np.pi - 0.1 * dang, dang)
    return center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)


def naca_symmetric(chord, spacing, thickness_pct, leading_edge=(0.0, 0.0)):
    """Marker points on a symmetric 4-digit NACA airfoil (e.g. thickness_pct=12 -> NACA 0012)."""
    t = 0.01 * thickness_pct * chord
    x = np.arange(0.0, chord + 0.1 * spacing, spacing)
    x = x[x <= chord]
    xn = x / chord
    y = 5.0 * t * (0.2969 * np.sqrt(xn) - 0.1260 * xn - 0.3516 * xn ** 2
                   + 0.2843 * xn ** 3 - 0.1015 * xn ** 4)
    lag_x = np.concatenate([x, x[1:]]) + leading_edge[0]
    lag_y = np.concatenate([y, -y[1:]]) + leading_edge[1]
    return lag_x, lag_y


# Names used in the original notebook
init_domains = init_domain
create_circle = circle
NACA_sym_airfoil = naca_symmetric
