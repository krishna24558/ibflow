"""EDAC collocated finite-volume solver with a direct-forcing immersed boundary method."""
from .solver import linear_interp, edacm_rhs, compute_divergence
from .ibm import ibm_forcing, ibm_body_force
from .boundary import (lid_driven_cavity, channel, external_flow, convective_outlet,
                       convective_outlet_forcing)
from .integrators import rk2_step, rk2_ibm_step, rk2_ibm_convective_step
from .geometry import init_domain, external_flow_grid, circle, naca_symmetric
from .runner import run

__version__ = "0.1.0"
