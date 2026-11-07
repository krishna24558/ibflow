# ibflow — 2-D Immersed Boundary Flow Solver

A compact 2-D incompressible-flow solver written in Python and accelerated with Numba. It combines

- an **Entropically Damped Artificial Compressibility (EDAC)** formulation on a **collocated finite-volume grid** (non-uniform Cartesian, ghost-cell boundaries),
- a **direct-forcing immersed boundary method (IBM)** with a cosine regularised delta kernel, so bodies of any shape can sit on a simple Cartesian mesh,
- **2nd-order TVD Runge–Kutta** time integration.

Cases included: lid-driven cavity, developing channel flow, flow past a circular cylinder and flow past a NACA 0012 airfoil.

## Validation: lid-driven cavity

Centreline u-velocity compared with Ghia, Ghia & Shin (1982), 80 × 80 grid (`make validate`):

| Re = 100 | Re = 400 |
|:---:|:---:|
| ![Re 100](docs/cavity_re100.png) | ![Re 400](docs/cavity_re400.png) |

| Re | max \|u − u<sub>Ghia</sub>\| | RMS error | min u (this code / Ghia) |
|---:|---:|---:|---:|
| 100 | 0.0044 | 0.0020 | −0.2130 / −0.2109 |
| 400 | 0.0128 | 0.0062 | −0.3204 / −0.3273 |

## Governing equations

With pressure *p*, velocity **u**, density ρ, viscosity ν and artificial-compressibility parameter β:

```
∂p/∂t + β ∇·u          = ν ∇²p          (EDAC pressure evolution)
∂u/∂t + ∇·(u u) + ∇p/ρ = ν ∇²u + f_IBM   (momentum)
```

The ν∇²p term damps the pressure (acoustic) waves that artificial compressibility introduces, so no pressure-Poisson solve is needed and every step is explicit and local. β sets the artificial sound speed √β and so trades the remaining ∇·u error against the stable time step.

**Immersed boundary.** The body is a set of Lagrangian markers. At each marker the fluid velocity is interpolated with a 4-point cosine delta, the forcing `f = −u_marker/Δt` (no-slip, stationary body) is computed, and it is spread back onto the grid with the same kernel. The hydrodynamic force on the body is the negative sum of that forcing.

## Repository layout

```
ibflow/
  solver.py       EDAC right-hand side, face interpolation, divergence
  ibm.py          IBM forcing (interpolate → force → spread) and body force
  boundary.py     cavity, channel, free-stream and convective-outlet ghost cells
  integrators.py  TVD RK2 steps (plain, with IBM, with IBM + convective outlet)
  geometry.py     grids, circle and NACA 4-digit marker generation
  runner.py       time loop, logging, progressive .npz saving
cases/            lid_cavity.py, channel.py, external_body.py (cylinder / naca0012)
validation/       Ghia et al. (1982) centreline data
tests/            fast sanity tests (pytest)
notebooks/        the original exploratory notebook (outputs stripped)
plot.py           post-processing: centreline, profile, field, Cd/Cl history
Makefile
```

## Quick start

```bash
git clone https://github.com/<your-username>/ibflow.git
cd ibflow
make install        # numpy, numba, matplotlib, pytest
make test           # seconds
make quick          # short run of every case to check the pipeline
make validate       # cavity Re=100 and 400 vs Ghia (~5 min)
```

Full runs (each saves `results/<case>.npz` and PNGs):

```bash
make cavity                 # Re=400, original 180x180 stretched grid
make cylinder               # Re=100
make airfoil                # NACA 0012, Re=100
make cylinder STEPS=50000   # shorter run
make cavity RE=100          # different Reynolds number
```

Every case script also takes command-line options (`python cases/external_body.py --help`). Runs save progressively, so stopping with Ctrl-C still leaves usable data.

## Cylinder at Re = 100

From the original 698,000-step run (D = 1.5, U = 5), the drag force settled around F<sub>x</sub> ≈ 24 per unit span, giving

C<sub>D</sub> = F<sub>x</sub> / (½ ρ U² D) ≈ 24 / 18.75 ≈ **1.3**,

which is in the 1.3–1.4 range typically reported for a 2-D cylinder at this Reynolds number.

**Known limitation:** the lift stayed close to zero (|C<sub>L</sub>| ≲ 0.003), whereas a cylinder at Re = 100 should shed vortices with a lift amplitude of roughly ±0.3. The set-up is perfectly symmetric and the run was stopped early, so the wake had most likely not yet become unstable; the slow oscillation in drag is probably artificial-compressibility pressure waves reflecting in the domain. Adding a small initial perturbation and running longer is the next step (see roadmap).

Note: the sphere drag correlation `24/Re (1 + 0.15 Re^0.687)` used in the original notebook applies to spheres, not 2-D cylinders, so it is not used here as a reference.

## Roadmap

- [ ] Trigger vortex shedding (perturbed inflow), report Strouhal number and C<sub>L</sub> amplitude
- [ ] Grid-convergence study for the cylinder drag
- [ ] Angle of attack for the airfoil and lift curve
- [ ] Moving bodies (IBM with prescribed motion)
- [ ] Energy equation (heat transfer from immersed bodies)

## References

- U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method", *J. Comput. Phys.* 48 (1982) 387–411.
- J. A. Clausen, "Entropically damped form of artificial compressibility for explicit simulation of incompressible flow", *Phys. Rev. E* 87 (2013) 013309.
- C. S. Peskin, "The immersed boundary method", *Acta Numerica* 11 (2002) 479–517.

## License

MIT, see [LICENSE](LICENSE).
