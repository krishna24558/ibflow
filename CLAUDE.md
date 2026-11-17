# ibflow

2-D incompressible-flow solver in Python + Numba: EDAC pressure evolution (no pressure-Poisson solve),
collocated finite volumes on a Cartesian grid with one ghost layer, direct-forcing immersed boundary
method (IBM), TVD (SSP) RK2 time stepping. Portfolio project; every claim in the README must be
backed by a run in the repo.

## Layout

- `ibflow/solver.py` – `edacm_rhs` (EDAC right-hand side), `linear_interp`, `compute_divergence`
- `ibflow/boundary.py` – ghost-cell BCs: `lid_driven_cavity`, `channel`, `external_flow`, `convective_outlet`
- `ibflow/ibm.py` – `ibm_forcing` (interpolate → force → spread, 4-point cosine kernel), `ibm_body_force`
- `ibflow/integrators.py` – `rk2_step`, `rk2_ibm_step`, `rk2_ibm_convective_step`
- `ibflow/geometry.py` – `init_domain`, `external_flow_grid`, `circle`, `naca_symmetric`
- `ibflow/runner.py` – `run`: time loop, divergence/force logging, progressive `.npz` saving
- `cases/` – `lid_cavity.py`, `channel.py`, `external_body.py` (cylinder / naca0012, `--perturb`)
- `plot.py` – `cavity | divergence | channel | field | forces`
- `validation/ghia_1982_u_centerline.csv` – Ghia, Ghia & Shin (1982) Table I
- `tests/test_solver.py` – 8 sanity tests; `.github/workflows/tests.yml` runs them on push/PR

## Commands (run from the repo root; Windows has no `make`, use the python lines)

```
python -m pip install -r requirements.txt
python -m pytest -q tests                     # 8 passed
python cases/lid_cavity.py --re 100 --dt 1e-4 --steps 200000 --n-fine 41 --n-coarse 41 --print-every 20000 --out results/val_cavity_re100.npz
python cases/lid_cavity.py --re 400 --dt 1e-4 --steps 600000 --n-fine 41 --n-coarse 41 --print-every 50000 --out results/val_cavity_re400.npz
python plot.py cavity results/val_cavity_re100.npz
python plot.py divergence results/val_cavity_re100.npz
python cases/external_body.py --body cylinder --perturb 0.1 --dt 4e-4 --steps 100000 --print-every 5000 --out results/cylinder_shedding.npz
python plot.py forces results/cylinder_shedding.npz
```

`results/` is gitignored (except `.gitkeep`); figures meant for the README are copied into `docs/`.

## Reference numbers (verified; use them to check changes)

| Check | Expected |
|---|---|
| Cavity Re=100, 80×80, t=20 | final max\|∇·u\| 6.751e-02; vs Ghia max err 0.0044, RMS 0.0020 |
| Cavity Re=400, t=20 (200k steps) | not steady; max err 0.0128, RMS 0.0062 |
| Cavity Re=400, t=60 (600k steps) | steady; max err 0.0044, RMS 0.0023; min u −0.3229 (Ghia −0.3273) |
| Divergence peak | (0.981, 0.981), top-right lid corner, both Re |
| β 500 → 2000 at Re=100 | max\|∇·u\| 0.0675 → 0.0177 (scales as 1/β) |
| Cylinder Re=100, perturbed, t=40 | St 0.165, C_L amplitude 0.475, mean C_D 1.555 |

Ghia errors: interpolate u to x=0.5 per row, then linearly in y at Ghia's 15 interior points
(drop y=0 and 1 and ghost cells); max and RMS over those 15, in units of the lid speed.

## Physics and numerics facts that matter when editing

- Implemented pressure equation: ∂p/∂t = −β∇·u + ν∇²p (Clausen's u·∇p term is omitted). At steady
  state ∇·u = (ν/β)∇²p, so a nonzero divergence is expected, not a bug.
- Time step is acoustic-limited: (|u| + √β)Δt/Δx ≲ 1. Cylinder set-up has Ma = U/√β = 0.25.
- Collocated grid without Rhie–Chow: faint checkerboard in ∇·u at ~1e-4 is known.
- Convection is central differencing; cell Péclet > 2 at Re=400 and in the cylinder case.
- IBM: stationary bodies only; forcing F = −U_marker/Δt; spreading reuses interpolation weights
  (no Δs/h² factor). The body acts slightly larger than its nominal diameter.
- Cylinder C_D and C_L are 15–45% above literature on this grid; St matches. No grid-convergence study yet.
- Two cells of the Ghia CSV (Re=3200 and 10000 columns) are repository edits, not verified errata;
  ibflow uses only Re=100 and 400.

## Working rules

- Any change to `ibflow/` numerics: run the tests and at least the Re=100 cavity, and compare with
  the table above before committing.
- Label claims as implemented / demonstrated / staged / proposed; never report a number that no run produced.
- Small commits with imperative messages; larger features on a branch with a pull request. Never
  backdate commits or force-push `main`.
- Keep `.npz` results and `runs/` caches out of git.
