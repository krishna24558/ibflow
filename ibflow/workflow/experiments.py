"""Programmatic, cached simulation runs: the building block every higher layer calls.

A run is fully described by a :class:`CylinderConfig` plus a Reynolds number. Its
results (force history, metrics, optional flow snapshots) are cached on disk under a
hash of those inputs, so a sweep, the active learner or an LLM agent asking for the
same case twice never pays for it twice.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from functools import partial

import numpy as np

from ..geometry import init_domain, external_flow_grid, circle
from ..integrators import rk2_ibm_convective_step
from ..ibm import ibm_body_force
from ..solver import compute_divergence
from .diagnostics import analyze_forces

DEFAULT_CACHE = os.path.join("runs")


@dataclass(frozen=True)
class CylinderConfig:
    """Numerical set-up of the flow-past-a-cylinder case (Re is given per run)."""
    diameter: float = 1.5
    u_in: float = 5.0
    rho: float = 1.0
    beta: float = 400.0
    dt: float = 1e-3
    t_end: float = 40.0
    n_conc: int = 48          # points across the refined box around the body
    conc_region: float = 2.0  # half-width of refined box, in diameters
    in_region: float = 6.0    # inlet distance, in diameters
    out_region: float = 12.0  # outlet distance, in diameters
    wall_region: float = 5.0  # side-wall distance, in diameters
    step_in: int = 25
    step_out: int = 80
    step_wall: int = 25
    perturb: float = 0.1      # initial wake kick (fraction of u_in) that breaks symmetry
    force_every: int = 20     # log forces every this many steps

    def key(self, re: float, extra: str = "") -> str:
        blob = json.dumps({"re": round(float(re), 6), **asdict(self), "extra": extra}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


PRESETS = {
    # ~1 min per run on a laptop core; St within ~2% of the literature at Re=100.
    "fast": CylinderConfig(),
    # The original study's grid and domain: ~16 cells per diameter, ~45 min per run.
    "default": CylinderConfig(dt=4e-4, n_conc=80, conc_region=2.5, in_region=8.0,
                              out_region=15.0, wall_region=6.0, step_in=60,
                              step_out=200, step_wall=50, force_every=200),
    # For tests: seconds per run, physics not converged.
    "smoke": CylinderConfig(t_end=1.0, n_conc=30, step_in=12, step_out=30, step_wall=12),
}


def get_config(preset_or_config) -> CylinderConfig:
    if isinstance(preset_or_config, CylinderConfig):
        return preset_or_config
    try:
        return PRESETS[preset_or_config]
    except KeyError:
        raise ValueError(f"Unknown preset {preset_or_config!r}; choose from {sorted(PRESETS)}")


def _setup(re: float, cfg: CylinderConfig):
    D, U_in = cfg.diameter, cfg.u_in
    x, y = external_flow_grid(D, cfg.conc_region, cfg.in_region, cfg.out_region,
                              cfg.wall_region, cfg.n_conc, cfg.step_in, cfg.step_out,
                              cfg.step_wall)
    x_c, y_c, dx, dy, xx, yy, P, U, V = init_domain(x, y, U_init=U_in, P_init=0.0)
    lag_x, lag_y = circle(D / 2, dx.min())
    if cfg.perturb:
        V += cfg.perturb * U_in * np.exp(-((xx - 1.5 * D) ** 2 + (yy - 0.25 * D) ** 2) / (0.5 * D) ** 2)
    nu = U_in * D / re
    step = partial(rk2_ibm_convective_step, U_in=U_in, P_out=0.0, dt=cfg.dt, rho=cfg.rho,
                   nu=nu, beta=cfg.beta, dx=dx, dy=dy, x_c=x_c, y_c=y_c,
                   lag_x=lag_x, lag_y=lag_y)
    return dict(x_c=x_c, y_c=y_c, dx=dx, dy=dy, lag_x=lag_x, lag_y=lag_y, nu=nu,
                P=P, U=U, V=V, step=step)


def run_cylinder(re: float, config="fast", cache_dir: str | None = DEFAULT_CACHE,
                 snapshots: int = 0, snapshot_window: float = 8.0,
                 t_end: float | None = None, verbose: bool = False) -> dict:
    """Run (or load from cache) flow past a cylinder at Reynolds number ``re``.

    Parameters
    ----------
    snapshots       : number of (u, v) flow fields to keep, evenly spaced over the last
                      ``snapshot_window`` time units (for reduced-order modelling).
    t_end           : override the configured end time (used to extend a run).

    Returns a dict with ``id``, ``re``, the metrics from
    :func:`~ibflow.workflow.diagnostics.analyze_forces`, the force history, and
    (if requested) ``snapshot_file``.
    """
    cfg = get_config(config)
    if t_end is not None:
        cfg = dataclasses.replace(cfg, t_end=float(t_end))
    run_id = f"re{re:g}_{cfg.key(re, extra=f's{snapshots}w{snapshot_window}')}"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, run_id + ".json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                rec = json.load(fh)
            rec["cached"] = True
            # Metrics are derived data: recompute them from the stored force history so
            # improvements to the diagnostics apply to old runs too.
            if rec.get("status") != "diverged" and rec.get("t"):
                rec.update(analyze_forces(rec["t"], rec["cl"], rec["cd"], cfg.diameter, cfg.u_in))
            return rec

    s = _setup(re, cfg)
    P, U, V, step = s["P"], s["U"], s["V"], s["step"]
    n_steps = int(round(cfg.t_end / cfg.dt))
    q = 0.5 * cfg.rho * cfg.u_in ** 2 * cfg.diameter
    t_hist, cd_hist, cl_hist = [], [], []
    snap_steps = set()
    if snapshots:
        first = n_steps - int(round(snapshot_window / cfg.dt))
        snap_steps = set(np.linspace(max(first, 0), n_steps - 1, snapshots).astype(int).tolist())
    snaps_u, snaps_v, snaps_t = [], [], []

    t0 = time.time()
    diverged = False
    for i in range(n_steps):
        P, U, V = step(P, U, V)
        if i % cfg.force_every == 0 or i == n_steps - 1:
            fx, fy = ibm_body_force(U, V, s["x_c"], s["y_c"], s["lag_x"], s["lag_y"], cfg.dt, cfg.rho)
            t_hist.append((i + 1) * cfg.dt)
            cd_hist.append(fx / q)
            cl_hist.append(fy / q)
            if not np.isfinite(fx):
                diverged = True
                break
        if i in snap_steps:
            snaps_u.append(U.astype(np.float32))
            snaps_v.append(V.astype(np.float32))
            snaps_t.append((i + 1) * cfg.dt)
        if verbose and i % max(n_steps // 10, 1) == 0:
            print(f"  Re={re:g}: step {i}/{n_steps}  C_L={cl_hist[-1]:+.3f}", flush=True)
    wall = time.time() - t0

    t_hist, cd_hist, cl_hist = map(np.asarray, (t_hist, cd_hist, cl_hist))
    metrics = analyze_forces(t_hist, cl_hist, cd_hist, cfg.diameter, cfg.u_in)
    if diverged:
        metrics["status"] = "diverged"
        metrics["reasons"] = ["force became NaN/Inf"] + metrics.get("reasons", [])
    div = float(np.nanmax(np.abs(compute_divergence(U, V, s["dx"], s["dy"])))) if not diverged else float("nan")

    rec = {
        "id": run_id, "re": float(re), "config": asdict(cfg), "wall_time_s": round(wall, 1),
        "max_div": div, **metrics,
        "t": t_hist.tolist(), "cd": cd_hist.tolist(), "cl": cl_hist.tolist(),
        "cached": False,
    }
    if snapshots and cache_dir:
        snap_file = os.path.join(cache_dir, run_id + "_snapshots.npz")
        np.savez_compressed(snap_file, u=np.array(snaps_u), v=np.array(snaps_v),
                            t=np.array(snaps_t), x_c=s["x_c"], y_c=s["y_c"],
                            lag_x=s["lag_x"], lag_y=s["lag_y"], re=re,
                            diameter=cfg.diameter, u_in=cfg.u_in)
        rec["snapshot_file"] = snap_file
    if cache_dir:
        with open(meta_path, "w") as fh:
            json.dump(rec, fh)
    return rec


def summary(rec: dict) -> dict:
    """The compact part of a run record (no time histories): what agents and tables see."""
    keys = ["id", "re", "status", "st", "cl_amp", "cd_mean", "growth_rate", "saturation",
            "max_div", "reasons", "wall_time_s", "cached"]
    out = {k: rec.get(k) for k in keys if k in rec}
    out["t_end"] = rec.get("config", {}).get("t_end")
    return out
