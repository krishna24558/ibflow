"""Shared time loop: steps the solution, logs residuals/forces, saves snapshots."""
import os
import time

import numpy as np

from .solver import compute_divergence
from .ibm import ibm_body_force


def run(step, P, U, V, n_steps, n_print=1000, n_save=None, body=None, out=None, meta=None):
    """Advance ``(P, U, V)`` ``n_steps`` times with ``step(P, U, V) -> (P, U, V)``.

    Parameters
    ----------
    body : dict with x_c, y_c, lag_x, lag_y, dt, rho -> also logs body force.
    out  : path of a .npz file to write snapshots, history and ``meta`` into.
           Saved progressively, so an interrupted run still leaves usable data.
    """
    n_save = n_save or max(n_steps // 10, 1)
    snaps = {"P": [], "U": [], "V": [], "step": []}
    hist = {"step": [], "max_div": [], "fx": [], "fy": []}
    dx, dy = meta["dx"], meta["dy"]
    t0 = time.time()

    def save():
        if out is None:
            return
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(
            out,
            **{k: np.array(v) for k, v in snaps.items()},
            **{"hist_" + k: np.array(v) for k, v in hist.items()},
            **meta,
        )

    try:
        for i in range(n_steps):
            P, U, V = step(P, U, V)
            last = i == n_steps - 1
            if i % n_print == 0 or last:
                div = float(np.max(np.abs(compute_divergence(U, V, dx, dy))))
                fx = fy = np.nan
                if body is not None:
                    fx, fy = ibm_body_force(U, V, body["x_c"], body["y_c"], body["lag_x"],
                                            body["lag_y"], body["dt"], body["rho"])
                hist["step"].append(i + 1)
                hist["max_div"].append(div)
                hist["fx"].append(fx)
                hist["fy"].append(fy)
                msg = f"step {i + 1:>8d}/{n_steps}  max|div u| = {div:.3e}"
                if body is not None:
                    msg += f"  Fx = {fx:9.4f}  Fy = {fy:9.4f}"
                print(msg + f"  ({time.time() - t0:6.1f} s)", flush=True)
                if not np.isfinite(div):
                    raise FloatingPointError("Solution diverged (NaN/Inf). Reduce dt or beta.")
            if i % n_save == 0 or last:
                for k, arr in zip("PUV", (P, U, V)):
                    snaps[k].append(np.copy(arr))
                snaps["step"].append(i + 1)
                save()
    except KeyboardInterrupt:
        print("\nInterrupted - saving what has been computed so far.")
        snaps["P"].append(np.copy(P)); snaps["U"].append(np.copy(U))
        snaps["V"].append(np.copy(V)); snaps["step"].append(i + 1)
        save()
    if out:
        print(f"Saved {out}")
    return P, U, V, hist
