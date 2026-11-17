"""Automatic quality checks on a run's force history.

Decides, from the lift signal alone, whether the wake is

* ``steady``   - lift oscillations decay: no vortex shedding at this Re,
* ``growing``  - oscillations grow but have not saturated yet: shedding is starting,
* ``periodic`` - saturated periodic shedding: St, C_L amplitude and C_D are trustworthy,
* ``diverged`` - the solver blew up.

It also measures the linear growth rate sigma of the lift envelope. Near the onset of
shedding sigma varies linearly with Re (a Hopf bifurcation), so a few runs either side
give the critical Reynolds number by interpolating sigma = 0.
"""
from __future__ import annotations

import numpy as np

T_SKIP = 5.0          # ignore the start-up transient from the impulsive start and the kick
FIT_FRAC = 0.4        # growth rate is fitted on the last 60% of the run, after the kick's transient
STEADY_AMP = 2e-3     # final lift amplitude below this is "no shedding"
LIMIT_CYCLE_AMP = 2e-2  # a saturated shedding cycle has at least this lift amplitude; below it a
                        # constant amplitude means "neutral" (right at the onset), not "saturated"
SAT_TOL = 0.05        # last two windows' amplitudes within 5% -> saturated


def _peaks(t, sig):
    """Times and absolute values of local maxima of |sig|."""
    a = np.abs(sig)
    idx = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1
    return t[idx], a[idx]


def analyze_forces(t, cl, cd, diameter, u_in) -> dict:
    t, cl, cd = (np.asarray(v, dtype=float) for v in (t, cl, cd))
    out = {"status": "steady", "st": 0.0, "cl_amp": 0.0, "cd_mean": float("nan"),
           "growth_rate": float("nan"), "saturation": float("nan"), "reasons": []}
    if len(t) < 20 or not np.all(np.isfinite(cl)):
        out["status"] = "diverged" if not np.all(np.isfinite(cl)) else "too_short"
        out["reasons"].append("non-finite forces" if out["status"] == "diverged" else "too few samples")
        return out

    late = t >= t[0] + 0.5 * (t[-1] - t[0])
    out["cd_mean"] = float(cd[late].mean())
    mask = t >= T_SKIP
    tt = t[mask]
    c = cl[mask] - np.median(cl[late])        # remove small asymmetry offset

    # Amplitude in the last two quarter-windows
    q = len(tt) // 4
    a_prev = float(np.abs(c[-2 * q:-q]).max()) if q > 2 else 0.0
    a_last = float(np.abs(c[-q:]).max()) if q > 2 else 0.0
    out["cl_amp"] = a_last
    out["saturation"] = a_last / a_prev if a_prev > 0 else float("nan")

    # Envelope growth rate: slope of log(peak) over the linear (small-amplitude) regime
    tp, ap = _peaks(tt, c)
    late_fit = tp >= t[0] + FIT_FRAC * (t[-1] - t[0])
    saturated = (a_last >= LIMIT_CYCLE_AMP and np.isfinite(out["saturation"])
                 and abs(out["saturation"] - 1) <= SAT_TOL)
    if saturated:
        lin = ap < 0.5 * a_last                      # growth phase, before nonlinear saturation
        if lin.sum() < 4:
            lin = late_fit & (ap < 0.5 * a_last)
    else:
        lin = late_fit                               # decaying, neutral or still growing
    if lin.sum() >= 4:
        out["growth_rate"] = float(np.polyfit(tp[lin], np.log(ap[lin] + 1e-15), 1)[0])

    # Frequency from upward zero crossings in the second half (needs some oscillation)
    half = c[len(c) // 2:]
    th = tt[len(c) // 2:]
    up = th[1:][(half[:-1] < 0) & (half[1:] >= 0)]
    if len(up) > 2 and a_last > STEADY_AMP:
        out["st"] = float(diameter / u_in / np.diff(up).mean())

    sigma = out["growth_rate"]
    if a_last < STEADY_AMP and (not np.isfinite(sigma) or sigma <= 0):
        out["status"] = "steady"
        out["st"] = 0.0
        out["reasons"].append(f"lift amplitude {a_last:.1e} below {STEADY_AMP:g} and decaying")
    elif saturated:
        out["status"] = "periodic"
        out["reasons"].append(f"amplitude constant to {abs(out['saturation'] - 1):.1%} over last two windows")
    elif a_last < LIMIT_CYCLE_AMP and np.isfinite(sigma):
        # small, slowly changing oscillation: right at the onset. The sign of the growth rate decides.
        out["status"] = "growing" if sigma > 0 else "steady"
        out["reasons"].append(f"near-neutral oscillation (amplitude {a_last:.1e}); growth rate {sigma:+.4f} "
                              f"says {'unstable' if sigma > 0 else 'stable'}")
    elif out["saturation"] > 1:
        out["status"] = "growing"
        out["reasons"].append(f"amplitude still growing ({out['saturation']:.2f}x between windows); run longer for St/C_D")
    else:
        # amplitude falling: decaying towards a steady wake, unless it is large and only dipping
        out["status"] = "steady" if (a_last < 10 * STEADY_AMP or sigma < 0) else "growing"
        out["reasons"].append(f"amplitude changing ({out['saturation']:.2f}x between windows, "
                              f"growth rate {sigma:+.3f})")
    if out["status"] == "steady":
        out["st"] = 0.0
    return out


def is_unstable(status: str) -> bool | None:
    """True if the wake sheds (or is starting to), False if steady, None if unknown."""
    return {"steady": False, "growing": True, "periodic": True}.get(status)


def estimate_critical_re(records) -> dict:
    """Critical Re from growth rates. Near a Hopf bifurcation sigma ~ k (Re - Re_c), so
    Re_c is where sigma crosses zero: interpolated linearly between the closest stable
    run (sigma < 0) and the closest unstable run (sigma > 0). Falls back to a straight-line
    fit through the runs with the smallest |sigma| when no such pair exists."""
    pts = [(r["re"], r["growth_rate"]) for r in records
           if r.get("status") in ("steady", "growing", "periodic") and np.isfinite(r.get("growth_rate", np.nan))]
    stable = [r["re"] for r in records if is_unstable(r.get("status")) is False]
    unstable = [r["re"] for r in records if is_unstable(r.get("status")) is True]
    out = {"bracket": [max(stable) if stable else None, min(unstable) if unstable else None],
           "re_c_growth": None, "n_points": len(pts)}
    neg = [p for p in pts if p[1] < 0]
    pos = [p for p in pts if p[1] > 0]
    if neg and pos:
        (r0, s0), (r1, s1) = max(neg), min(pos)
        if r0 < r1:
            out["re_c_growth"] = float(r0 + (r1 - r0) * (-s0) / (s1 - s0))
            return out
    if len(pts) >= 2:
        pts.sort(key=lambda p: abs(p[1]))
        sel = np.array(pts[:3])
        if np.ptp(sel[:, 0]) > 0:
            k, b = np.polyfit(sel[:, 0], sel[:, 1], 1)
            if k > 0:
                out["re_c_growth"] = float(-b / k)
    return out
