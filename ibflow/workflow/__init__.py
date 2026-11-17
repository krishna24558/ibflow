"""Simulation automation: cached runs, automatic quality checks, parallel sweeps.

The sweep CLI lives in ``ibflow.workflow.sweep`` (``python -m ibflow.workflow.sweep``).
"""
from .experiments import CylinderConfig, PRESETS, get_config, run_cylinder, summary
from .diagnostics import analyze_forces, is_unstable, estimate_critical_re
