# Licensed under the Apache License, Version 2.0.
"""Common benchmark metrics."""

from __future__ import annotations

def maxcut_approximation(best_cut:float,opt_cut:float)->float:
    """Return the MaxCut approximation ratio relative to the exact optimum."""
    if opt_cut<=0: return 1.0 if best_cut>=opt_cut else 0.0
    return float(best_cut/opt_cut)
