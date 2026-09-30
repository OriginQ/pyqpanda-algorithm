# Licensed under the Apache License, Version 2.0.
"""Smooth interpolation schedule used by BF-DCQO."""
from __future__ import annotations

import numpy as np


def schedule_lambda(t: float, total_time: float) -> float:
    """Evaluate the smooth BF-DCQO interpolation schedule ``lambda(t)``."""
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    x = np.pi * t / (2.0 * total_time)
    return float(np.sin(0.5 * np.pi * np.sin(x) ** 2) ** 2)


def schedule_lambda_dot(t: float, total_time: float) -> float:
    """Evaluate the analytic time derivative of :func:`schedule_lambda`."""
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    x = np.pi * t / (2.0 * total_time)
    return float(
        (np.pi**2 / (4.0 * total_time))
        * np.sin(np.pi * np.sin(x) ** 2)
        * np.sin(np.pi * t / total_time)
    )
