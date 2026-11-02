"""Display-only completion outside the simulated air domain."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


def complete_blocked_exposure(
    exposure_grid: NDArray[np.floating],
    planning_blocked_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Smoothly extend exposure into blocked cells without changing air cells.

    Each distance layer receives the mean of already resolved cells in the
    preceding layer.  Unlike a categorical nearest-worst fill, this does not
    replicate a threshold crossing across an entire wall or furniture block.
    The completed surface is visualization-only and must never enter metrics or
    route planning.
    """
    exposure = np.asarray(exposure_grid, dtype=np.float64)
    blocked = np.asarray(planning_blocked_mask, dtype=np.bool_)
    if exposure.ndim != 2 or blocked.shape != exposure.shape:
        raise ValueError("Exposure completion requires matching 2D grids")
    valid = ~blocked & np.isfinite(exposure)
    if not np.any(valid):
        raise ValueError("Exposure completion requires at least one air-domain cell")
    rows, cols = exposure.shape
    distance = np.full(exposure.shape, -1, dtype=np.int32)
    distance[valid] = 0
    queue: deque[tuple[int, int]] = deque(
        (int(row), int(col)) for row, col in np.argwhere(valid)
    )
    while queue:
        row, col = queue.popleft()
        next_distance = int(distance[row, col]) + 1
        for next_row, next_col in (
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col),
        ):
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and distance[next_row, next_col] < 0
            ):
                distance[next_row, next_col] = next_distance
                queue.append((next_row, next_col))
    completed = np.array(exposure, copy=True)
    for current_distance in range(1, int(distance.max()) + 1):
        for row, col in np.argwhere(distance == current_distance):
            neighbours = [
                float(completed[next_row, next_col])
                for next_row, next_col in (
                    (int(row) - 1, int(col)),
                    (int(row), int(col) - 1),
                    (int(row), int(col) + 1),
                    (int(row) + 1, int(col)),
                )
                if 0 <= next_row < rows
                and 0 <= next_col < cols
                and distance[next_row, next_col] == current_distance - 1
            ]
            completed[row, col] = float(np.mean(neighbours))
    completed = np.clip(completed, 0.0, 1.0)
    completed.setflags(write=False)
    return completed


def exposure_risk_levels(
    exposure_grid: NDArray[np.floating],
    finite_upper_bounds: tuple[float, ...],
) -> NDArray[np.int8]:
    """Classify a complete exposure surface using fixed upper bounds."""
    exposure = np.asarray(exposure_grid, dtype=np.float64)
    if exposure.ndim != 2 or not np.all(np.isfinite(exposure)):
        raise ValueError("Risk classification requires a finite 2D exposure surface")
    if tuple(sorted(set(finite_upper_bounds))) != finite_upper_bounds:
        raise ValueError("Risk upper bounds must be unique and increasing")
    levels = np.zeros(exposure.shape, dtype=np.int8)
    for upper_bound in finite_upper_bounds:
        levels += exposure > upper_bound
    levels.setflags(write=False)
    return levels
