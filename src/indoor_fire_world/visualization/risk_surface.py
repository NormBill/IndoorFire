"""Display-only completion of categorical risk outside the simulated air domain."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


def complete_blocked_risk_levels(
    risk_grid: NDArray[np.integer],
) -> NDArray[np.int8]:
    """Fill negative cells from nearest air cells, taking the worst level on ties.

    This completion is for continuous map display only.  It must never replace
    the occupancy mask used by transport or route planning.
    """
    raw = np.asarray(risk_grid, dtype=np.int8)
    if raw.ndim != 2:
        raise ValueError("Risk-level completion requires a 2D grid")
    valid = raw >= 0
    if not np.any(valid):
        raise ValueError("Risk-level completion requires at least one air-domain cell")
    rows, cols = raw.shape
    distance = np.full(raw.shape, -1, dtype=np.int32)
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
    completed = np.array(raw, copy=True)
    for current_distance in range(1, int(distance.max()) + 1):
        for row, col in np.argwhere(distance == current_distance):
            neighbours = [
                int(completed[next_row, next_col])
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
            completed[row, col] = max(neighbours)
    if np.any(completed < 0):
        raise RuntimeError("Risk-level display completion left unfilled cells")
    completed.setflags(write=False)
    return completed
