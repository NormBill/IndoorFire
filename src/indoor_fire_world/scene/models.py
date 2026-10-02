"""Immutable scene data and the three independent spatial layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Iterator, Mapping


class Occupancy(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class GridGeometry:
    width_m: float
    height_m: float
    resolution_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0 or self.resolution_m <= 0:
            raise ValueError("Map dimensions and resolution must be positive")
        for extent in (self.width_m, self.height_m):
            cells = extent / self.resolution_m
            if not math.isclose(cells, round(cells), abs_tol=1e-9):
                raise ValueError("Map dimensions must be integer multiples of resolution")

    @property
    def rows(self) -> int:
        return round(self.height_m / self.resolution_m)

    @property
    def cols(self) -> int:
        return round(self.width_m / self.resolution_m)

    def metric_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert metric coordinates to a zero-based ``(row, col)`` cell."""
        col = math.floor((x - self.origin_x) / self.resolution_m)
        row = math.floor((y - self.origin_y) / self.resolution_m)
        cell = (row, col)
        if not self.contains_cell(cell):
            raise ValueError(f"Metric position ({x}, {y}) is outside the map")
        return cell

    def grid_to_metric(self, row: int, col: int) -> tuple[float, float]:
        """Return the metric coordinates of a grid cell's center."""
        if not self.contains_cell((row, col)):
            raise ValueError(f"Grid cell ({row}, {col}) is outside the map")
        return (
            self.origin_x + (col + 0.5) * self.resolution_m,
            self.origin_y + (row + 0.5) * self.resolution_m,
        )

    def contains_cell(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen=True)
class SemanticObject:
    id: str
    kind: str
    position: tuple[float, float] | None = None
    footprint: Rectangle | None = None


@dataclass(frozen=True)
class SemanticLayer:
    objects: tuple[SemanticObject, ...]

    def by_kind(self, kind: str) -> tuple[SemanticObject, ...]:
        return tuple(obj for obj in self.objects if obj.kind == kind)

    def by_id(self, object_id: str) -> SemanticObject:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(object_id)


@dataclass(frozen=True)
class OccupancyLayer:
    geometry: GridGeometry
    cells: tuple[tuple[Occupancy, ...], ...]

    def __post_init__(self) -> None:
        if len(self.cells) != self.geometry.rows:
            raise ValueError("Occupancy row count does not match geometry")
        if any(len(row) != self.geometry.cols for row in self.cells):
            raise ValueError("Occupancy column count does not match geometry")

    def at(self, cell: tuple[int, int]) -> Occupancy:
        row, col = cell
        if not self.geometry.contains_cell(cell):
            raise ValueError(f"Grid cell {cell} is outside the map")
        return self.cells[row][col]


@dataclass(frozen=True)
class NavigationLayer:
    occupancy: OccupancyLayer

    def is_traversable(self, cell: tuple[int, int]) -> bool:
        return self.occupancy.geometry.contains_cell(cell) and self.occupancy.at(cell) == Occupancy.FREE

    def metric_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return self.occupancy.geometry.metric_to_grid(x, y)

    def grid_to_metric(self, row: int, col: int) -> tuple[float, float]:
        return self.occupancy.geometry.grid_to_metric(row, col)

    def neighbors(self, cell: tuple[int, int]) -> Iterator[tuple[int, int]]:
        row, col = cell
        for candidate in ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)):
            if self.is_traversable(candidate):
                yield candidate


@dataclass(frozen=True)
class Scene:
    scene_id: str
    name: str
    geometry_status: str
    notes: str
    semantic: SemanticLayer
    occupancy: OccupancyLayer
    navigation: NavigationLayer
    ugv_initial_pose: Pose2D
    raw_config: Mapping[str, Any]

