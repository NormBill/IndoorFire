"""Immutable scene data and the three independent spatial layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path
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
class Polygon:
    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least three vertices")
        if not all(math.isfinite(value) for point in self.vertices for value in point):
            raise ValueError("Polygon vertices must be finite")

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point is inside using deterministic ray casting."""
        inside = False
        previous = self.vertices[-1]
        for current in self.vertices:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside


@dataclass(frozen=True)
class MapRegistration:
    source_image: Path
    source_width_px: int
    source_height_px: int
    floor_roi_px: tuple[float, float, float, float]
    metric_bounds_m: tuple[float, float, float, float]
    calibration_status: str
    scale_reference: str

    def __post_init__(self) -> None:
        if self.source_width_px <= 0 or self.source_height_px <= 0:
            raise ValueError("Reference image dimensions must be positive")
        left, top, width, height = self.floor_roi_px
        if width <= 0 or height <= 0 or left < 0 or top < 0:
            raise ValueError("Reference floor ROI must be a positive pixel rectangle")
        if left + width > self.source_width_px or top + height > self.source_height_px:
            raise ValueError("Reference floor ROI must fit inside the source image")
        x, y, width_m, height_m = self.metric_bounds_m
        if width_m <= 0 or height_m <= 0 or not all(
            math.isfinite(value) for value in (x, y, width_m, height_m)
        ):
            raise ValueError("Reference metric bounds must be finite and positive")
        if not self.calibration_status:
            raise ValueError("Reference calibration status must not be empty")

    def pixel_to_metric(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        left, top, width, height = self.floor_roi_px
        x, y, width_m, height_m = self.metric_bounds_m
        return (
            x + (pixel_x - left) * width_m / width,
            y + (top + height - pixel_y) * height_m / height,
        )

    def metric_to_pixel(self, metric_x: float, metric_y: float) -> tuple[float, float]:
        left, top, width, height = self.floor_roi_px
        x, y, width_m, height_m = self.metric_bounds_m
        return (
            left + (metric_x - x) * width / width_m,
            top + height - (metric_y - y) * height / height_m,
        )


@dataclass(frozen=True)
class SemanticObject:
    id: str
    kind: str
    position: tuple[float, float] | None = None
    footprint: Rectangle | Polygon | None = None


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
    map_registration: MapRegistration | None = None

