"""Deterministically construct a scene from authoritative structured config."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .models import (
    GridGeometry,
    NavigationLayer,
    Occupancy,
    OccupancyLayer,
    Pose2D,
    Rectangle,
    Scene,
    SemanticLayer,
    SemanticObject,
)


def _rectangle(values: list[float]) -> Rectangle:
    if len(values) != 4:
        raise ValueError("rect_m must contain [x, y, width, height]")
    rect = Rectangle(*(float(value) for value in values))
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("Rectangle dimensions must be positive")
    return rect


def _point(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError("position_m must contain [x, y]")
    return float(values[0]), float(values[1])


def _validate_point(geometry: GridGeometry, point: tuple[float, float], label: str) -> None:
    try:
        geometry.metric_to_grid(*point)
    except ValueError as exc:
        raise ValueError(f"{label} is outside map bounds") from exc


def load_scene(config_path: str | Path) -> Scene:
    """Load a new immutable scene; config ordering and results are deterministic."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)

    map_cfg = data["map"]
    origin = map_cfg.get("origin_m", [0.0, 0.0])
    geometry = GridGeometry(
        width_m=float(map_cfg["width_m"]),
        height_m=float(map_cfg["height_m"]),
        resolution_m=float(map_cfg["resolution_m"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )
    default = Occupancy[map_cfg.get("default_occupancy", "FREE")]
    cells = [[default for _ in range(geometry.cols)] for _ in range(geometry.rows)]
    semantic_objects: list[SemanticObject] = []

    def paint(rect: Rectangle, value: Occupancy) -> None:
        for row in range(geometry.rows):
            for col in range(geometry.cols):
                x, y = geometry.grid_to_metric(row, col)
                if rect.contains(x, y):
                    cells[row][col] = value

    for obstacle in data.get("obstacles", []):
        rect = _rectangle(obstacle["rect_m"])
        paint(rect, Occupancy.OCCUPIED)
        semantic_objects.append(SemanticObject(obstacle["id"], obstacle["type"], footprint=rect))

    for index, unknown in enumerate(data.get("unknown_regions", [])):
        rect = _rectangle(unknown["rect_m"])
        paint(rect, Occupancy.UNKNOWN)
        semantic_objects.append(SemanticObject(unknown.get("id", f"unknown_{index}"), "unknown_region", footprint=rect))

    for item in data.get("semantic_objects", []):
        position = _point(item["position_m"])
        _validate_point(geometry, position, item["id"])
        semantic_objects.append(SemanticObject(item["id"], item["type"], position=position))

    for collection, kind in (("exits", "exit"), ("fixed_sensors", "fixed_sensor")):
        for item in data.get(collection, []):
            position = _point(item["position_m"])
            _validate_point(geometry, position, item["id"])
            semantic_objects.append(SemanticObject(item["id"], kind, position=position))

    microwave = data["microwave_ignition"]
    microwave_position = _point(microwave["position_m"])
    _validate_point(geometry, microwave_position, microwave["id"])
    semantic_objects.append(SemanticObject(microwave["id"], "microwave", position=microwave_position))

    pose_cfg = data["ugv_initial_pose"]
    pose = Pose2D(float(pose_cfg["x_m"]), float(pose_cfg["y_m"]), float(pose_cfg["yaw_rad"]))
    _validate_point(geometry, (pose.x, pose.y), "ugv_initial_pose")
    semantic_objects.append(SemanticObject("ugv_start", "ugv_start", position=(pose.x, pose.y)))

    occupancy = OccupancyLayer(geometry, tuple(tuple(row) for row in cells))
    navigation = NavigationLayer(occupancy)
    if not navigation.is_traversable(navigation.metric_to_grid(pose.x, pose.y)):
        raise ValueError("UGV initial pose must be in a FREE cell")

    scene_cfg = data["scene"]
    return Scene(
        scene_id=scene_cfg["id"],
        name=scene_cfg["name"],
        geometry_status=scene_cfg.get("geometry_status", "unspecified"),
        notes=scene_cfg.get("notes", ""),
        semantic=SemanticLayer(tuple(semantic_objects)),
        occupancy=occupancy,
        navigation=navigation,
        ugv_initial_pose=pose,
        raw_config=MappingProxyType(data),
    )

