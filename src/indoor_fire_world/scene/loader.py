"""Deterministically construct a scene from authoritative structured config."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from .models import (
    GridGeometry,
    MapRegistration,
    NavigationLayer,
    Occupancy,
    OccupancyLayer,
    Pose2D,
    Polygon,
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


def _polygon(values: list[list[float]]) -> Polygon:
    return Polygon(tuple(_point(point) for point in values))


def _footprint(item: dict[str, Any]) -> Rectangle | Polygon:
    if "rect_m" in item:
        return _rectangle(item["rect_m"])
    if "polygon_m" in item:
        return _polygon(item["polygon_m"])
    raise ValueError("Spatial region requires rect_m or polygon_m")


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Reference map must be a PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def _map_registration(path: Path, data: dict[str, Any]) -> MapRegistration | None:
    config = data.get("reference_map")
    if config is None:
        return None
    source = (path.parent / str(config["source_image"])).resolve()
    if not source.is_file():
        raise ValueError(f"Reference map image does not exist: {source}")
    configured_size = tuple(int(value) for value in config["source_size_px"])
    if len(configured_size) != 2 or configured_size != _png_dimensions(source):
        raise ValueError("Configured reference image size does not match the PNG")
    roi = tuple(float(value) for value in config["floor_roi_px"])
    bounds = tuple(float(value) for value in config["metric_bounds_m"])
    if len(roi) != 4 or len(bounds) != 4:
        raise ValueError("Reference ROI and metric bounds must each contain four values")
    return MapRegistration(
        source_image=source,
        source_width_px=configured_size[0],
        source_height_px=configured_size[1],
        floor_roi_px=roi,
        metric_bounds_m=bounds,
        calibration_status=str(config["calibration_status"]),
        scale_reference=str(config.get("scale_reference", "")),
    )


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
    registration = _map_registration(path, data)
    if registration is not None:
        registered_bounds = registration.metric_bounds_m
        geometry_bounds = (
            geometry.origin_x,
            geometry.origin_y,
            geometry.width_m,
            geometry.height_m,
        )
        if registered_bounds != geometry_bounds:
            raise ValueError("Reference metric bounds must match map geometry")
    default = Occupancy[map_cfg.get("default_occupancy", "FREE")]
    cells = [[default for _ in range(geometry.cols)] for _ in range(geometry.rows)]
    semantic_objects: list[SemanticObject] = []

    def paint(footprint: Rectangle | Polygon, value: Occupancy) -> None:
        for row in range(geometry.rows):
            for col in range(geometry.cols):
                x, y = geometry.grid_to_metric(row, col)
                if footprint.contains(x, y):
                    cells[row][col] = value

    for region in data.get("free_space", []):
        paint(_footprint(region), Occupancy.FREE)

    for obstacle in data.get("obstacles", []):
        footprint = _footprint(obstacle)
        paint(footprint, Occupancy.OCCUPIED)
        semantic_objects.append(
            SemanticObject(obstacle["id"], obstacle["type"], footprint=footprint)
        )

    # The registered evacuation arrows are authoritative evidence that these
    # narrow corridors remain traversable.  They intentionally override the
    # coarse furniture polygons digitized from the same source image.
    for corridor in data.get("traversable_overrides", []):
        paint(_footprint(corridor), Occupancy.FREE)

    for index, unknown in enumerate(data.get("unknown_regions", [])):
        footprint = _footprint(unknown)
        paint(footprint, Occupancy.UNKNOWN)
        semantic_objects.append(
            SemanticObject(
                unknown.get("id", f"unknown_{index}"),
                "unknown_region",
                footprint=footprint,
            )
        )

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
        map_registration=registration,
    )

