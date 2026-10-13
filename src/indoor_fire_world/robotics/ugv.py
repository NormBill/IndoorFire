"""Minimal predefined-waypoint UGV state without path planning."""

from __future__ import annotations

import math

from indoor_fire_world.scene import Pose2D, Scene


class UGV:
    def __init__(self, scene: Scene, initial_pose: Pose2D | None = None) -> None:
        self.scene = scene
        self._pose = initial_pose if initial_pose is not None else scene.ugv_initial_pose
        self._validate_pose(self._pose)

    @property
    def pose(self) -> Pose2D:
        return self._pose

    def move_to(self, pose: Pose2D) -> None:
        """Set a validated predefined pose; this does not plan or simulate a path."""
        self._validate_pose(pose)
        self._pose = pose

    def _validate_pose(self, pose: Pose2D) -> None:
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.yaw)):
            raise ValueError("UGV pose values must be finite")
        cell = self.scene.navigation.metric_to_grid(pose.x, pose.y)
        if not self.scene.navigation.is_traversable(cell):
            raise ValueError(f"UGV pose ({pose.x}, {pose.y}) is not traversable")
