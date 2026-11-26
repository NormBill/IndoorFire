"""Configuration contract for P9 post-arrival intervention experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from indoor_fire_world.intervention import InterventionAction


@dataclass(frozen=True)
class P9Parameters:
    config_path: Path
    experiment_id: str
    p8_config: Path
    firefighter_availability_time_s: float
    evaluation_end_time_s: float
    evaluation_times_s: tuple[float, ...]
    post_action_observation_time_s: float
    intervention_staging_node: str
    staging_semantics: str
    safe_staging_candidate_nodes: tuple[str, ...]
    safe_staging_selection: str
    firefighter_travel_speed_m_s: float
    actions: tuple[InterventionAction, ...]
    assumption_status: str
    scope_exclusions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id or self.assumption_status != "simulation_assumed":
            raise ValueError("P9 requires an experiment ID and simulation_assumed status")
        if self.firefighter_availability_time_s != 300.0:
            raise ValueError("P9 v1 freezes firefighter availability at the 300 s handoff")
        if self.evaluation_end_time_s != 480.0:
            raise ValueError("P9 v1 evaluation must end at 480 s")
        if self.evaluation_times_s != (300.0, 360.0, 420.0, 480.0):
            raise ValueError("P9 v1 evaluation times must be 300/360/420/480 s")
        if self.post_action_observation_time_s != 420.0:
            raise ValueError("P9 v1 post-action observation time must be 420 s")
        if not self.intervention_staging_node or not self.staging_semantics:
            raise ValueError("P9 requires an explicit intervention staging node")
        if (
            not self.safe_staging_candidate_nodes
            or len(set(self.safe_staging_candidate_nodes))
            != len(self.safe_staging_candidate_nodes)
        ):
            raise ValueError("P9 requires unique source-near safe-staging candidates")
        if not self.safe_staging_selection:
            raise ValueError("P9 requires explicit safe-staging selection semantics")
        if not math.isfinite(self.firefighter_travel_speed_m_s) or (
            self.firefighter_travel_speed_m_s <= 0.0
        ):
            raise ValueError("P9 firefighter travel speed must be positive and finite")
        if len(self.actions) != 4 or len({item.id for item in self.actions}) != 4:
            raise ValueError("P9 v1 requires four unique action branches")
        if sum(item.action_type == "no_intervention" for item in self.actions) != 1:
            raise ValueError("P9 requires exactly one no-intervention branch")
        for action in self.actions:
            if action.firefighter_available_time_s != self.firefighter_availability_time_s:
                raise ValueError("All P9 actions must share the configured availability time")
            if action.activation_time_s is not None and action.activation_time_s > 360.0:
                raise ValueError("P9 v1 interventions must activate no later than 360 s")
        if not self.scope_exclusions:
            raise ValueError("P9 must serialize its simulation-only scope exclusions")


def load_p9_parameters(config_path: str | Path) -> P9Parameters:
    path = Path(config_path).resolve()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported P9 schema_version")
    availability = float(data["firefighter_availability_time_s"])
    if not math.isfinite(availability):
        raise ValueError("P9 firefighter availability must be finite")
    actions = tuple(
        InterventionAction(
            id=str(item["id"]),
            action_type=str(item["action_type"]),
            firefighter_available_time_s=availability,
            activation_time_s=(
                None
                if item["activation_time_s"] is None
                else float(item["activation_time_s"])
            ),
            suppression_rate_per_s=float(item["suppression_rate_per_s"]),
            residual_hrr_fraction=float(item["residual_hrr_fraction"]),
        )
        for item in data["actions"]
    )
    return P9Parameters(
        config_path=path,
        experiment_id=str(data["experiment_id"]),
        p8_config=(path.parent / str(data["p8_config"])).resolve(),
        firefighter_availability_time_s=availability,
        evaluation_end_time_s=float(data["evaluation_end_time_s"]),
        evaluation_times_s=tuple(float(value) for value in data["evaluation_times_s"]),
        post_action_observation_time_s=float(data["post_action_observation_time_s"]),
        intervention_staging_node=str(data["intervention_staging_node"]),
        staging_semantics=str(data["staging_semantics"]),
        safe_staging_candidate_nodes=tuple(
            str(value) for value in data["safe_staging_candidate_nodes"]
        ),
        safe_staging_selection=str(data["safe_staging_selection"]),
        firefighter_travel_speed_m_s=float(data["firefighter_travel_speed_m_s"]),
        actions=actions,
        assumption_status=str(data["assumption_status"]),
        scope_exclusions=tuple(str(value) for value in data["scope_exclusions"]),
    )

