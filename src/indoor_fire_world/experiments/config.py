"""Immutable P7 experiment-suite configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    scenario_id: str
    ablation_type: str
    mismatch_level: str
    sensor_noise: str
    seed: int
    observation_enabled: bool
    assimilation_enabled: bool
    oracle_planning: bool
    overrides: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.ablation_type:
            raise ValueError("Experiment scenario and ablation IDs must not be empty")
        if self.seed < 0:
            raise ValueError("Experiment seed must be non-negative")
        allowed = {"truth", "sensor", "assimilation", "planning"}
        unsupported = set(self.overrides) - allowed
        if unsupported:
            raise ValueError(f"Unsupported experiment override groups: {unsupported}")
        if self.assimilation_enabled and not self.observation_enabled:
            raise ValueError("Assimilation cannot be enabled without an observation")


@dataclass(frozen=True)
class ExperimentSuite:
    suite_id: str
    base_config: Path
    experiments: tuple[ExperimentSpec, ...]
    source_config: Path

    def __post_init__(self) -> None:
        ids = [experiment.scenario_id for experiment in self.experiments]
        if not self.suite_id or not ids or len(ids) != len(set(ids)):
            raise ValueError("Suite and unique experiment IDs are required")


def load_experiment_suite(config_path: str | Path) -> ExperimentSuite:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported P7 experiment schema_version")
    experiments = tuple(
        ExperimentSpec(
            scenario_id=str(item["scenario_id"]),
            ablation_type=str(item["ablation_type"]),
            mismatch_level=str(item["mismatch_level"]),
            sensor_noise=str(item["sensor_noise"]),
            seed=int(item["seed"]),
            observation_enabled=bool(item["observation_enabled"]),
            assimilation_enabled=bool(item["assimilation_enabled"]),
            oracle_planning=bool(item["oracle_planning"]),
            overrides={
                str(group): dict(values)
                for group, values in dict(item.get("overrides", {})).items()
            },
        )
        for item in data["experiments"]
    )
    return ExperimentSuite(
        suite_id=str(data["suite_id"]),
        base_config=(path.parent / str(data["base_config"])).resolve(),
        experiments=experiments,
        source_config=path,
    )
