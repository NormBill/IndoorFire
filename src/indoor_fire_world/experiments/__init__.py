"""Reproducible P7 experiment configuration, execution, and aggregation."""

from .config import ExperimentSpec, ExperimentSuite, load_experiment_suite
from .runner import (
    ExperimentResult,
    aggregate_results,
    artifact_directory_name,
    run_experiment,
    run_experiment_suite,
)
from .p8_runner import P8SuiteOutput, run_p8_suite, write_p8_artifacts
from .p8_v2_runner import P8V2SuiteOutput, run_p8_v2_suite, write_p8_v2_artifacts

__all__ = [
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentSuite",
    "aggregate_results",
    "artifact_directory_name",
    "load_experiment_suite",
    "run_experiment",
    "run_experiment_suite",
    "P8SuiteOutput",
    "run_p8_suite",
    "write_p8_artifacts",
    "P8V2SuiteOutput",
    "run_p8_v2_suite",
    "write_p8_v2_artifacts",
]
