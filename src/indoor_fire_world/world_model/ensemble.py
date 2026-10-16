"""Deterministic reduced-order parameter ensemble used by P8 v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.fire import FireField, FireParameters, ReducedOrderFireSimulator
from indoor_fire_world.p3_config import SensorNoiseParameters
from indoor_fire_world.scene.models import Scene
from indoor_fire_world.sensing import Observation

from .belief import BeliefMean
from .uncertainty import UncertaintyField


MODALITIES = ("temperature", "co", "smoke", "visibility")


@dataclass(frozen=True)
class EnsembleCandidate:
    id: str
    parameters: FireParameters
    hrr_factor: float
    yield_factor: float
    ventilation_factor: float


@dataclass(frozen=True)
class WeightedForecast:
    mean: BeliefMean
    uncertainty: UncertaintyField


def _scale_parameters(base: FireParameters, hrr: float, yield_factor: float, ventilation: float) -> FireParameters:
    phases = tuple(
        replace(
            phase,
            initial_heat_release_rate_kw=phase.initial_heat_release_rate_kw * hrr,
            fire_growth_rate_kw_per_s2=phase.fire_growth_rate_kw_per_s2 * hrr,
            max_heat_release_rate_kw=phase.max_heat_release_rate_kw * hrr,
            smoke_yield_per_kw_s=phase.smoke_yield_per_kw_s * yield_factor,
            co_yield_per_kw_s=phase.co_yield_per_kw_s * yield_factor,
        )
        for phase in base.source_phases
    )
    return replace(
        base,
        model_name=f"{base.model_name}:h{hrr:g}:y{yield_factor:g}:v{ventilation:g}",
        ventilation_x_mps=base.ventilation_x_mps * ventilation,
        ventilation_y_mps=base.ventilation_y_mps * ventilation,
        fire_growth_rate_kw_per_s2=base.fire_growth_rate_kw_per_s2 * hrr,
        max_heat_release_rate_kw=base.max_heat_release_rate_kw * hrr,
        smoke_yield_per_kw_s=base.smoke_yield_per_kw_s * yield_factor,
        co_yield_per_kw_s=base.co_yield_per_kw_s * yield_factor,
        source_phases=phases,
        assumption_status="simulation_assumed_candidate",
    )


def build_candidates(base: FireParameters, hrr_factors: Iterable[float], yield_factors: Iterable[float], ventilation_factors: Iterable[float]) -> tuple[EnsembleCandidate, ...]:
    return tuple(
        EnsembleCandidate(
            f"h{hrr:g}-y{yield_factor:g}-v{ventilation:g}",
            _scale_parameters(base, hrr, yield_factor, ventilation),
            hrr,
            yield_factor,
            ventilation,
        )
        for hrr in hrr_factors
        for yield_factor in yield_factors
        for ventilation in ventilation_factors
    )


def simulate_candidate_bank(scene: Scene, candidates: tuple[EnsembleCandidate, ...], times_s: tuple[float, ...]) -> dict[str, dict[float, FireField]]:
    requested = tuple(sorted(set(times_s)))
    bank: dict[str, dict[float, FireField]] = {}
    for candidate in candidates:
        simulator = ReducedOrderFireSimulator(scene, candidate.parameters)
        snapshots: dict[float, FireField] = {}
        for target in requested:
            while simulator.current_state.timestamp < target - 1e-12:
                simulator.step(min(simulator.parameters.time_step_s, target - simulator.current_state.timestamp))
            snapshots[target] = simulator.current_state
        bank[candidate.id] = snapshots
    return bank


def posterior_weights(candidates: tuple[EnsembleCandidate, ...], bank: dict[str, dict[float, FireField]], observation: Observation, cell: tuple[int, int], noise: SensorNoiseParameters, modalities: tuple[str, ...] = ("temperature", "co", "smoke")) -> NDArray[np.float64]:
    log_weights = np.empty(len(candidates), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        state = bank[candidate.id][observation.timestamp]
        residual = 0.0
        for modality in modalities:
            observed = float(getattr(observation, modality))
            predicted = float(getattr(state, modality)[cell])
            sigma = float(getattr(noise, f"{modality}_std"))
            if sigma <= 0.0:
                raise ValueError("P8 v2 likelihood requires non-zero sensor errors")
            residual += ((observed - predicted) / sigma) ** 2
        log_weights[index] = -0.5 * residual
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    return weights / total


def weighted_forecast(candidates: tuple[EnsembleCandidate, ...], bank: dict[str, dict[float, FireField]], time_s: float, weights: NDArray[np.floating]) -> WeightedForecast:
    probabilities = np.asarray(weights, dtype=np.float64)
    if probabilities.shape != (len(candidates),) or not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Ensemble weights must match candidates and sum to one")
    means: dict[str, NDArray[np.float64]] = {}
    variances: dict[str, NDArray[np.float64]] = {}
    for modality in MODALITIES:
        stack = np.stack([np.asarray(getattr(bank[item.id][time_s], modality)) for item in candidates])
        mean = np.tensordot(probabilities, stack, axes=(0, 0))
        means[modality] = mean
        variances[modality] = np.tensordot(probabilities, (stack - mean) ** 2, axes=(0, 0))
    return WeightedForecast(
        BeliefMean(**means),
        UncertaintyField(**variances, meaning="weighted reduced-order candidate variance in squared native units"),
    )


def weight_diagnostics(weights: NDArray[np.floating]) -> dict[str, float]:
    values = np.asarray(weights, dtype=np.float64)
    positive = values[values > 0.0]
    return {
        "entropy": -float(np.sum(positive * np.log(positive))),
        "effective_sample_size": float(1.0 / np.sum(values**2)),
        "maximum_weight": float(np.max(values)),
    }
