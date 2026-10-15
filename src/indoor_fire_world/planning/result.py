"""Typed route and mission planning results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


GridCell = tuple[int, int]


@dataclass(frozen=True)
class RouteResult:
    path: tuple[GridCell, ...]
    feasible: bool
    geometric_length_m: float
    total_cost: float
    base_contribution: float
    hazard_contribution: float
    uncertainty_contribution: float
    integrated_hazard: float
    integrated_uncertainty: float
    max_hazard: float
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feasible:
            if not self.path:
                raise ValueError("A feasible route must contain a path")
            values = (
                self.geometric_length_m,
                self.total_cost,
                self.base_contribution,
                self.hazard_contribution,
                self.uncertainty_contribution,
                self.integrated_hazard,
                self.integrated_uncertainty,
                self.max_hazard,
            )
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError("Feasible route metrics must be finite and non-negative")
            if self.rejection_reasons:
                raise ValueError("A feasible route cannot contain rejection reasons")
        else:
            if self.path:
                raise ValueError("An infeasible route must not contain a path")
            if not self.rejection_reasons:
                raise ValueError("An infeasible route requires an explicit reason")

    @classmethod
    def rejected(cls, *reasons: str) -> "RouteResult":
        return cls(
            path=(),
            feasible=False,
            geometric_length_m=0.0,
            total_cost=math.inf,
            base_contribution=0.0,
            hazard_contribution=0.0,
            uncertainty_contribution=0.0,
            integrated_hazard=0.0,
            integrated_uncertainty=0.0,
            max_hazard=0.0,
            rejection_reasons=tuple(reasons),
        )

    @property
    def path_hash(self) -> str | None:
        if not self.feasible:
            return None
        encoded = ";".join(f"{row},{col}" for row, col in self.path).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExitRouteEvaluation:
    exit_id: str
    route: RouteResult


@dataclass(frozen=True)
class RetreatPlan:
    route: RouteResult
    selected_exit: str | None
    candidates: tuple[ExitRouteEvaluation, ...]
    feasible: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feasible != self.route.feasible:
            raise ValueError("Retreat feasibility must match the selected route")
        if self.route.feasible != (self.selected_exit is not None):
            raise ValueError("A selected exit is required exactly when retreat is feasible")
        if self.feasible and self.rejection_reasons:
            raise ValueError("A feasible retreat cannot contain rejection reasons")
        if not self.feasible and not self.rejection_reasons:
            raise ValueError("An infeasible retreat requires explicit reasons")


@dataclass(frozen=True)
class MissionPlan:
    entry_route: RouteResult
    retreat_route: RouteResult
    selected_exit: str | None
    retreat_candidates: tuple[ExitRouteEvaluation, ...]
    feasible: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = self.entry_route.feasible and self.retreat_route.feasible
        if self.feasible != expected:
            raise ValueError("Mission feasibility must require both entry and retreat")
        if self.retreat_route.feasible != (self.selected_exit is not None):
            raise ValueError("A selected exit is required exactly when retreat is feasible")
        if self.feasible and self.rejection_reasons:
            raise ValueError("A feasible mission cannot contain rejection reasons")
        if not self.feasible and not self.rejection_reasons:
            raise ValueError("An infeasible mission requires explicit reasons")
