"""Named protocols prevent hazard, uncertainty, and cost from being conflated."""

from typing import Protocol


class _GridField(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def value_at(self, row: int, col: int) -> float: ...


class HazardField(_GridField, Protocol):
    pass


class UncertaintyField(_GridField, Protocol):
    pass


class PlanningCostField(_GridField, Protocol):
    pass

