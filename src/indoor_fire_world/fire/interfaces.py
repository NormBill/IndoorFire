"""Protocol for consumers of hidden ground-truth simulation."""

from typing import Protocol

from .state import FireField


class HiddenGroundTruth(Protocol):
    """Simulator-owned truth, intentionally separate from a belief state."""

    @property
    def fire_field(self) -> FireField: ...

    @property
    def current_state(self) -> FireField: ...

    def step(self, dt: float | None = None) -> FireField: ...
