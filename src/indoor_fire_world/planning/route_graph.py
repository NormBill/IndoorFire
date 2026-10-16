"""Directed evacuation-route graph registered from the green map arrows."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from indoor_fire_world.scene.models import NavigationLayer, Scene

from .astar import GridAStarPlanner
from .cost_field import PlanningCostParameters
from .result import (
    EntranceRouteEvaluation,
    ExitRouteEvaluation,
    FirefighterIngressPlan,
    GridCell,
    RetreatPlan,
    RouteResult,
)


@dataclass(frozen=True)
class DirectedRouteNode:
    id: str
    cell: GridCell
    position_m: tuple[float, float]


@dataclass(frozen=True)
class DirectedRouteEdge:
    id: str
    source: str
    target: str
    path: tuple[GridCell, ...]


@dataclass(frozen=True)
class DirectedEvacuationGraph:
    graph_id: str
    scene_id: str
    direction_semantics: str
    registration_status: str
    source_evidence: str
    notes: str
    nodes: Mapping[str, DirectedRouteNode]
    edges: tuple[DirectedRouteEdge, ...]

    def __post_init__(self) -> None:
        if not self.graph_id or not self.scene_id:
            raise ValueError("Directed route graph IDs must not be empty")
        if not self.nodes or not self.edges:
            raise ValueError("Directed route graph requires nodes and edges")
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"Directed route edge {edge.id!r} has an unknown endpoint")
            if len(edge.path) < 2:
                raise ValueError(f"Directed route edge {edge.id!r} must contain movement")
            if edge.path[0] != self.nodes[edge.source].cell:
                raise ValueError(f"Directed route edge {edge.id!r} source cell mismatch")
            if edge.path[-1] != self.nodes[edge.target].cell:
                raise ValueError(f"Directed route edge {edge.id!r} target cell mismatch")

    def outgoing(self, node_id: str) -> tuple[DirectedRouteEdge, ...]:
        return tuple(sorted((edge for edge in self.edges if edge.source == node_id), key=lambda edge: edge.id))

    def corridor_edges(self, node_id: str) -> tuple[DirectedRouteEdge, ...]:
        """Return both orientations of every physical green-map corridor."""
        result: list[DirectedRouteEdge] = []
        for edge in self.edges:
            if edge.source == node_id:
                result.append(edge)
            elif edge.target == node_id:
                result.append(
                    DirectedRouteEdge(
                        id=f"{edge.id}:reverse",
                        source=edge.target,
                        target=edge.source,
                        path=tuple(reversed(edge.path)),
                    )
                )
        return tuple(sorted(result, key=lambda item: item.id))

    @property
    def allowed_transitions(self) -> frozenset[tuple[GridCell, GridCell]]:
        return frozenset(
            (source, target)
            for edge in self.edges
            for source, target in zip(edge.path, edge.path[1:])
        )

    def path_is_directed(self, path: tuple[GridCell, ...]) -> bool:
        return bool(path) and all(
            transition in self.allowed_transitions
            for transition in zip(path, path[1:])
        )

    def path_is_corridor_conformant(self, path: tuple[GridCell, ...]) -> bool:
        transitions = self.allowed_transitions
        bidirectional = transitions | frozenset((target, source) for source, target in transitions)
        return bool(path) and all(
            transition in bidirectional for transition in zip(path, path[1:])
        )


def _axis_path(source: GridCell, target: GridCell) -> tuple[GridCell, ...]:
    if source[0] != target[0] and source[1] != target[1]:
        raise ValueError("Directed route edges must be axis-aligned on the registered grid")
    row_step = 0 if source[0] == target[0] else (1 if target[0] > source[0] else -1)
    col_step = 0 if source[1] == target[1] else (1 if target[1] > source[1] else -1)
    current = source
    path = [current]
    while current != target:
        current = (current[0] + row_step, current[1] + col_step)
        path.append(current)
    return tuple(path)


def load_directed_evacuation_graph(
    config_path: str | Path,
    scene: Scene,
) -> DirectedEvacuationGraph:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported directed evacuation-route schema_version")
    if str(data["scene_id"]) != scene.scene_id:
        raise ValueError("Directed route graph scene_id does not match the loaded scene")
    nodes: dict[str, DirectedRouteNode] = {}
    for item in data["nodes"]:
        node_id = str(item["id"])
        if node_id in nodes:
            raise ValueError(f"Duplicate directed route node {node_id!r}")
        if "semantic_id" in item:
            semantic = scene.semantic.by_id(str(item["semantic_id"]))
            if semantic.position is None:
                raise ValueError(f"Route node semantic {semantic.id!r} has no position")
            position = semantic.position
        else:
            values = item["position_m"]
            if len(values) != 2:
                raise ValueError("Directed route position_m must contain [x, y]")
            position = (float(values[0]), float(values[1]))
        if not all(math.isfinite(value) for value in position):
            raise ValueError("Directed route positions must be finite")
        cell = scene.navigation.metric_to_grid(*position)
        if not scene.navigation.is_traversable(cell):
            raise ValueError(f"Directed route node {node_id!r} is not traversable")
        nodes[node_id] = DirectedRouteNode(node_id, cell, position)
    edges: list[DirectedRouteEdge] = []
    edge_ids: set[str] = set()
    for item in data["edges"]:
        edge_id = str(item["id"])
        source = str(item["source"])
        target = str(item["target"])
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate directed route edge {edge_id!r}")
        if source not in nodes or target not in nodes:
            raise ValueError(f"Directed route edge {edge_id!r} has an unknown endpoint")
        cell_path = _axis_path(nodes[source].cell, nodes[target].cell)
        if not all(scene.navigation.is_traversable(cell) for cell in cell_path):
            raise ValueError(f"Directed route edge {edge_id!r} crosses a non-traversable cell")
        edges.append(DirectedRouteEdge(edge_id, source, target, cell_path))
        edge_ids.add(edge_id)
    return DirectedEvacuationGraph(
        graph_id=str(data["route_graph_id"]),
        scene_id=str(data["scene_id"]),
        direction_semantics=str(data["direction_semantics"]),
        registration_status=str(data["registration_status"]),
        source_evidence=str(data["source_evidence"]),
        notes=str(data.get("notes", "")),
        nodes=MappingProxyType(nodes),
        edges=tuple(edges),
    )


class DirectedRoutePlanner:
    def __init__(self, parameters: PlanningCostParameters, graph: DirectedEvacuationGraph) -> None:
        self.parameters = parameters
        self.graph = graph
        self._path_evaluator = GridAStarPlanner(parameters)

    def plan(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        start_node: str,
        goal_node: str,
        navigation: NavigationLayer,
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
        *,
        allow_reverse: bool = False,
    ) -> RouteResult:
        if start_node not in self.graph.nodes:
            return RouteResult.rejected("start_node_not_in_directed_route_graph")
        if goal_node not in self.graph.nodes:
            return RouteResult.rejected("goal_node_not_in_directed_route_graph")
        cost = np.asarray(planning_cost_field, dtype=np.float64)
        forbidden = np.asarray(forbidden_mask, dtype=np.bool_)
        start_cell = self.graph.nodes[start_node].cell
        goal_cell = self.graph.nodes[goal_node].cell
        if forbidden[start_cell] or not math.isfinite(float(cost[start_cell])):
            return RouteResult.rejected("start_forbidden")
        if forbidden[goal_cell] or not math.isfinite(float(cost[goal_cell])):
            return RouteResult.rejected("goal_forbidden")
        distances: dict[str, float] = {start_node: 0.0}
        previous: dict[str, tuple[str, DirectedRouteEdge]] = {}
        frontier: list[tuple[float, str]] = [(0.0, start_node)]
        while frontier:
            distance, node_id = heapq.heappop(frontier)
            if distance > distances.get(node_id, math.inf) + 1e-12:
                continue
            if node_id == goal_node:
                break
            available = (
                self.graph.corridor_edges(node_id)
                if allow_reverse
                else self.graph.outgoing(node_id)
            )
            for edge in available:
                entered = edge.path[1:]
                if any(forbidden[cell] or not math.isfinite(float(cost[cell])) for cell in entered):
                    continue
                edge_cost = sum(float(cost[cell]) for cell in entered) * navigation.occupancy.geometry.resolution_m
                candidate = distance + edge_cost
                current = distances.get(edge.target, math.inf)
                if candidate + 1e-12 < current:
                    distances[edge.target] = candidate
                    previous[edge.target] = (node_id, edge)
                    heapq.heappush(frontier, (candidate, edge.target))
        if goal_node not in distances:
            return RouteResult.rejected("goal_unreachable_on_directed_route_graph")
        edges: list[DirectedRouteEdge] = []
        current_node = goal_node
        while current_node != start_node:
            prior_node, edge = previous[current_node]
            edges.append(edge)
            current_node = prior_node
        path: list[GridCell] = [start_cell]
        for edge in reversed(edges):
            path.extend(edge.path[1:])
        route = self._path_evaluator.evaluate_path(
            tuple(path),
            planning_cost_field,
            forbidden_mask,
            navigation,
            hazard_field,
            uncertainty_field,
        )
        if route.feasible:
            conformant = (
                self.graph.path_is_corridor_conformant(route.path)
                if allow_reverse
                else self.graph.path_is_directed(route.path)
            )
            if not conformant:
                raise RuntimeError("Route planner emitted a non-conforming corridor path")
        return route

    def plan_ingress(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
        navigation: NavigationLayer,
        entrances: tuple[str, ...],
        target_node: str,
    ) -> FirefighterIngressPlan:
        candidates = tuple(
            EntranceRouteEvaluation(
                entrance_id=entrance_id,
                route=self.plan(
                    planning_cost_field,
                    forbidden_mask,
                    entrance_id,
                    target_node,
                    navigation,
                    hazard_field,
                    uncertainty_field,
                    allow_reverse=True,
                ),
            )
            for entrance_id in sorted(entrances)
        )
        feasible = [candidate for candidate in candidates if candidate.route.feasible]
        if feasible:
            selected = min(
                feasible,
                key=lambda candidate: (
                    candidate.route.total_cost,
                    candidate.entrance_id,
                    candidate.route.path_hash or "",
                ),
            )
            return FirefighterIngressPlan(
                selected.route,
                selected.entrance_id,
                target_node,
                candidates,
                True,
            )
        return FirefighterIngressPlan(
            RouteResult.rejected("no_feasible_firefighter_ingress"),
            None,
            target_node,
            candidates,
            False,
            ("no_feasible_firefighter_ingress",),
        )

    def plan_retreat(
        self,
        planning_cost_field: NDArray[np.floating],
        forbidden_mask: NDArray[np.bool_],
        hazard_field: NDArray[np.floating],
        uncertainty_field: NDArray[np.floating],
        navigation: NavigationLayer,
        start_node: str,
        exits: tuple[str, ...],
    ) -> RetreatPlan:
        candidates = tuple(
            ExitRouteEvaluation(
                exit_id=exit_id,
                route=self.plan(
                    planning_cost_field,
                    forbidden_mask,
                    start_node,
                    exit_id,
                    navigation,
                    hazard_field,
                    uncertainty_field,
                ),
            )
            for exit_id in sorted(exits)
        )
        feasible = [candidate for candidate in candidates if candidate.route.feasible]
        if feasible:
            selected = min(feasible, key=lambda candidate: (candidate.route.total_cost, candidate.exit_id))
            return RetreatPlan(selected.route, selected.exit_id, candidates, True)
        route = RouteResult.rejected("no_feasible_exit_on_directed_route_graph")
        return RetreatPlan(
            route,
            None,
            candidates,
            False,
            ("retreat_infeasible", "retreat:no_feasible_exit_on_directed_route_graph"),
        )
