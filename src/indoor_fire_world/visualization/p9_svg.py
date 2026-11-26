"""Deterministic registered-map visualization for P9 action branches."""

from __future__ import annotations

import hashlib
from html import escape
from pathlib import Path

import numpy as np

from indoor_fire_world.p9_config import load_p9_parameters
from indoor_fire_world.visualization.risk_surface import (
    complete_blocked_exposure,
    exposure_risk_levels,
)


def _write_svg_atomically(output: Path, content: str) -> Path:
    """Preserve the last valid SVG when a browser has the target open."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    try:
        temporary.replace(output)
    except OSError:
        if output.exists() and hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(
            temporary.read_bytes()
        ).digest():
            temporary.unlink()
        else:
            raise
    return output.resolve()
from indoor_fire_world.visualization.svg import (
    _p8_v2_categorical_cells,
    _p8_v2_compact_categorical_cells,
    _p8_v2_route_points,
    _p8_v2_source_marker,
    _p8_v2_svg_defs,
    _p8_v2_visual_context,
)


def render_p9_action_comparison_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Render t=480 action-conditioned risk levels for all four branches."""
    p9 = load_p9_parameters(config_path)
    p8, scene, registration, _, encoded = _p8_v2_visual_context(p9.p8_config)
    scale = 0.35
    panel_width = registration.source_width_px * scale
    panel_height = registration.source_height_px * scale
    title_gap = 45
    width = int(panel_width * 2)
    height = int((panel_height + title_gap) * 2 + 145)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-phase="post-arrival-intervention" '
        'data-risk-map-source="action-conditioned-posterior">',
        *_p8_v2_svg_defs(registration, encoded),
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    blocked = np.asarray(scene.occupancy.cells) != 0
    bounds = tuple(
        float(level.upper_bound)
        for level in p8.exposure_levels
        if level.upper_bound is not None
    )
    for index, action in enumerate(p9.actions):
        row, col = divmod(index, 2)
        x = col * panel_width
        y = title_gap + row * (panel_height + title_gap)
        field = np.asarray(
            arrays[f"post_update_exposure_{action.id}_t480"], dtype=np.float64
        )
        display = complete_blocked_exposure(field, blocked)
        levels = exposure_risk_levels(display, bounds)
        selected = action.id == summary["predicted_selected_action"]
        label = action.id.replace("_", " ") + (" · predicted choice" if selected else "")
        parts.append(
            f'<text x="{x+panel_width/2:.1f}" y="{y-11:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="17" font-weight="bold">{escape(label)}</text>'
        )
        parts.append(
            f'<g transform="translate({x:.2f},{y:.2f}) scale({scale})">'
            '<use href="#p8v2-map" opacity="0.60"/>'
        )
        parts.extend(
            _p8_v2_categorical_cells(
                display,
                levels,
                p8.exposure_levels,
                scene,
                registration,
                provenance=f"p9_{action.id}_t480",
                planning_blocked_mask=blocked,
            )
        )
        parts.append(_p8_v2_source_marker(scene, registration))
        parts.append("</g>")
    note_y = (panel_height + title_gap) * 2 + 25
    parts.append(
        f'<text x="24" y="{note_y:.1f}" font-family="sans-serif" font-size="16" font-weight="bold">'
        'P9 phase contract: no firefighter intervention before t=300 s; action branches begin only after the configurable handoff.</text>'
    )
    parts.append(
        f'<text x="24" y="{note_y+28:.1f}" font-family="sans-serif" font-size="14">'
        'Colors are action-conditioned posterior risk levels at t=480 s. Hatched cells remain planning-blocked. Simulation-only.</text>'
    )
    legend_y = note_y + 55
    lower = 0.0
    for index, level in enumerate(p8.exposure_levels):
        x = 24 + index * 210
        band = (
            f"> {lower:.2f}"
            if level.upper_bound is None
            else f"{lower:.2f}–{level.upper_bound:.2f}"
        )
        parts.append(
            f'<rect x="{x}" y="{legend_y}" width="22" height="18" fill="{level.color}" stroke="#333"/>'
            f'<text x="{x+29}" y="{legend_y+14}" font-family="sans-serif" font-size="13">{escape(level.id)} ({band})</text>'
        )
        if level.upper_bound is not None:
            lower = level.upper_bound
    parts.append("</svg>")
    output = Path(output_path)
    return _write_svg_atomically(output, "\n".join(parts))


def render_p9_safe_staging_routes_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Compare three entrances to independently selected safe staging nodes."""
    p9 = load_p9_parameters(config_path)
    p8, scene, registration, graph, encoded = _p8_v2_visual_context(p9.p8_config)
    times = np.asarray(arrays["dynamic_evaluation_times_s"], dtype=np.float64)
    matches = np.flatnonzero(np.isclose(times, 300.0))
    if len(matches) != 1:
        raise ValueError("P9 safe-staging figure requires a t=300 risk frame")
    field = np.asarray(
        arrays["dynamic_no_intervention_exposure"][int(matches[0])],
        dtype=np.float64,
    )
    blocked = np.asarray(scene.occupancy.cells) != 0
    display = complete_blocked_exposure(field, blocked)
    bounds = tuple(
        float(level.upper_bound)
        for level in p8.exposure_levels
        if level.upper_bound is not None
    )
    levels = exposure_risk_levels(display, bounds)
    access = summary["intervention_access"]
    strict = access["strict"]
    diagnostic = access["diagnostic_geometry_only"]
    colors = {
        "exit_west": "#0072B2",
        "exit_center_stair": "#CC79A7",
        "exit_southeast": "#E69F00",
    }
    scale = 0.72
    width = int(registration.source_width_px * scale)
    map_height = registration.source_height_px * scale
    height = int(map_height + 330)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-planning-time-s="300" '
        'data-target="safety-first-source-near-staging" data-entrance-count="3">',
        *_p8_v2_svg_defs(registration, encoded),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="21" font-weight="bold">t=300 s: safety-first access from three entrances</text>',
        f'<g transform="translate(0,42) scale({scale})"><use href="#p8v2-map" opacity="0.62"/>',
        *_p8_v2_compact_categorical_cells(
            levels,
            p8.exposure_levels,
            scene,
            registration,
            provenance="p9_t300_safe_staging",
            planning_blocked_mask=blocked,
        ),
        '<g fill="none" stroke="#009E73" stroke-opacity="0.34" stroke-width="8" stroke-linecap="round">',
    ]
    for edge in graph.edges:
        edge_route = {"path": [{"row": row, "col": col} for row, col in edge.path]}
        parts.append(
            f'<polyline points="{_p8_v2_route_points(scene, registration, edge_route)}"/>'
        )
    parts.append("</g>")
    displayed: dict[str, dict] = {}
    for entrance_id in p8.entrances:
        strict_entry = strict["entrances"][entrance_id]
        entry = strict_entry if strict_entry["feasible"] else diagnostic["entrances"][entrance_id]
        displayed[entrance_id] = entry
        route = entry["route"]
        if not route["feasible"]:
            continue
        points = _p8_v2_route_points(scene, registration, route)
        color = colors[entrance_id]
        dash = "" if strict_entry["feasible"] else ' stroke-dasharray="7 7"'
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="white" stroke-width="14" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )
        parts.append(
            f'<polyline class="p9-safe-staging-route" data-entrance="{entrance_id}" '
            f'data-strict-feasible="{str(strict_entry["feasible"]).lower()}" points="{points}" '
            f'fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" '
            f'stroke-linejoin="round"{dash}/>'
        )
        target_id = entry["selected_target_node"]
        if target_id is not None:
            node = graph.nodes[target_id]
            x, y = registration.metric_to_pixel(*node.position_m)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="13" fill="white" '
                f'stroke="{color}" stroke-width="6"/>'
            )
    parts.append(_p8_v2_source_marker(scene, registration))
    parts.append("</g>")
    table_y = map_height + 82
    headings = (
        "Entrance",
        "Strictly feasible",
        "Selected safe staging",
        "Length (m)",
        "Time (s)",
        "Integrated risk",
        "Max risk",
        "Role",
    )
    columns = (24, 180, 330, 565, 675, 780, 950, 1060)
    for x, heading in zip(columns, headings):
        parts.append(
            f'<text x="{x}" y="{table_y:.1f}" font-family="sans-serif" '
            f'font-size="13" font-weight="bold">{escape(heading)}</text>'
        )
    for row_index, entrance_id in enumerate(p8.entrances):
        strict_entry = strict["entrances"][entrance_id]
        entry = displayed[entrance_id]
        route = entry["route"]
        y = table_y + 32 + row_index * 31
        roles: list[str] = []
        if entrance_id == strict["recommended_entrance"]:
            roles.append("recommended")
        elif entrance_id == strict["fastest_entrance"]:
            roles.append("fastest")
        elif entrance_id == strict["lowest_integrated_exposure_entrance"]:
            roles.append("lowest exposure")
        values = (
            entrance_id,
            "yes" if strict_entry["feasible"] else "no (diagnostic shown)",
            entry["selected_target_node"] or "unavailable",
            f'{route["geometric_length_m"]:.2f}' if route["feasible"] else "—",
            f'{entry["estimated_travel_time_s"]:.1f}' if entry["estimated_travel_time_s"] is not None else "—",
            f'{route["integrated_exposure"]:.4f}' if route["feasible"] else "—",
            f'{route["max_exposure"]:.4f}' if route["feasible"] else "—",
            ", ".join(roles) if roles else "alternative",
        )
        parts.append(
            f'<line x1="24" y1="{y-20:.1f}" x2="{width-24}" y2="{y-20:.1f}" stroke="#DDD"/>'
        )
        for x, value in zip(columns, values):
            parts.append(
                f'<text x="{x}" y="{y:.1f}" font-family="sans-serif" '
                f'font-size="13" fill="{colors[entrance_id]}">{escape(str(value))}</text>'
            )
    note_y = table_y + 145
    recommended = strict["recommended_entrance"]
    recommended_target = strict["recommended_target_node"]
    parts.append(
        f'<text x="24" y="{note_y:.1f}" font-family="sans-serif" font-size="14" font-weight="bold">'
        f'Recommended strict route: {escape(str(recommended or "none"))} → '
        f'{escape(str(recommended_target or "none"))}.</text>'
    )
    parts.append(
        f'<text x="24" y="{note_y+27:.1f}" font-family="sans-serif" font-size="14">'
        'Each entrance selects among four preconfigured green-corridor staging nodes using safety first; the microwave source cell is never a goal.</text>'
    )
    parts.append(
        f'<text x="24" y="{note_y+54:.1f}" font-family="sans-serif" font-size="14">'
        f'Time = length / {p9.firefighter_travel_speed_m_s:.2f} m/s (simulation assumption). Dashed routes are geometry-only diagnostics, not safe-route claims.</text>'
    )
    parts.append("</svg>")
    output = Path(output_path)
    return _write_svg_atomically(output, "\n".join(parts))

