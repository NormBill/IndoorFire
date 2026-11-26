"""Deterministic registered-map visualization for P9 action branches."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from indoor_fire_world.p9_config import load_p9_parameters
from indoor_fire_world.visualization.risk_surface import (
    complete_blocked_exposure,
    exposure_risk_levels,
)
from indoor_fire_world.visualization.svg import (
    _p8_v2_categorical_cells,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()

