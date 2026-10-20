"""Dependency-free, deterministic SVG rendering of the configured scene."""

from __future__ import annotations

import base64
from html import escape
import math
from pathlib import Path

import numpy as np

from indoor_fire_world.scene.models import Polygon, Scene


def _render_p8_v2_forecast_svg_legacy(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
) -> Path:
    """Render the four fixed-scale P8 v2 forecast-exposure snapshots."""
    from indoor_fire_world.p8_v2_config import load_p8_v2_parameters
    from indoor_fire_world.scene import load_scene

    parameters = load_p8_v2_parameters(config_path)
    scene = load_scene(parameters.scene_config)
    registration = scene.map_registration
    if registration is None:
        raise ValueError("P8 v2 visualization requires registered map geometry")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(registration.source_image.read_bytes()).decode("ascii")
    panel_scale = 0.42
    panel_width = registration.source_width_px * panel_scale
    panel_height = registration.source_height_px * panel_scale
    width = int(panel_width * 2)
    height = int(panel_height * 2 + 150)
    levels = parameters.exposure_levels

    def color(value: float) -> str:
        for level in levels:
            if level.upper_bound is None or value <= level.upper_bound:
                return level.color
        raise RuntimeError("Exposure-level configuration has no terminal band")

    def route_points(route: dict) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (
                registration.metric_to_pixel(*scene.navigation.grid_to_metric(item["row"], item["col"]))
                for item in route["path"]
            )
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        f'<image id="p8v2-map" href="data:image/png;base64,{encoded}" width="{registration.source_width_px}" height="{registration.source_height_px}"/>',
        '<pattern id="p8v2-blocked" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#6B7280"/><line x1="0" y1="0" x2="0" y2="8" stroke="#374151" stroke-width="3"/></pattern>',
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    geometry = scene.occupancy.geometry
    for index, time_s in enumerate(parameters.forecast_times_s):
        offset_x = (index % 2) * panel_width
        offset_y = (index // 2) * panel_height + 34
        field = np.asarray(arrays[f"posterior_exposure_t{time_s:g}"], dtype=np.float64)
        risk_grid = np.asarray(arrays[f"posterior_risk_level_t{time_s:g}"], dtype=np.int8)
        parts.append(f'<text x="{offset_x + panel_width / 2:.1f}" y="{offset_y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">Posterior forecast exposure index — t={time_s:g} s</text>')
        parts.append(f'<g transform="translate({offset_x:.2f},{offset_y:.2f}) scale({panel_scale})">')
        parts.append('<use href="#p8v2-map"/>')
        parts.append('<g shape-rendering="crispEdges">')
        for row in range(geometry.rows):
            for col in range(geometry.cols):
                x0 = geometry.origin_x + col * geometry.resolution_m
                x1 = x0 + geometry.resolution_m
                y0 = geometry.origin_y + row * geometry.resolution_m
                y1 = y0 + geometry.resolution_m
                px0, py0 = registration.metric_to_pixel(x0, y1)
                px1, py1 = registration.metric_to_pixel(x1, y0)
                level_index = int(risk_grid[row, col])
                fill = "url(#p8v2-blocked)" if level_index < 0 else levels[level_index].color
                opacity = "0.58" if level_index < 0 else "0.68"
                parts.append(f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{px1-px0:.2f}" height="{py1-py0:.2f}" fill="{fill}" fill-opacity="{opacity}" stroke="#FFFFFF" stroke-opacity="0.22" stroke-width="0.35"><title>row={row}, col={col}, exposure={float(field[row,col]):.6f}, level={"blocked" if level_index < 0 else levels[level_index].id}</title></rect>')
        parts.append('</g>')
        updated = summary["updated_ingress"]["route"]
        prior = summary["prior_ingress"]["route"]
        if prior["feasible"]:
            parts.append(f'<polyline points="{route_points(prior)}" fill="none" stroke="#0072B2" stroke-width="7" stroke-dasharray="12 7"/>')
        if updated["feasible"]:
            parts.append(f'<polyline points="{route_points(updated)}" fill="none" stroke="#CC79A7" stroke-width="7"/>')
        parts.append("</g>")
    legend_y = panel_height * 2 + 78
    parts.append(f'<text x="24" y="{legend_y}" font-family="sans-serif" font-size="16" font-weight="bold">Forecast exposure index (simulation planning scale):</text>')
    x = 390
    for level in levels:
        parts.append(f'<rect x="{x}" y="{legend_y-16}" width="22" height="22" fill="{level.color}" stroke="#333"/><text x="{x+28}" y="{legend_y+1}" font-family="sans-serif" font-size="14">{escape(level.id)}</text>')
        x += 125
    parts.append(f'<rect x="{x}" y="{legend_y-16}" width="22" height="22" fill="url(#p8v2-blocked)" stroke="#333"/><text x="{x+28}" y="{legend_y+1}" font-family="sans-serif" font-size="14">blocked</text>')
    parts.append(f'<text x="24" y="{legend_y+38}" font-family="sans-serif" font-size="14">Blue dashed: prior ingress · magenta: posterior ingress · synthetic reference and thresholds; not certified firefighter safety.</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


_CIVIDIS_STOPS = (
    (0.00, (0, 32, 76)),
    (0.25, (65, 77, 107)),
    (0.50, (125, 124, 120)),
    (0.75, (188, 174, 108)),
    (1.00, (254, 232, 56)),
)


def _continuous_exposure_color(value: float) -> str:
    clipped = min(1.0, max(0.0, float(value)))
    for (left_value, left_rgb), (right_value, right_rgb) in zip(_CIVIDIS_STOPS, _CIVIDIS_STOPS[1:]):
        if clipped <= right_value:
            fraction = (clipped - left_value) / (right_value - left_value)
            rgb = tuple(round(left + fraction * (right - left)) for left, right in zip(left_rgb, right_rgb))
            return "#" + "".join(f"{channel:02X}" for channel in rgb)
    return "#FEE838"


def _p8_v2_visual_context(config_path: str | Path):
    from indoor_fire_world.p8_v2_config import load_p8_v2_parameters
    from indoor_fire_world.planning import load_directed_evacuation_graph
    from indoor_fire_world.scene import load_scene

    parameters = load_p8_v2_parameters(config_path)
    scene = load_scene(parameters.scene_config)
    registration = scene.map_registration
    if registration is None:
        raise ValueError("P8 v2 visualization requires registered map geometry")
    graph = load_directed_evacuation_graph(parameters.corridor_config, scene)
    encoded = base64.b64encode(registration.source_image.read_bytes()).decode("ascii")
    return parameters, scene, registration, graph, encoded


def _p8_v2_svg_defs(registration, encoded: str) -> list[str]:
    return [
        "<defs>",
        f'<image id="p8v2-map" href="data:image/png;base64,{encoded}" width="{registration.source_width_px}" height="{registration.source_height_px}"/>',
        '<pattern id="p8v2-blocked" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#B8BDC6"/><line x1="0" y1="0" x2="0" y2="8" stroke="#737984" stroke-width="2"/></pattern>',
        "</defs>",
    ]


def _p8_v2_cell_geometry(scene, registration, row: int, col: int) -> tuple[float, float, float, float]:
    geometry = scene.occupancy.geometry
    x0 = geometry.origin_x + col * geometry.resolution_m
    y0 = geometry.origin_y + row * geometry.resolution_m
    px0, py0 = registration.metric_to_pixel(x0, y0 + geometry.resolution_m)
    px1, py1 = registration.metric_to_pixel(x0 + geometry.resolution_m, y0)
    return px0, py0, px1 - px0, py1 - py0


def _p8_v2_route_points(scene, registration, route: dict) -> str:
    return " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in (
            registration.metric_to_pixel(*scene.navigation.grid_to_metric(item["row"], item["col"]))
            for item in route["path"]
        )
    )


def _p8_v2_source_marker(scene, registration) -> str:
    microwave = scene.semantic.by_id("microwave")
    if microwave.position is None:
        raise ValueError("Microwave source requires a registered position")
    x, y = registration.metric_to_pixel(*microwave.position)
    return (
        f'<g class="primary-fire-source" data-ignition-source="microwave">'
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="15" fill="white" stroke="#D55E00" stroke-width="7"/>'
        f'<path d="M {x:.2f} {y+8:.2f} C {x-11:.2f} {y-2:.2f}, {x-2:.2f} {y-16:.2f}, {x+1:.2f} {y-23:.2f} C {x+4:.2f} {y-10:.2f}, {x+15:.2f} {y-6:.2f}, {x+8:.2f} {y+8:.2f} Z" fill="#D55E00"/>'
        '<title>Microwave primary source; the only ignition source</title></g>'
    )


def _p8_v2_contours(risk_grid: np.ndarray, scene, registration) -> list[str]:
    rows, cols = risk_grid.shape
    parts = ['<g class="risk-level-contours" fill="none" stroke="#111827" stroke-opacity="0.78">']
    for row in range(rows):
        for col in range(cols):
            current = int(risk_grid[row, col])
            if current < 0:
                continue
            px, py, cell_width, cell_height = _p8_v2_cell_geometry(scene, registration, row, col)
            if col + 1 < cols and int(risk_grid[row, col + 1]) >= 0 and int(risk_grid[row, col + 1]) != current:
                level = max(current, int(risk_grid[row, col + 1]))
                parts.append(f'<line data-boundary-level="{level}" x1="{px+cell_width:.2f}" y1="{py:.2f}" x2="{px+cell_width:.2f}" y2="{py+cell_height:.2f}" stroke-width="{0.9+0.25*level:.2f}"/>')
            if row + 1 < rows and int(risk_grid[row + 1, col]) >= 0 and int(risk_grid[row + 1, col]) != current:
                level = max(current, int(risk_grid[row + 1, col]))
                parts.append(f'<line data-boundary-level="{level}" x1="{px:.2f}" y1="{py:.2f}" x2="{px+cell_width:.2f}" y2="{py:.2f}" stroke-width="{0.9+0.25*level:.2f}"/>')
    parts.append("</g>")
    return parts


def _p8_v2_panel_cells(
    field: np.ndarray,
    risk_grid: np.ndarray,
    scene,
    registration,
    *,
    diagnostic: bool = False,
    component_grid: np.ndarray | None = None,
    component_classes: dict[int, str] | None = None,
) -> list[str]:
    rows, cols = field.shape
    parts = [f'<g class="registered-risk-grid" data-grid-cell-count="{rows*cols}" data-color-scale-min="0" data-color-scale-max="1" shape-rendering="crispEdges">']
    for row in range(rows):
        for col in range(cols):
            px, py, cell_width, cell_height = _p8_v2_cell_geometry(scene, registration, row, col)
            level_index = int(risk_grid[row, col])
            blocked = level_index < 0
            fill = "url(#p8v2-blocked)" if blocked else _continuous_exposure_color(float(field[row, col]))
            opacity = "0.42" if blocked else "0.74"
            component_id = -1 if component_grid is None else int(component_grid[row, col])
            classification = "" if component_id < 0 else (component_classes or {}).get(component_id, "")
            stroke = "none"
            stroke_width = "0"
            if diagnostic and component_id >= 0:
                stroke = {
                    "source_connected_exposure": "#009E73",
                    "threshold_disconnected_transport_region": "#E69F00",
                    "weakly_supported_transport_region": "#7B3294",
                    "unexpected_disconnected_transport": "#D73027",
                }.get(classification, "#FFFFFF")
                stroke_width = "0.9"
            tooltip = f"row={row}, col={col}, exposure={float(field[row,col]):.6f}, level={'blocked' if blocked else level_index}, component={component_id}, classification={classification or 'none'}"
            parts.append(
                f'<rect class="registered-cell" data-row="{row}" data-col="{col}" x="{px:.2f}" y="{py:.2f}" '
                f'width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{fill}" fill-opacity="{opacity}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"><title>{tooltip}</title></rect>'
            )
    parts.append("</g>")
    return parts


def _p8_v2_categorical_cells(
    field: np.ndarray,
    risk_grid: np.ndarray,
    levels,
    scene,
    registration,
    *,
    provenance: str = "posterior_model_forecast",
) -> list[str]:
    """Render one explicit model-predicted level for every registered cell."""
    rows, cols = risk_grid.shape
    parts = [
        f'<g class="categorical-risk-grid" data-grid-cell-count="{rows*cols}" '
        f'data-risk-map-provenance="{escape(provenance)}" shape-rendering="crispEdges">'
    ]
    for row in range(rows):
        for col in range(cols):
            px, py, cell_width, cell_height = _p8_v2_cell_geometry(
                scene, registration, row, col
            )
            level_index = int(risk_grid[row, col])
            blocked = level_index < 0
            level_id = "blocked" if blocked else levels[level_index].id
            fill = "url(#p8v2-blocked)" if blocked else levels[level_index].color
            opacity = "0.48" if blocked else "0.82"
            parts.append(
                f'<rect class="risk-level-cell" data-row="{row}" data-col="{col}" '
                f'data-risk-level="{escape(level_id)}" data-burning="false" x="{px:.2f}" y="{py:.2f}" '
                f'width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{fill}" '
                f'fill-opacity="{opacity}" stroke="#FFFFFF" stroke-opacity="0.28" '
                f'stroke-width="0.28"><title>row={row}, col={col}, model-predicted '
                f'exposure-risk level={escape(level_id)}, exposure={float(field[row,col]):.6f}, '
                f'burning=false</title></rect>'
            )
    parts.append("</g>")
    return parts


def _p8_v2_routes(summary: dict, scene, registration) -> list[str]:
    parts: list[str] = []
    for name, color, dash in (("prior_ingress", "#56B4E9", "18 10"), ("updated_ingress", "#CC79A7", "")):
        route = summary[name]["route"]
        if not route["feasible"]:
            continue
        points = _p8_v2_route_points(scene, registration, route)
        parts.append(f'<polyline points="{points}" fill="none" stroke="white" stroke-opacity="0.92" stroke-width="13" stroke-linejoin="round" stroke-linecap="round"/>')
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline class="{name}" points="{points}" fill="none" stroke="{color}" stroke-width="7" stroke-linejoin="round" stroke-linecap="round"{dash_attribute}/>')
    return parts


def _render_p8_v2_categorical_risk(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
    *,
    include_routes: bool,
    source_kind: str = "posterior",
) -> Path:
    parameters, scene, registration, _, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_scale = 0.42
    panel_width = registration.source_width_px * panel_scale
    panel_height = registration.source_height_px * panel_scale
    width = int(panel_width * 2)
    height = int(panel_height * 2 + 260)
    if source_kind not in {"posterior", "synthetic_reference"}:
        raise ValueError("P8 v2 categorical risk source must be posterior or synthetic_reference")
    is_reference = source_kind == "synthetic_reference"
    map_source = "synthetic_reference_model" if is_reference else "posterior_model_forecast"
    title_prefix = "Synthetic-reference" if is_reference else "Posterior forecast"
    field_prefix = "synthetic_reference_exposure" if is_reference else "posterior_exposure"
    level_prefix = "synthetic_reference_risk_level" if is_reference else "posterior_risk_level"
    counts_root = "synthetic_reference_risk_grid" if is_reference else "risk_grid"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-risk-map-source="{map_source}">'
    ]
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    for index, time_s in enumerate(parameters.forecast_times_s):
        key = f"t{time_s:g}"
        offset_x = (index % 2) * panel_width
        offset_y = (index // 2) * panel_height + 48
        field = np.asarray(arrays[f"{field_prefix}_{key}"], dtype=np.float64)
        risk_grid = np.asarray(arrays[f"{level_prefix}_{key}"], dtype=np.int8)
        counts = summary[counts_root]["counts_by_forecast_time"][key]
        nonminimal = sum(
            int(counts[level.id]) for level in parameters.exposure_levels[1:]
        )
        parts.append(
            f'<text x="{offset_x+panel_width/2:.1f}" y="{offset_y-24:.1f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="20" '
            f'font-weight="bold">{title_prefix} exposure-risk levels — t={time_s:g} s</text>'
        )
        parts.append(
            f'<text x="{offset_x+panel_width/2:.1f}" y="{offset_y-6:.1f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="14">'
            f'{nonminimal} cells above minimal; colored cells are exposure, not fire</text>'
        )
        parts.append(
            f'<g transform="translate({offset_x:.2f},{offset_y:.2f}) scale({panel_scale})">'
            '<use href="#p8v2-map" opacity="0.64"/>'
        )
        parts.extend(
            _p8_v2_categorical_cells(
                field,
                risk_grid,
                parameters.exposure_levels,
                scene,
                registration,
                provenance=map_source,
            )
        )
        parts.append(_p8_v2_source_marker(scene, registration))
        if include_routes:
            parts.extend(_p8_v2_routes(summary, scene, registration))
        parts.append("</g>")
    legend_y = panel_height * 2 + 108
    parts.append(
        f'<text x="24" y="{legend_y}" font-family="sans-serif" font-size="17" '
        'font-weight="bold">Model-predicted exposure-risk per 0.25 m grid cell:</text>'
    )
    x = 390
    lower_bound = 0.0
    for level_index, level in enumerate(parameters.exposure_levels):
        upper_text = "1.00" if level.upper_bound is None else f"{level.upper_bound:.2f}"
        range_text = (
            f"[0.00, {upper_text}]"
            if level_index == 0
            else f"({lower_bound:.2f}, {upper_text}]"
        )
        parts.append(
            f'<rect x="{x}" y="{legend_y-18}" width="24" height="24" fill="{level.color}" '
            f'stroke="#222"/><text x="{x+31}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="14">{escape(level.id)} {range_text}</text>'
        )
        x += 185
        if level.upper_bound is not None:
            lower_bound = level.upper_bound
    parts.append(
        f'<text x="24" y="{legend_y+42}" font-family="sans-serif" font-size="16">'
        'Each traversable cell is classified from the posterior ensemble forecast at that time. '
        'Blocked cells are hatched and have no risk level.</text>'
    )
    provenance_note = (
        "This is the configured synthetic model-comparison reference, not measured W1 ground truth."
        if is_reference
        else "The synthetic reference is used only to synthesize observations and evaluate forecast error; it is not used to construct this posterior risk map."
    )
    parts.append(
        f'<text x="24" y="{legend_y+76}" font-family="sans-serif" font-size="16">'
        f'{escape(provenance_note)}</text>'
    )
    parts.append(
        f'<text x="24" y="{legend_y+104}" font-family="sans-serif" font-size="16" '
        'font-weight="bold" fill="#9C2F21">Every colored cell has burning=false; only the microwave source marker denotes ignition.</text>'
    )
    if include_routes:
        parts.append(
            f'<text x="24" y="{legend_y+136}" font-family="sans-serif" font-size="15">'
            'Blue dashed: prior ingress; purple solid: posterior ingress; circle/flame: the only ignition source.</text>'
        )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def _render_p8_v2_evolution(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path, *, include_routes: bool) -> Path:
    parameters, scene, registration, _, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_scale = 0.42
    panel_width = registration.source_width_px * panel_scale
    panel_height = registration.source_height_px * panel_scale
    width = int(panel_width * 2)
    height = int(panel_height * 2 + 190)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-fixed-color-scale="0:1">']
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    for index, time_s in enumerate(parameters.forecast_times_s):
        offset_x = (index % 2) * panel_width
        offset_y = (index // 2) * panel_height + 38
        field = np.asarray(arrays[f"posterior_exposure_t{time_s:g}"], dtype=np.float64)
        risk_grid = np.asarray(arrays[f"posterior_risk_level_t{time_s:g}"], dtype=np.int8)
        parts.append(f'<text x="{offset_x+panel_width/2:.1f}" y="{offset_y-10:.1f}" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Forecast exposure at t={time_s:g} s</text>')
        parts.append(f'<g transform="translate({offset_x:.2f},{offset_y:.2f}) scale({panel_scale})"><use href="#p8v2-map" opacity="0.72"/>')
        parts.extend(_p8_v2_panel_cells(field, risk_grid, scene, registration))
        parts.extend(_p8_v2_contours(risk_grid, scene, registration))
        parts.append(_p8_v2_source_marker(scene, registration))
        if include_routes:
            parts.extend(_p8_v2_routes(summary, scene, registration))
        parts.append("</g>")
    legend_y = panel_height * 2 + 88
    parts.append(f'<text x="24" y="{legend_y}" font-family="sans-serif" font-size="18" font-weight="bold">Continuous forecast exposure: 0</text>')
    for index in range(101):
        parts.append(f'<rect x="{258+index*3}" y="{legend_y-17}" width="3" height="22" fill="{_continuous_exposure_color(index/100)}"/>')
    parts.append(f'<text x="568" y="{legend_y}" font-family="sans-serif" font-size="18" font-weight="bold">1</text>')
    thresholds = ", ".join(f"{level.id} <= {level.upper_bound:g}" for level in parameters.exposure_levels if level.upper_bound is not None)
    parts.append(f'<text x="620" y="{legend_y}" font-family="sans-serif" font-size="16">Contour boundaries: {escape(thresholds)}; extreme &gt; 0.70</text>')
    parts.append(f'<text x="24" y="{legend_y+42}" font-family="sans-serif" font-size="16" font-weight="bold" fill="#D55E00">Circle/flame: microwave primary source (only ignition).</text>')
    parts.append(f'<text x="500" y="{legend_y+42}" font-family="sans-serif" font-size="16">Colored cells indicate forecast exposure, not burning or secondary ignition. Shared fixed scale [0,1].</text>')
    if include_routes:
        parts.append(f'<text x="24" y="{legend_y+78}" font-family="sans-serif" font-size="15">Blue dashed: prior ingress; purple solid: posterior ingress. Routes are planned on the worst-case envelope.</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_v2_forecast_svg(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path) -> Path:
    """Backward-compatible full-grid categorical risk map with route context."""
    return _render_p8_v2_categorical_risk(
        config_path, summary, arrays, output_path, include_routes=True
    )


def render_p8_v2_categorical_risk_svg(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path) -> Path:
    """Render one posterior model-predicted risk level in every registered cell."""
    return _render_p8_v2_categorical_risk(
        config_path, summary, arrays, output_path, include_routes=False
    )


def render_p8_v2_synthetic_reference_risk_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
) -> Path:
    """Render the synthetic comparison reference with the posterior map's thresholds."""
    return _render_p8_v2_categorical_risk(
        config_path,
        summary,
        arrays,
        output_path,
        include_routes=False,
        source_kind="synthetic_reference",
    )


def render_p8_v2_posterior_reference_comparison_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
) -> Path:
    """Render synthetic reference and posterior risk grids on one shared scale."""
    parameters, scene, registration, _, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_scale = 0.24
    panel_width = registration.source_width_px * panel_scale
    panel_height = registration.source_height_px * panel_scale
    row_gap = 42
    width = int(panel_width * 4)
    height = int(panel_height * 2 + row_gap * 2 + 190)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-comparison="posterior_vs_synthetic_reference">'
    ]
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    sources = (
        (
            "Synthetic reference",
            "synthetic_reference_exposure",
            "synthetic_reference_risk_level",
            "synthetic_reference_model",
        ),
        (
            "Posterior forecast",
            "posterior_exposure",
            "posterior_risk_level",
            "posterior_model_forecast",
        ),
    )
    for row_index, (label, field_prefix, level_prefix, provenance) in enumerate(sources):
        for column_index, time_s in enumerate(parameters.forecast_times_s):
            key = f"t{time_s:g}"
            offset_x = column_index * panel_width
            offset_y = row_gap + row_index * (panel_height + row_gap)
            field = np.asarray(arrays[f"{field_prefix}_{key}"], dtype=np.float64)
            risk_grid = np.asarray(arrays[f"{level_prefix}_{key}"], dtype=np.int8)
            parts.append(
                f'<text x="{offset_x+panel_width/2:.1f}" y="{offset_y-12:.1f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="16" '
                f'font-weight="bold">{label} — t={time_s:g} s</text>'
            )
            parts.append(
                f'<g transform="translate({offset_x:.2f},{offset_y:.2f}) scale({panel_scale})">'
                '<use href="#p8v2-map" opacity="0.62"/>'
            )
            parts.extend(
                _p8_v2_categorical_cells(
                    field,
                    risk_grid,
                    parameters.exposure_levels,
                    scene,
                    registration,
                    provenance=provenance,
                )
            )
            parts.append(_p8_v2_source_marker(scene, registration))
            parts.append("</g>")
            if row_index == 1:
                comparison = summary["synthetic_reference_risk_grid"][
                    "posterior_comparison_by_forecast_time"
                ][key]
                parts.append(
                    f'<text x="{offset_x+panel_width/2:.1f}" '
                    f'y="{offset_y+panel_height+19:.1f}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="12">'
                    f'level agreement={comparison["exact_level_agreement_cell_count"]}/5800; '
                    f'exposure MAE={comparison["exposure_mae"]:.3f}</text>'
                )
    legend_y = panel_height * 2 + row_gap * 2 + 74
    parts.append(
        f'<text x="24" y="{legend_y}" font-family="sans-serif" font-size="16" '
        'font-weight="bold">Shared exposure-risk levels:</text>'
    )
    x = 250
    for level in parameters.exposure_levels:
        parts.append(
            f'<rect x="{x}" y="{legend_y-17}" width="22" height="22" fill="{level.color}" '
            f'stroke="#222"/><text x="{x+28}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="13">{escape(level.id)}</text>'
        )
        x += 155
    parts.append(
        f'<text x="24" y="{legend_y+38}" font-family="sans-serif" font-size="15">'
        'Both rows use identical posterior-planning thresholds and the registered 50×160 grid; no panel normalization.</text>'
    )
    parts.append(
        f'<text x="24" y="{legend_y+68}" font-family="sans-serif" font-size="15" '
        'font-weight="bold" fill="#9C2F21">Synthetic reference is a simulation comparison target, not measured W1 ground truth. Colored cells are exposure, not burning.</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_v2_pre_entry_keyframes_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
) -> Path:
    """Render sparse pre-entry keyframes from the dense causal prediction record."""
    parameters, scene, registration, _, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dense_times = np.asarray(arrays["dense_prediction_times_s"], dtype=np.float64)
    availability = np.asarray(arrays["dense_posterior_available"], dtype=np.bool_)
    panel_scale = 0.22
    panel_width = registration.source_width_px * panel_scale
    panel_height = registration.source_height_px * panel_scale
    row_gap = 42
    width = int(panel_width * 4)
    height = int(panel_height * 3 + row_gap * 3 + 180)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-time-semantics="pre_entry_sparse_keyframes_from_10s_dense_prediction">'
    ]
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    rows = (
        ("Synthetic reference", "dense_synthetic_reference_exposure", "dense_synthetic_reference_risk_level", "synthetic_reference_model"),
        ("Physics prior", "dense_prior_exposure", "dense_prior_risk_level", "physics_prior"),
        ("Observation-conditioned posterior", "dense_posterior_exposure", "dense_posterior_risk_level", "posterior_model_forecast"),
    )
    for row_index, (label, field_name, level_name, provenance) in enumerate(rows):
        for column_index, time_s in enumerate(parameters.pre_entry_visualization_times_s):
            matches = np.flatnonzero(np.isclose(dense_times, time_s))
            if len(matches) != 1:
                raise ValueError("Pre-entry keyframe is missing from dense prediction times")
            dense_index = int(matches[0])
            offset_x = column_index * panel_width
            offset_y = row_gap + row_index * (panel_height + row_gap)
            parts.append(
                f'<text x="{offset_x+panel_width/2:.1f}" y="{offset_y-12:.1f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="15" '
                f'font-weight="bold">{label} — t={time_s:g} s</text>'
            )
            parts.append(
                f'<g transform="translate({offset_x:.2f},{offset_y:.2f}) scale({panel_scale})">'
                '<use href="#p8v2-map" opacity="0.60"/>'
            )
            posterior_unavailable = row_index == 2 and not availability[dense_index]
            if posterior_unavailable:
                parts.append(
                    f'<rect x="0" y="0" width="{registration.source_width_px}" '
                    f'height="{registration.source_height_px}" fill="#F2F2F2" fill-opacity="0.78"/>'
                )
                parts.append(
                    f'<text x="{registration.source_width_px/2:.1f}" '
                    f'y="{registration.source_height_px/2:.1f}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="54" font-weight="bold" fill="#555">'
                    f'Unavailable before observation at t={summary["observation_time_s"]:g} s</text>'
                )
            else:
                field = np.asarray(arrays[field_name][dense_index], dtype=np.float64)
                levels = np.asarray(arrays[level_name][dense_index], dtype=np.int8)
                parts.extend(
                    _p8_v2_categorical_cells(
                        field,
                        levels,
                        parameters.exposure_levels,
                        scene,
                        registration,
                        provenance=provenance,
                    )
                )
                parts.append(_p8_v2_source_marker(scene, registration))
            parts.append("</g>")
    legend_y = panel_height * 3 + row_gap * 3 + 62
    parts.append(
        f'<text x="24" y="{legend_y}" font-family="sans-serif" font-size="16" '
        'font-weight="bold">Sparse display only; underlying evaluation cadence is 10 s.</text>'
    )
    parts.append(
        f'<text x="24" y="{legend_y+32}" font-family="sans-serif" font-size="15">'
        'Posterior panels are never backfilled before the independent case observation. All available panels share the same risk thresholds and registered grid.</text>'
    )
    parts.append(
        f'<text x="24" y="{legend_y+62}" font-family="sans-serif" font-size="15">'
        'Colored cells represent forecast exposure-risk, not burning; the microwave marker is the only ignition source.</text>'
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_v2_dense_pre_entry_timeseries_svg(
    config_path: str | Path,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
) -> Path:
    """Plot all 10 s pre-entry evaluations while keeping maps sparse."""
    from indoor_fire_world.p8_v2_config import load_p8_v2_parameters

    parameters = load_p8_v2_parameters(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        record
        for record in summary["dense_prediction"]["records"]
        if record["time_s"] <= parameters.handoff_time_s
    ]
    width, height = 1280, 720
    left, right = 92.0, 1215.0
    top_a, bottom_a = 72.0, 315.0
    top_b, bottom_b = 405.0, 620.0
    times = np.asarray([record["time_s"] for record in records], dtype=np.float64)
    x_for = lambda value: left + (value - times[0]) / (times[-1] - times[0]) * (right - left)

    def nonminimal(counts: dict[str, int] | None) -> float:
        return math.nan if counts is None else float(5800 - counts["minimal"])

    reference_counts = np.asarray(
        [nonminimal(record["synthetic_reference_risk_level_counts"]) for record in records]
    )
    prior_counts = np.asarray(
        [nonminimal(record["prior_risk_level_counts"]) for record in records]
    )
    posterior_counts = np.asarray(
        [nonminimal(record["posterior_risk_level_counts"]) for record in records]
    )
    prior_mae = np.asarray(
        [record["prior_reference_exposure_mae"] for record in records], dtype=np.float64
    )
    posterior_mae = np.asarray(
        [
            math.nan
            if record["posterior_reference_exposure_mae"] is None
            else record["posterior_reference_exposure_mae"]
            for record in records
        ],
        dtype=np.float64,
    )
    max_count = max(1.0, float(np.nanmax([reference_counts, prior_counts, posterior_counts])))
    max_mae = max(1e-6, float(np.nanmax([prior_mae, posterior_mae])))

    def polyline(values: np.ndarray, top: float, bottom: float, maximum: float) -> str:
        return " ".join(
            f"{x_for(time):.2f},{bottom-(value/maximum)*(bottom-top):.2f}"
            for time, value in zip(times, values)
            if np.isfinite(value)
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-dense-interval-s="10">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Dense pre-entry prediction record (10 s cadence)</text>',
    ]
    for top, bottom, title in (
        (top_a, bottom_a, "Traversable cells above minimal exposure-risk"),
        (top_b, bottom_b, "Exposure MAE against synthetic reference"),
    ):
        parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#FAFAFA" stroke="#999"/>')
        parts.append(f'<text x="{left}" y="{top-12}" font-family="sans-serif" font-size="16" font-weight="bold">{title}</text>')
        for tick in (60, 100, 140, 180, 220, 260, 300):
            x = x_for(float(tick))
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#E2E2E2"/>')
            parts.append(f'<text x="{x:.2f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{tick}</text>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom_a - fraction * (bottom_a - top_a)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#E2E2E2"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{int(round(fraction*max_count))}</text>')
        y_mae = bottom_b - fraction * (bottom_b - top_b)
        parts.append(f'<line x1="{left}" y1="{y_mae:.2f}" x2="{right}" y2="{y_mae:.2f}" stroke="#E2E2E2"/>')
        parts.append(f'<text x="{left-12}" y="{y_mae+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{fraction*max_mae:.3f}</text>')
    parts.extend(
        [
            f'<polyline points="{polyline(reference_counts, top_a, bottom_a, max_count)}" fill="none" stroke="#000000" stroke-width="4"/>',
            f'<polyline points="{polyline(prior_counts, top_a, bottom_a, max_count)}" fill="none" stroke="#56B4E9" stroke-width="4" stroke-dasharray="10 7"/>',
            f'<polyline points="{polyline(posterior_counts, top_a, bottom_a, max_count)}" fill="none" stroke="#CC79A7" stroke-width="4"/>',
            f'<polyline points="{polyline(prior_mae, top_b, bottom_b, max_mae)}" fill="none" stroke="#56B4E9" stroke-width="4" stroke-dasharray="10 7"/>',
            f'<polyline points="{polyline(posterior_mae, top_b, bottom_b, max_mae)}" fill="none" stroke="#CC79A7" stroke-width="4"/>',
        ]
    )
    observation_x = x_for(float(summary["observation_time_s"]))
    parts.append(f'<line x1="{observation_x:.2f}" y1="{top_a}" x2="{observation_x:.2f}" y2="{bottom_b}" stroke="#D55E00" stroke-width="3" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{observation_x+8:.2f}" y="{top_a+20}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#D55E00">UGV observation</text>')
    parts.extend(
        [
            '<line x1="92" y1="675" x2="142" y2="675" stroke="#000" stroke-width="4"/><text x="152" y="680" font-family="sans-serif" font-size="14">Synthetic reference</text>',
            '<line x1="340" y1="675" x2="390" y2="675" stroke="#56B4E9" stroke-width="4" stroke-dasharray="10 7"/><text x="400" y="680" font-family="sans-serif" font-size="14">Physics prior</text>',
            '<line x1="570" y1="675" x2="620" y2="675" stroke="#CC79A7" stroke-width="4"/><text x="630" y="680" font-family="sans-serif" font-size="14">Posterior (available only after observation)</text>',
            '<text x="1215" y="680" text-anchor="end" font-family="sans-serif" font-size="13">Simulation reference, not measured W1 ground truth</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_v2_exposure_evolution_svg(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path) -> Path:
    """Render four fixed-scale exposure snapshots without route clutter."""
    return _render_p8_v2_evolution(config_path, summary, arrays, output_path, include_routes=False)


def _risk_grid_for_visual(field: np.ndarray, levels, scene) -> np.ndarray:
    traversable = np.asarray(scene.occupancy.cells) == 0
    grid = np.full(field.shape, -1, dtype=np.int8)
    remaining = traversable.copy()
    for index, level in enumerate(levels):
        selected = remaining if level.upper_bound is None else remaining & (field <= level.upper_bound + 1e-12)
        grid[selected] = index
        remaining[selected] = False
    return grid


def render_p8_v2_planning_envelope_svg(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path) -> Path:
    """Render the worst-case planning envelope, green corridor and compared routes."""
    parameters, scene, registration, graph, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = 0.72
    width = int(registration.source_width_px * scale)
    height = int(registration.source_height_px * scale + 150)
    field = np.asarray(arrays["planning_exposure_envelope"], dtype=np.float64)
    risk_grid = _risk_grid_for_visual(field, parameters.exposure_levels, scene)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-fixed-color-scale="0:1">']
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.extend(['<rect width="100%" height="100%" fill="white"/>', '<text x="24" y="28" font-family="sans-serif" font-size="21" font-weight="bold">Worst-case forecast exposure envelope and firefighter ingress</text>', f'<g transform="translate(0,42) scale({scale})"><use href="#p8v2-map" opacity="0.72"/>'])
    parts.extend(_p8_v2_panel_cells(field, risk_grid, scene, registration))
    parts.extend(_p8_v2_contours(risk_grid, scene, registration))
    parts.append('<g class="registered-green-corridor" fill="none" stroke="#009E73" stroke-opacity="0.72" stroke-width="9" stroke-linecap="round">')
    for edge in graph.edges:
        route = {"path": [{"row": row, "col": col} for row, col in edge.path]}
        parts.append(f'<polyline points="{_p8_v2_route_points(scene, registration, route)}"/>')
    parts.append("</g>")
    parts.extend(_p8_v2_routes(summary, scene, registration))
    parts.append(_p8_v2_source_marker(scene, registration))
    parts.append("</g>")
    legend_y = registration.source_height_px * scale + 78
    parts.append(f'<text x="24" y="{legend_y:.1f}" font-family="sans-serif" font-size="16">Green: registered corridor; blue dashed: prior ingress; purple solid: posterior ingress.</text>')
    parts.append(f'<text x="24" y="{legend_y+32:.1f}" font-family="sans-serif" font-size="16">Planning field = cellwise max of t=300/360/420/480 s exposure; colored cells are not burning cells.</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_v2_grid_diagnostic_svg(config_path: str | Path, summary: dict, arrays: dict, output_path: str | Path) -> Path:
    """Render exact cells and component provenance for the first forecast snapshot."""
    parameters, scene, registration, _, encoded = _p8_v2_visual_context(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    time_s = parameters.forecast_times_s[0]
    key = f"t{time_s:g}"
    field = np.asarray(arrays[f"posterior_exposure_{key}"], dtype=np.float64)
    risk_grid = np.asarray(arrays[f"posterior_risk_level_{key}"], dtype=np.int8)
    component_grid = np.asarray(arrays[f"posterior_exposure_component_{key}"], dtype=np.int32)
    first_level = next(level.id for level in parameters.exposure_levels if level.upper_bound is not None)
    component_classes = {
        int(record["component_id"]): str(record["classification"])
        for record in summary["risk_component_diagnostics"][key]["thresholds"][first_level]["components"]
    }
    scale = 0.72
    width = int(registration.source_width_px * scale)
    height = int(registration.source_height_px * scale + 150)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-fixed-color-scale="0:1">']
    parts.extend(_p8_v2_svg_defs(registration, encoded))
    parts.extend(['<rect width="100%" height="100%" fill="white"/>', f'<text x="24" y="28" font-family="sans-serif" font-size="21" font-weight="bold">Grid/component audit at t={time_s:g} s</text>', f'<g transform="translate(0,42) scale({scale})"><use href="#p8v2-map" opacity="0.72"/>'])
    parts.extend(_p8_v2_panel_cells(field, risk_grid, scene, registration, diagnostic=True, component_grid=component_grid, component_classes=component_classes))
    parts.append(_p8_v2_source_marker(scene, registration))
    parts.append("</g>")
    legend_y = registration.source_height_px * scale + 78
    parts.append(f'<text x="24" y="{legend_y:.1f}" font-family="sans-serif" font-size="16">Green: source-connected; orange: meaningful transport; purple: weak numerical-tail support; red: unexpected disconnected transport.</text>')
    parts.append(f'<text x="24" y="{legend_y+32:.1f}" font-family="sans-serif" font-size="16">All 50×160 cells are retained. Hover a cell for exposure, level, component, and classification.</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_scene_svg(scene: Scene, output_path: str | Path, scale: int = 45) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = scene.occupancy.geometry
    width = int(geometry.width_m * scale)
    height = int(geometry.height_m * scale)

    def px(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return ((x - geometry.origin_x) * scale, height - (y - geometry.origin_y) * scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
    ]
    obstacle_colors = {"wall": "#30343b", "workbench": "#9b7048", "equipment": "#64748b"}
    point_styles = {
        "door": ("#d9a441", "D"),
        "exit": ("#168a45", "EXIT"),
        "fixed_sensor": ("#2563b9", "S"),
        "microwave": ("#d33b32", "MW"),
        "ugv_start": ("#7c3aed", "UGV"),
    }
    for obj in scene.semantic.objects:
        if obj.footprint is not None:
            color = obstacle_colors.get(obj.kind, "#777")
            if isinstance(obj.footprint, Polygon):
                points = " ".join(
                    f"{x},{y}" for x, y in (px(point) for point in obj.footprint.vertices)
                )
                parts.append(f'<polygon points="{points}" fill="{color}"/>')
                x, y = px(obj.footprint.vertices[0])
            else:
                rect = obj.footprint
                x = (rect.x - geometry.origin_x) * scale
                y = height - (rect.y + rect.height - geometry.origin_y) * scale
                parts.append(f'<rect x="{x}" y="{y}" width="{rect.width * scale}" height="{rect.height * scale}" fill="{color}"/>')
            if obj.kind != "wall":
                parts.append(f'<text x="{x + 4}" y="{y + 15}" font-size="11" fill="white">{escape(obj.id)}</text>')
        elif obj.position is not None:
            x, y = px(obj.position)
            color, label = point_styles.get(obj.kind, ("#222", obj.kind))
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}" stroke="white" stroke-width="2"/>')
            parts.append(f'<text x="{x + 10}" y="{y + 4}" font-size="12" font-family="sans-serif" fill="#111">{escape(label)}: {escape(obj.id)}</text>')
    parts.append(f'<text x="12" y="24" font-size="18" font-family="sans-serif" font-weight="bold">{escape(scene.name)} (approximate)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_registration_overlay_svg(scene: Scene, output_path: str | Path) -> Path:
    """Overlay digitized geometry on the immutable registered floor-plan image."""
    registration = scene.map_registration
    if registration is None:
        raise ValueError("Scene does not define a reference-map registration")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(registration.source_image.read_bytes()).decode("ascii")

    def pixel(point: tuple[float, float]) -> tuple[float, float]:
        return registration.metric_to_pixel(*point)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{registration.source_width_px}" height="{registration.source_height_px}" '
            f'viewBox="0 0 {registration.source_width_px} {registration.source_height_px}">'
        ),
        (
            f'<image href="data:image/png;base64,{encoded}" x="0" y="0" '
            f'width="{registration.source_width_px}" height="{registration.source_height_px}"/>'
        ),
        '<g fill="none" stroke="#2563eb" stroke-width="3" opacity="0.78">',
    ]
    for obj in scene.semantic.objects:
        if obj.footprint is None:
            continue
        if isinstance(obj.footprint, Polygon):
            points = " ".join(f"{x},{y}" for x, y in map(pixel, obj.footprint.vertices))
            parts.append(f'<polygon points="{points}"/>')
        else:
            rect = obj.footprint
            x1, y1 = pixel((rect.x, rect.y + rect.height))
            x2, y2 = pixel((rect.x + rect.width, rect.y))
            parts.append(
                f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}"/>'
            )
    parts.append("</g>")
    styles = {
        "exit": ("#16a34a", "EXIT"),
        "microwave": ("#dc2626", "MW"),
        "ugv_start": ("#7c3aed", "UGV"),
    }
    for obj in scene.semantic.objects:
        if obj.position is None or obj.kind not in styles:
            continue
        x, y = pixel(obj.position)
        color, label = styles[obj.kind]
        use_left_label = x > registration.source_width_px - 260
        label_x = x - 13 if use_left_label else x + 13
        anchor = "end" if use_left_label else "start"
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="10" fill="{color}" stroke="white" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{label_x}" y="{y - 13}" text-anchor="{anchor}" font-size="18" '
            f'font-family="sans-serif" font-weight="bold" fill="{color}">'
            f'{escape(label)}: {escape(obj.id)}</text>'
        )
    left, top, width, height = registration.floor_roi_px
    parts.extend(
        [
            (
                f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
                'fill="none" stroke="#f59e0b" stroke-width="3" stroke-dasharray="10 8"/>'
            ),
            (
                f'<text x="{left + 8}" y="{top + 26}" font-size="18" font-family="sans-serif" '
                f'font-weight="bold" fill="#92400e">{escape(registration.calibration_status)}</text>'
            ),
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()


def render_p8_registered_overlay_svg(
    scene: Scene,
    summary: dict,
    arrays: dict,
    output_path: str | Path,
    panels: tuple[str, ...] = ("prior", "posterior", "delta"),
) -> Path:
    """Render registered P8 hazard maps with directed green-route risk."""
    registration = scene.map_registration
    if registration is None:
        raise ValueError("P8 visualization requires a registered source map")
    if summary.get("case_status") != "valid":
        raise ValueError("P8 visualization requires a valid case")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(registration.source_image.read_bytes()).decode("ascii")

    def path_points(route: dict) -> str:
        metric = (
            scene.navigation.grid_to_metric(item["row"], item["col"])
            for item in route["path"]
        )
        return " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (registration.metric_to_pixel(*point) for point in metric)
        )

    prior_cost = np.asarray(arrays["prior_planning_cost"], dtype=np.float64)
    posterior_cost = np.asarray(arrays["posterior_planning_cost"], dtype=np.float64)
    prior_finite = np.isfinite(prior_cost)
    posterior_finite = np.isfinite(posterior_cost)
    common_finite = prior_finite & posterior_finite
    if not np.any(common_finite):
        raise ValueError("P8 risk map requires finite traversable planning costs")
    scheme = summary["risk_level_scheme"]
    levels = tuple(scheme["levels"])
    base_cost = float(scheme["base_cost"])
    prior = np.full_like(prior_cost, np.nan)
    posterior = np.full_like(posterior_cost, np.nan)
    prior[prior_finite] = np.maximum(prior_cost[prior_finite] - base_cost, 0.0)
    posterior[posterior_finite] = np.maximum(
        posterior_cost[posterior_finite] - base_cost,
        0.0,
    )
    delta = np.zeros_like(prior_cost)
    np.subtract(posterior_cost, prior_cost, out=delta, where=common_finite)
    fields = {
        "prior": prior,
        "posterior": posterior,
        "delta": delta,
    }
    titles = {
        "prior": f"Prior decision-risk levels at t={summary['fire_time_s']:g} s",
        "posterior": (
            "Posterior decision-risk levels after UGV observation "
            f"at t={summary['fire_time_s']:g} s"
        ),
        "delta": (
            "Planning-cost delta (posterior - prior) "
            f"at t={summary['fire_time_s']:g} s"
        ),
    }
    if not panels or any(name not in fields for name in panels):
        raise ValueError("P8 risk-map panels must be prior, posterior, or delta")

    diverging_negative = ("#2166AC", "#67A9CF", "#D1E5F0")
    diverging_positive = ("#FDDBC7", "#EF8A62", "#B2182B")

    def level_for(excess_cost: float) -> dict:
        for level in levels:
            upper = level["upper_bound_excess_cost"]
            if upper is None or excess_cost <= float(upper) + 1e-12:
                return level
        raise ValueError("P8 risk-level scheme has no terminal band")

    def risk_color(value: float, *, delta: bool, scale: float) -> str:
        if delta:
            if abs(value) <= 1e-12:
                return "#F7F7F7"
            normalized = 0.0 if scale <= 1e-12 else min(1.0, abs(value) / scale)
            palette = diverging_positive if value >= 0.0 else diverging_negative
            index = min(len(palette) - 1, int(normalized * len(palette)))
            return palette[index]
        if not np.isfinite(value):
            return str(scheme["untenable"]["color"])
        return str(level_for(max(0.0, value))["color"])

    source_width = registration.source_width_px
    source_height = registration.source_height_px
    panel_scale = 1.0 if len(panels) == 1 else 0.5
    panel_width = source_width * panel_scale
    panel_height = source_height * panel_scale
    width = int(panel_width * len(panels))
    height = int(panel_height + 184)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        f'<image id="p8-base-map" href="data:image/png;base64,{encoded}" x="0" y="0" width="{source_width}" height="{source_height}"/>',
        '<marker id="p8-directed-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#009E73"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    graph = summary["directed_evacuation_route_graph"]
    geometry = scene.occupancy.geometry
    delta_scale = float(np.max(np.abs(fields["delta"])))
    point = summary["observation_point"]["resolved_position_m"]
    observation_px = registration.metric_to_pixel(point["x"], point["y"])
    for panel_index, panel_name in enumerate(panels):
        field = fields[panel_name]
        is_delta = panel_name == "delta"
        offset_x = panel_index * panel_width
        parts.append(
            f'<text x="{offset_x + panel_width / 2:.2f}" y="28" text-anchor="middle" font-size="20" font-family="sans-serif" font-weight="bold" fill="#222">{escape(titles[panel_name])}</text>'
        )
        parts.append(f'<g transform="translate({offset_x:.2f},40) scale({panel_scale})">')
        parts.append('<use href="#p8-base-map"/>')
        for row in range(geometry.rows):
            for col in range(geometry.cols):
                if not scene.navigation.is_traversable((row, col)):
                    continue
                value = float(field[row, col])
                if is_delta and abs(value) <= 1e-9:
                    continue
                x0 = geometry.origin_x + col * geometry.resolution_m
                x1 = x0 + geometry.resolution_m
                y0 = geometry.origin_y + row * geometry.resolution_m
                y1 = y0 + geometry.resolution_m
                px0, py0 = registration.metric_to_pixel(x0, y1)
                px1, py1 = registration.metric_to_pixel(x1, y0)
                color = risk_color(value, delta=is_delta, scale=delta_scale)
                parts.append(
                    f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{px1-px0:.2f}" height="{py1-py0:.2f}" fill="{color}" opacity="0.62"/>'
                )
        for edge in graph["edges"]:
            edge_cells = [(item["row"], item["col"]) for item in edge["path"]]
            points = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    registration.metric_to_pixel(*scene.navigation.grid_to_metric(*cell))
                    for cell in edge_cells
                )
            )
            for source_cell, target_cell in zip(edge_cells, edge_cells[1:]):
                source_px = registration.metric_to_pixel(
                    *scene.navigation.grid_to_metric(*source_cell)
                )
                target_px = registration.metric_to_pixel(
                    *scene.navigation.grid_to_metric(*target_cell)
                )
                segment_value = float(field[target_cell])
                segment_color = risk_color(
                    segment_value,
                    delta=is_delta,
                    scale=delta_scale,
                )
                parts.append(
                    f'<line x1="{source_px[0]:.2f}" y1="{source_px[1]:.2f}" x2="{target_px[0]:.2f}" y2="{target_px[1]:.2f}" stroke="{segment_color}" stroke-width="13" opacity="0.95" stroke-linecap="round"/>'
                )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="#009E73" stroke-width="3" stroke-dasharray="10 8" marker-end="url(#p8-directed-arrow)"/>'
            )
        routes = [
            (summary["ugv_approach"], "#7C3AED", "10 8", 6),
        ]
        if panel_name in ("prior", "delta"):
            routes.append((summary["prior_retreat"]["route"], "#0072B2", "12 7", 7))
        if panel_name in ("posterior", "delta"):
            routes.append((summary["updated_retreat"]["route"], "#CC79A7", "", 7))
        for route, color, dash, stroke_width in routes:
            if not route["feasible"]:
                continue
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<polyline points="{path_points(route)}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"{dash_attribute}/>'
            )
        if is_delta:
            # The correction is intentionally local and may sit underneath a route.
            # Re-outline changed cells after route drawing so the delta stays visible.
            for row in range(geometry.rows):
                for col in range(geometry.cols):
                    value = float(field[row, col])
                    if abs(value) <= 1e-9:
                        continue
                    x0 = geometry.origin_x + col * geometry.resolution_m
                    x1 = x0 + geometry.resolution_m
                    y0 = geometry.origin_y + row * geometry.resolution_m
                    y1 = y0 + geometry.resolution_m
                    px0, py0 = registration.metric_to_pixel(x0, y1)
                    px1, py1 = registration.metric_to_pixel(x1, y0)
                    parts.append(
                        f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{px1-px0:.2f}" height="{py1-py0:.2f}" fill="none" stroke="{risk_color(value, delta=True, scale=delta_scale)}" stroke-width="2.5"/>'
                    )
            changed = np.argwhere(np.abs(field) > 1e-9)
            if changed.size:
                row_min, col_min = changed.min(axis=0)
                row_max, col_max = changed.max(axis=0)
                metric_left = geometry.origin_x + col_min * geometry.resolution_m
                metric_right = geometry.origin_x + (col_max + 1) * geometry.resolution_m
                metric_bottom = geometry.origin_y + row_min * geometry.resolution_m
                metric_top = geometry.origin_y + (row_max + 1) * geometry.resolution_m
                box_left, box_bottom = registration.metric_to_pixel(
                    metric_left,
                    metric_bottom,
                )
                box_right, box_top = registration.metric_to_pixel(
                    metric_right,
                    metric_top,
                )
                parts.append(
                    f'<rect x="{box_left - 7:.2f}" y="{box_top - 7:.2f}" width="{box_right-box_left+14:.2f}" height="{box_bottom-box_top+14:.2f}" fill="none" stroke="#2166AC" stroke-width="4" stroke-dasharray="7 5"/>'
                )
                parts.append(
                    f'<text x="{box_right + 12:.2f}" y="{box_top + 18:.2f}" font-size="16" font-family="sans-serif" font-weight="bold" fill="#2166AC" paint-order="stroke" stroke="white" stroke-width="4">local posterior cost decrease</text>'
                )
        parts.append(
            f'<circle cx="{observation_px[0]:.2f}" cy="{observation_px[1]:.2f}" r="12" fill="#E69F00" stroke="#111" stroke-width="3"/>'
        )
        parts.append("</g>")

    legend_y = panel_height + 72
    parts.extend(
        [
            f'<line x1="28" y1="{legend_y}" x2="84" y2="{legend_y}" stroke="#009E73" stroke-width="4" stroke-dasharray="10 8"/><text x="94" y="{legend_y + 6}" font-size="16" font-family="sans-serif">Directed green-arrow route</text>',
            f'<line x1="330" y1="{legend_y}" x2="386" y2="{legend_y}" stroke="#7C3AED" stroke-width="6" stroke-dasharray="10 8"/><text x="396" y="{legend_y + 6}" font-size="16" font-family="sans-serif">UGV approach</text>',
            f'<line x1="570" y1="{legend_y}" x2="626" y2="{legend_y}" stroke="#0072B2" stroke-width="7" stroke-dasharray="12 7"/><text x="636" y="{legend_y + 6}" font-size="16" font-family="sans-serif">Prior retreat</text>',
            f'<line x1="820" y1="{legend_y}" x2="876" y2="{legend_y}" stroke="#CC79A7" stroke-width="7"/><text x="886" y="{legend_y + 6}" font-size="16" font-family="sans-serif">Updated retreat</text>',
            f'<circle cx="1115" cy="{legend_y}" r="8" fill="#E69F00" stroke="#111" stroke-width="2"/><text x="1130" y="{legend_y + 6}" font-size="16" font-family="sans-serif">UGV observation</text>',
        ]
    )
    level_y = legend_y + 34
    parts.append(
        f'<text x="28" y="{level_y + 6}" font-size="15" font-family="sans-serif" font-weight="bold">Decision-risk level:</text>'
    )
    level_x = 180
    for level in levels:
        upper = level["upper_bound_excess_cost"]
        bound = "open" if upper is None else f"≤{float(upper):g}"
        parts.append(
            f'<rect x="{level_x}" y="{level_y - 10}" width="20" height="20" fill="{escape(str(level["color"]))}" stroke="#333" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{level_x + 27}" y="{level_y + 6}" font-size="14" font-family="sans-serif">{escape(str(level["label"]))} ({bound})</text>'
        )
        level_x += 170
    untenable = scheme["untenable"]
    parts.append(
        f'<rect x="{level_x}" y="{level_y - 10}" width="20" height="20" fill="{escape(str(untenable["color"]))}" stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{level_x + 27}" y="{level_y + 6}" font-size="14" font-family="sans-serif">{escape(str(untenable["label"]))}</text>'
    )
    parts.extend(
        [
            f'<text x="28" y="{legend_y + 70}" font-size="14" font-family="sans-serif" fill="#111">Heatmap classes use fixed planning-cost excess thresholds shared by prior/posterior cases. Delta: outlined blue = decrease, red = increase.</text>',
            f'<text x="28" y="{legend_y + 94}" font-size="14" font-family="sans-serif" fill="#111">Research decision levels only, not safety certification. Fire is frozen during approach; 300 s is a configurable research horizon.</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()

