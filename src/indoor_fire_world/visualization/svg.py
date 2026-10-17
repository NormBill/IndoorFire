"""Dependency-free, deterministic SVG rendering of the configured scene."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import numpy as np

from indoor_fire_world.scene.models import Polygon, Scene


def render_p8_v2_forecast_svg(
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

