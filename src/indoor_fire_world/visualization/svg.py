"""Dependency-free, deterministic SVG rendering of the configured scene."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import numpy as np

from indoor_fire_world.scene.models import Polygon, Scene


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
    common_finite = np.isfinite(prior_cost) & np.isfinite(posterior_cost)
    if not np.any(common_finite):
        raise ValueError("P8 risk map requires finite traversable planning costs")
    common_min = float(min(prior_cost[common_finite].min(), posterior_cost[common_finite].min()))
    common_max = float(max(prior_cost[common_finite].max(), posterior_cost[common_finite].max()))
    common_range = max(common_max - common_min, 1e-12)
    prior = np.where(common_finite, (prior_cost - common_min) / common_range, 0.0)
    posterior = np.where(
        common_finite,
        (posterior_cost - common_min) / common_range,
        0.0,
    )
    fields = {
        "prior": prior,
        "posterior": posterior,
        "delta": posterior - prior,
    }
    titles = {
        "prior": "Prior planning risk/cost",
        "posterior": "Posterior planning risk/cost after UGV observation",
        "delta": "Planning risk/cost delta (posterior - prior)",
    }
    if not panels or any(name not in fields for name in panels):
        raise ValueError("P8 risk-map panels must be prior, posterior, or delta")

    sequential = ("#FFF7EC", "#FEE8C8", "#FDBB84", "#FC8D59", "#D7301F", "#7F0000")
    diverging_negative = ("#2166AC", "#67A9CF", "#D1E5F0")
    diverging_positive = ("#FDDBC7", "#EF8A62", "#B2182B")

    def risk_color(value: float, *, delta: bool, scale: float) -> str:
        if delta:
            if abs(value) <= 1e-12:
                return "#F7F7F7"
            normalized = 0.0 if scale <= 1e-12 else min(1.0, abs(value) / scale)
            palette = diverging_positive if value >= 0.0 else diverging_negative
            index = min(len(palette) - 1, int(normalized * len(palette)))
            return palette[index]
        index = min(len(sequential) - 1, int(max(0.0, min(1.0, value)) * len(sequential)))
        return sequential[index]

    source_width = registration.source_width_px
    source_height = registration.source_height_px
    panel_scale = 1.0 if len(panels) == 1 else 0.5
    panel_width = source_width * panel_scale
    panel_height = source_height * panel_scale
    width = int(panel_width * len(panels))
    height = int(panel_height + 142)
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
                if abs(value) <= 1e-9:
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
            edge_value = float(np.mean([field[cell] for cell in edge_cells]))
            edge_color = risk_color(edge_value, delta=is_delta, scale=delta_scale)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{edge_color}" stroke-width="13" opacity="0.95" stroke-linecap="round"/>'
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
            f'<text x="28" y="{legend_y + 34}" font-size="14" font-family="sans-serif" fill="#111">Cell color encodes planning risk/cost (physical hazard + uncertainty) on one shared scale. Delta: outlined blue = decrease, red = increase.</text>',
            f'<text x="28" y="{legend_y + 58}" font-size="14" font-family="sans-serif" fill="#111">Snapshot model: fire is frozen during approach validation and comparison. 300 s is a configurable research horizon, not measured W1 response time.</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()

