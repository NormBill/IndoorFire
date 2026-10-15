"""Dependency-free, deterministic SVG rendering of the configured scene."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

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
    output_path: str | Path,
) -> Path:
    """Render the primary P8 approach and retreat comparison on the source map."""
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

    width = registration.source_width_px
    height = registration.source_height_px
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 94}" viewBox="0 0 {width} {height + 94}">',
        f'<image href="data:image/png;base64,{encoded}" x="0" y="0" width="{width}" height="{height}"/>',
    ]
    routes = (
        (summary["ugv_approach"], "#7c3aed", "8 7", 5, "UGV reachability path"),
        (summary["prior_retreat"]["route"], "#2563eb", "", 6, "Prior retreat"),
        (summary["updated_retreat"]["route"], "#dc2626", "", 6, "Updated retreat"),
    )
    for route, color, dash, stroke_width, _ in routes:
        if route["feasible"]:
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<polyline points="{path_points(route)}" fill="none" stroke="{color}" '
                f'stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"{dash_attribute}/>'
            )
    point = summary["observation_point"]["resolved_position_m"]
    px, py = registration.metric_to_pixel(point["x"], point["y"])
    parts.extend(
        [
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="11" fill="#f59e0b" stroke="white" stroke-width="3"/>',
            f'<text x="{px + 15:.2f}" y="{py - 12:.2f}" font-size="18" font-family="sans-serif" font-weight="bold" fill="#92400e">geometry-offset observation</text>',
        ]
    )
    for exit_id in summary["prior_retreat"]["candidates"]:
        semantic = scene.semantic.by_id(exit_id)
        if semantic.position is None:
            continue
        ex, ey = registration.metric_to_pixel(*semantic.position)
        parts.append(f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="9" fill="#16a34a" stroke="white" stroke-width="3"/>')
        parts.append(
            f'<text x="{ex + 12:.2f}" y="{ey - 10:.2f}" font-size="15" font-family="sans-serif" font-weight="bold" fill="#166534">{escape(exit_id)}</text>'
        )
    caption_y = height + 24
    parts.extend(
        [
            f'<rect x="0" y="{height}" width="{width}" height="94" fill="#ffffff"/>',
            f'<line x1="28" y1="{caption_y}" x2="88" y2="{caption_y}" stroke="#7c3aed" stroke-width="5" stroke-dasharray="8 7"/><text x="100" y="{caption_y + 6}" font-size="17" font-family="sans-serif">UGV reachability</text>',
            f'<line x1="310" y1="{caption_y}" x2="370" y2="{caption_y}" stroke="#2563eb" stroke-width="6"/><text x="382" y="{caption_y + 6}" font-size="17" font-family="sans-serif">Prior retreat</text>',
            f'<line x1="560" y1="{caption_y}" x2="620" y2="{caption_y}" stroke="#dc2626" stroke-width="6"/><text x="632" y="{caption_y + 6}" font-size="17" font-family="sans-serif">Updated retreat</text>',
            f'<text x="28" y="{height + 55}" font-size="15" font-family="sans-serif" fill="#111">Snapshot closed loop: fire evolution is frozen during UGV reachability validation and prior/posterior comparison.</text>',
            f'<text x="28" y="{height + 78}" font-size="15" font-family="sans-serif" fill="#111">300 s is a configurable research horizon, not W1 measured arrival or firefighter interior-deployment time.</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")
    return output.resolve()

