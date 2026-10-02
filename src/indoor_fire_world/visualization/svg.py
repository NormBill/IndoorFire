"""Dependency-free, deterministic SVG rendering of the configured scene."""

from __future__ import annotations

from html import escape
from pathlib import Path

from indoor_fire_world.scene.models import Scene


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
            rect = obj.footprint
            x = (rect.x - geometry.origin_x) * scale
            y = height - (rect.y + rect.height - geometry.origin_y) * scale
            color = obstacle_colors.get(obj.kind, "#777")
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

