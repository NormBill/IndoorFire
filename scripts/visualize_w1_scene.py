#!/usr/bin/env python3
"""Render the deterministic W1 4F scene without simulating fire."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indoor_fire_world.scene import load_scene
from indoor_fire_world.visualization import render_scene_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "scene_w1_4f.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "w1_4f_scene.svg")
    parser.add_argument("--show", action="store_true", help="Open the generated SVG in the default browser")
    args = parser.parse_args()
    output = render_scene_svg(load_scene(args.config), args.output)
    print(f"Wrote {output}")
    if args.show:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()

