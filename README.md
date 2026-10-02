# IndoorFireWorld

Phases P0-P2 establish a deterministic digital scene and reduced-order fire
simulation for the approximate W1 4F Makerspace. The authoritative geometry is
[`configs/scene_w1_4f.yaml`](configs/scene_w1_4f.yaml); the file uses the JSON
subset of YAML so it can be loaded with the Python standard library and without
an environment-specific YAML dependency.

The primary representation is a continuous world frame (`x`, `y` in metres)
rasterized into a `[row, col]` grid. Semantic, occupancy, and navigation layers
are separate. Any floor-plan image is visualization-only and is never sampled
to create occupancy.

## Run

From the repository root:

```bash
python scripts/visualize_w1_scene.py
python -m unittest discover -s tests -v
python scripts/run_p2_fire_simulation.py
```

The visualization is written to `artifacts/w1_4f_scene.svg`. Use `--show` to
open it in the default browser.

P2 numerical snapshots are written under `artifacts/p2_fire/`. Development
machines that cannot execute code should instead synchronize the repository to
HPC and run `bash scripts/hpc_validate_p2.sh` there.

## Current scope

The `fire` package implements deterministic temperature, smoke, and CO
transport with visibility derived from smoke. The hidden ground-truth simulator
and `world_model.PhysicsPrior` are independent instances driven by deliberately
mismatched synthetic configurations. Belief, uncertainty, hazard, and
planning-cost concepts remain distinct. No observation assimilation, hazard
scoring, or planning behavior is implemented.

## P2 numerical model

For smoke and CO, the solver applies
`dq/dt = D laplacian(q) - u dot grad(q) + source - decay*q` using an explicit
finite-volume diffusion stencil and first-order upwind advection. Temperature
uses the same equation for its anomaly above ambient, with relaxation replacing
pollutant decay. Solid and outer-domain faces use zero normal flux; occupied and
unknown cells are excluded from transport. Fields are clamped non-negative.

Before initialization and every step, a conservative monotonicity condition is
checked:
`dt * (4D/dx^2 + (abs(ux)+abs(uy))/dx + decay) <= 1`.
Violation raises `ValueError` instead of advancing an unstable field.

The microwave electrical fault is only an ignition label. Its heat-release
curve is `Q(t) = min(alpha*t^2, Q_max)`; no electrical or CFD physics is
modeled. All P2 parameters live in `configs/fire_truth.yaml` and
`configs/fire_prior.yaml`, both JSON-subset YAML files.

## Geometry assumptions

No surveyed W1 4F dimensions were available in this workspace. The config uses
an explicitly labeled 20 m by 14 m approximation, approximate internal walls,
equipment, doors, exits, sensors, ignition point, and initial UGV pose. Replace
the config values when surveyed measurements become available; Python code does
not embed this geometry.

Fire coefficients, yields, ventilation, and growth rates are also synthetic,
uncalibrated research assumptions. The ground-truth values are intentionally
stronger than the prior so later observation-assimilation experiments have a
controlled deterministic model mismatch. Ventilation is a constant 2D vector
in P2. The microwave is a lumped, cell-local source; its configured gains and
yields convert heat-release rate into prototype field rates rather than
calibrated physical mass or energy units.
