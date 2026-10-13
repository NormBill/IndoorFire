# IndoorFireWorld

Phases P0-P6 establish a deterministic digital scene, reduced-order fire
simulation, UGV point sensing, local belief correction, and separate hazard,
uncertainty, and planning-cost fields for the approximate W1 4F Makerspace,
followed by risk-aware entry/retreat planning and a controlled closed-loop
replanning experiment.
The authoritative geometry is
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
python scripts/run_p3_assimilation_demo.py
python scripts/run_p4_hazard_demo.py
python scripts/run_p5_weight_sensitivity.py
python scripts/run_p5_planning_demo.py
python scripts/run_p6_closed_loop_demo.py
```

The visualization is written to `artifacts/w1_4f_scene.svg`. Use `--show` to
open it in the default browser.

P2 numerical snapshots are written under `artifacts/p2_fire/`. Development
machines that cannot execute code should instead synchronize the repository to
HPC and run `bash scripts/hpc_validate_p6.sh` there. The P3 demo writes its
machine-readable comparison to `artifacts/p3_assimilation/result.json`; the P4
demo writes arrays and `summary.json` under `artifacts/p4_hazard/`. P5 writes
the mission result plus JSON/CSV weight-sensitivity summaries under
`artifacts/p5_planning/`. P6 writes its causal before/after report to
`artifacts/p6/summary.json`.

## Current scope

The `fire` package implements deterministic temperature, smoke, and CO
transport with visibility derived from smoke. The hidden ground-truth simulator
and `world_model.PhysicsPrior` are independent instances driven by deliberately
mismatched synthetic configurations. A predefined-waypoint UGV can sample an
immutable truth snapshot, and the deterministic P3 baseline corrects a separate
belief mean while reducing a separate uncertainty field. Hazard and
planning-cost concepts remain distinct. P4 converts the posterior belief into
an interpretable physical hazard field, separately normalizes heterogeneous
uncertainty, and combines both only in a decision-cost field. P5 performs
static deterministic route search over those precomputed fields. P6 adds one
controlled, single-observation replan using the same P2-P5 components; it does
not add a new planning algorithm, continuous/time-dependent route switching, or
active sensing.

## P6 controlled closed-loop replanning

`configs/p6_closed_loop.yaml` freezes the observation time and location,
assimilation settings, mission, and planning weights before simulation. Its
referenced truth and prior fire configs encode a deterministic hidden mismatch:
the prior underestimates growth, pollutant yields, and plume transport near the
predefined Route A probe. The fixed planning weights emphasize physical hazard
over the uncertainty reduction so the observation is not rewarded merely for
making the local estimate more certain.

The P6 demo first plans Route A from the physics-prior belief. A predefined UGV
waypoint then samples hidden truth, the existing spatial assimilator updates the
belief, and the existing hazard, uncertainty, planning-cost, and A* components
produce the posterior Route B. No hazard cells, path penalties, or weights are
changed after the observation.

The JSON report includes parameter fingerprints, observation residuals,
prior/posterior local fields, both optimized missions, and four fixed-path
counterfactuals: A and B evaluated under both prior and posterior fields. Its
acceptance block machine-checks the causal chain and requires a route-hash
change only for this declared controlled scenario.

## P5 entry and retreat planning

`configs/planning_mission.yaml` explicitly defines the entry start, inspection
target, candidate exits, and the hazard/uncertainty weights used by the
sensitivity script. The deterministic four-connected A* implementation consumes
the P4 planning cost, forbidden mask, hazard, and uncertainty arrays without
recomputing them. Its admissible heuristic uses the minimum traversable cost
density; equal-cost choices are resolved deterministically by grid coordinates.

Each move charges the destination cell's cost density multiplied by the 0.25 m
cell resolution. Route output includes feasibility and rejection reasons,
geometric length, total and decomposed base/hazard/uncertainty costs, integrated
hazard and uncertainty, maximum hazard, and a path hash. Retreat planning
evaluates every configured exit and selects the lowest-cost feasible route; a
mission is feasible only when both entry and retreat are feasible.

The sensitivity script evaluates configured hazard and uncertainty weights and
reports each route and exit outcome without declaring an automatically optimal
weight pair. This is an analysis aid, not parameter tuning or policy learning.

## P4 hazard, uncertainty, and planning cost

`configs/hazard.yaml` defines unit-documented safe/danger thresholds and
uncertainty reference scales. Temperature, CO, and smoke use increasing
piecewise-linear severity; visibility uses the reverse direction. Each severity
is clipped to `[0,1]`, and physical hazard is their conservative maximum. The
dominant modality is retained for explanations, while a zero-hazard cell uses
the explicit label `none`. The synthetic CO and smoke
units from P2 are preserved rather than relabeled as ppm or optical density.

Raw uncertainty modalities are variance proxies with different units, so each
is first normalized as `clip(U_i/U_i_ref, 0, 1)`. Normalized uncertainty is the
maximum across modalities and never enters the physical hazard definition.

`configs/planning_cost.yaml` defines
`C = base_cost + hazard_weight*H + uncertainty_weight*U`. Occupied, unknown, and
physically untenable cells at or above the configured threshold receive infinite
cost. Their underlying hazard values are retained. All thresholds and weights
are transparent research assumptions, not certified tenability standards.

## P3 sensing and correction

`configs/p3_assimilation.yaml` contains sensor standard deviations, random
seed, initial per-cell uncertainty, predefined inspection waypoints, Gaussian
kernel parameters, and correction gain. The default controlled demo uses zero
noise; nonzero noise uses a seeded NumPy generator.

For an observed modality with residual `r`, the posterior mean is
`mean_post = mean_prior + gain * K(distance) * r`, where
`K(d) = exp(-d^2 / (2*sigma^2))` inside the configured radius and zero outside.
Only free cells and observed modalities are updated. CO, smoke, and visibility
are clamped non-negative; temperature is bounded below by the configured
physical minimum. Ground-truth and physics-prior visibility remain derived from
smoke as in P2; the P3 belief keeps visibility as a separate observed estimate
so a visibility measurement can be corrected without silently changing smoke.

Uncertainty values are diagonal epistemic variance proxies in squared native
units, not calibrated probabilities. For each observed modality,
`U_post = U_prior * (1 - reduction_fraction*K)`. They remain non-negative and
are not used as hazard or planning-cost values.

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
