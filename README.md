# IndoorFireWorld

Phases P0-P8 establish a deterministic digital scene, reduced-order fire
simulation, UGV point sensing, local belief correction, and separate hazard,
uncertainty, and planning-cost fields for the approximate W1 4F Makerspace,
followed by risk-aware entry/retreat planning and a controlled closed-loop
replanning experiment and a reproducible evaluation suite.
P0-P7 retain the frozen schematic geometry in
[`configs/scene_w1_4f.yaml`](configs/scene_w1_4f.yaml), while P8 uses the
image-registered migration target
[`configs/scene_w1_4f_registered_v1.yaml`](configs/scene_w1_4f_registered_v1.yaml).
Configuration files use the JSON subset of YAML so they load without an
environment-specific YAML dependency.

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
python scripts/run_experiment.py --config configs/experiments/p7_baseline.yaml
python scripts/run_experiment_suite.py
python scripts/run_p8_registered_closed_loop.py
```

The visualization is written to `artifacts/w1_4f_scene.svg`. Use `--show` to
open it in the default browser.

P2 numerical snapshots are written under `artifacts/p2_fire/`. Development
machines that cannot execute code should instead synchronize the repository to
HPC and run `bash scripts/hpc_validate_p7.sh` there. The P3 demo writes its
machine-readable comparison to `artifacts/p3_assimilation/result.json`; the P4
demo writes arrays and `summary.json` under `artifacts/p4_hazard/`. P5 writes
the mission result plus JSON/CSV weight-sensitivity summaries under
`artifacts/p5_planning/`. P6 writes its causal before/after report to
`artifacts/p6/summary.json`. P7 writes per-run artifacts plus aggregate JSON and
CSV results under `artifacts/p7/`.
P8 writes its 4-by-4 registered-map snapshot matrix, primary-case arrays, a
three-panel registered overlay, and separate prior, posterior, and delta risk
maps under `artifacts/p8_registered_closed_loop/`.

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

## P8 registered-map snapshot closed loop

`configs/p8_registered_closed_loop.yaml` integrates the Gate 1 registered map,
Gate 2 staged microwave electrical-fire truth/prior pair, a geometry-defined
UGV observation, local World Model correction, and retreat-only replanning.
The UGV approach and human/robot retreat are constrained to the directed route
graph transcribed from the green evacuation arrows in the registered map; the
planner may not cut across nominally open floor cells or travel against an
arrow.
Each case is a snapshot closed loop: UGV travel validates reachability, but the
fire fields remain frozen during the approach check and prior/posterior route
comparison. P8 is not a fully time-coupled robot-fire simulation.

All four observation points are frozen before the truth simulator is created.
They are specified only as fixed offsets from the registered microwave anchor
and checked against map traversability; plume, truth-hazard, posterior, and
route-result fields cannot select or move them. The normal matrix evaluates
four fire ages against all four points. The primary case also includes no
observation, observation-without-fusion, normal-fusion, and truth-oracle
controls.

The configured 300 s value is a **configurable pre-response research horizon
motivated by the 5-min fire-station layout principle**. It is not a measured W1
response time, an actual fire-engine arrival time, or the time at which
firefighters complete interior deployment. The public five-minute principle is
used only as motivation for a replaceable research horizon. The registered
40.0 m by 12.5 m scale also remains provisional.

P8 separates `ugv_operability_threshold` from `retreat_hazard_threshold`, even
when both have the same numerical value. Decision improvement is evaluated as
`J_post(prior_route) - J_post(updated_route)`, with both fixed paths evaluated
on the identical posterior planning field. If the old route is posterior-
infeasible, no finite delta is invented; the result is reported as a
feasibility improvement when the updated route is feasible. Invalid geometry
or approach cases use `invalid_observation_case` and remain distinct from a
valid scenario with no feasible prior retreat.

The visualization classifies `planning_cost - base_cost` into five fixed,
configuration-defined decision-risk levels: Minimal, Guarded, Elevated, High,
and Critical. Prior and posterior heatmaps use the same absolute thresholds and
colors across cases; the signed cost-delta panel remains zero-centred. These are
research decision levels, not firefighter-tenability or safety-certification
thresholds. The underlying decision field combines the separately retained
physical hazard and uncertainty components. Route edges are colored per short
grid segment rather than by a whole-edge average, while the initial and updated
retreat remain overlaid on the directed graph. With the current strict topology, the primary case has
only the centre-exit branch available after the selected observation point, so
the posterior route remains unchanged (`delta_j = 0`). This replaces the
earlier free-grid reroute, which did not respect the evacuation arrows. The
World Model and risk fields can still change without implying a route change
when the registered topology offers no alternative downstream branch.
The artifact writer retains the required `t=100 s` primary comparison and also
writes posterior level heatmaps for every valid evaluation time at the primary
observation point, making later fire-field expansion visible without rescaling
or altering the underlying values.

Run the complete P0-P8 validation and artifact generation on HPC with:

```bash
bash scripts/hpc_validate_p8.sh
```

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

## P7 reproducible experiments and metrics

`configs/experiments/p7_suite.yaml` freezes Baseline v1 and all declared P7
ablations. The reusable runner applies only validated configuration overrides
to the existing P6 pipeline. Normal runs plan from the posterior belief; the
perfect-information reference is explicitly marked and is the only mode that
may provide the hidden truth field to the planner.

The suite includes no-assimilation, no-observation, hazard-only, baseline
uncertainty-aware, and perfect-information runs; low/medium/high model mismatch;
zero/low/medium seeded sensor noise; and sensitivity runs over existing
assimilation gain, kernel, and uncertainty-reduction parameters. These values
are predefined for sensitivity analysis and are not automatically optimized.

Every run receives a directory named from its scenario ID and configuration
hash. It contains the exact config snapshot, complete summary, field metrics,
route metrics, and path hashes. `results.json` and `results.csv` contain one
machine-readable row per run, while `aggregate_stats.json` provides concise
suite statistics. Estimation metrics include local and global MAE against hidden
truth for evaluation only. Decision improvement is the posterior cost of the
old route minus the posterior cost of the replanned route.

The frozen P6 regression is represented accurately: A and B have equal prior
cost, A is selected by deterministic tie-breaking, and B becomes strictly
cheaper under the posterior field.

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

## Gate 1: registered W1 4F base map

`configs/scene_w1_4f_registered_v1.yaml` is a separate migration target and
does not replace the frozen P0-P7 schematic scene. It registers the supplied
1713 by 849 pixel evacuation map, digitizes the irregular floor envelope and
major fixed obstacles with polygon occupancy, and records three exits, the
microwave anchor, and the UGV start. The original image is retained at
`assets/maps/w1_4f_evacuation_base.png` so every metric feature can be checked
against its source.

The registration deliberately reports
`provisional_aspect_registered_scale_pending`. Its 40.0 m by 12.5 m metric
bounds preserve the image aspect and support a 0.25 m working grid, but are not
surveyed dimensions. Replace `reference_map.metric_bounds_m` after obtaining
one reliable long horizontal distance; do not report route lengths from this
map as measured W1 distances before that calibration.

Generate the self-contained overlay with:

```bash
python scripts/validate_gate1_gate2.py
```

## Gate 2: staged microwave electrical-fire scenario

The registered-map scenario is declared in
`configs/microwave_electrical_scenario_v1.yaml`. Its schema-version 2 truth and
prior fire configurations use four deterministic phases:

1. overheating before room-scale ignition;
2. concealed, smoke-rich smoldering;
3. internal flaming;
4. enclosure breakout.

Each phase freezes its time interval, heat-release-rate model, temperature
gain, smoke yield, CO yield, interpretation, and evidence status. The prior is
intentionally delayed and weaker than truth, giving the UGV correction a
declared hidden-fire mismatch. Numerical values remain literature-structured
research assumptions rather than W1 appliance calibration.

Secondary cable ignition and flame spread are explicitly disabled. They require
cable material, routing, incident heat flux, and a separately validated
THIEF/CHRISTIFIRE-compatible parameterization; the current phase model must not
be presented as a universal electrical-wire propagation model.

Validate Gate 1, Gate 2, and all frozen regressions on HPC with:

```bash
bash scripts/hpc_validate_gate1_gate2.sh
```
