# Ball-Through-Tube Teleport Debug Notes

Date: 2026-05-06

This note summarizes the current fix for the Repulsive Shells ball-through-
barrier-with-hole demo. The important bug was not just bad beta tuning: our
path energy did not match the released Repulsive Shells implementation.

## Target

We are reproducing the Figure 4 style setup:

- translated sphere on the left and right of a fixed wall,
- circular hole centered on the translation axis,
- the released tunnel initializer: all free frames start at the left sphere,
  followed by the fixed right endpoint,
- no SDF guard in the paper preset,
- self TPE and obstacle TPE as graph-manifold coordinates,
- rigid translation and rotation regularizers from Section 7.4.3.

The default paper preset is:

```bash
./build/demo_phase3_ball_tube_interp --paper-fig4
```

It writes to:

```text
out/ball_tube_interp_paper_fig4
```

## What Was Wrong

The old implementation used a single combined graph coordinate:

```text
Phi_total = self_tpe_weight * Phi_self
          + tpe_barrier_weight * Phi_obstacle
          + sdf_weight * Phi_sdf

E_graph = graph_beta * (Phi_total[k-1] - Phi_total[k])^2
```

That is not what the released implementation does. In
`repulsive-shells/src/AugmentedSurfaceInterpolation.cpp`, self TPE and
obstacle TPE are separate `DifferencePathEnergy` terms and are weighted
after squaring:

```text
E_graph =
  tpeWeight      * (Phi_self[k-1]     - Phi_self[k])^2
+ obstacleWeight * (Phi_obstacle[k-1] - Phi_obstacle[k])^2
```

This matters because summing coordinates before squaring creates cross terms
and lets one potential change cancel another. It also made CLI weights mean
"scale Phi before differencing" instead of "weight this graph coordinate",
which is not the paper implementation.

The paper preset also had an initialization mismatch. The released
`Interpolation_SphereThroughTunnel.yaml` provides three `initFile` entries,
and all three point to `TranslationLoop4Sphere_L1.ply`. With `numSteps: 4`,
the initial path is therefore:

```text
[start, start, start, start, end]
```

Our earlier `--paper-fig4` default used:

```text
[start, start, end, end, end]
```

That pre-inserted a middle-frame centroid jump before the optimizer ever ran.
The preset now defaults to `init_mode=start-stack` to match the YAML.

## Current Code Fix

`PathEnergy` now evaluates separate graph coordinates:

```text
self_phi     = tpe_inner_weight * raw_self_tpe
barrier_phi  = tpe_inner_weight * raw_surface_obstacle_tpe
obstacle_phi = raw_sdf_guard
```

and assembles:

```text
E_graph = graph_beta * (
    self_tpe_weight    * d(self_phi)^2
  + tpe_barrier_weight * d(barrier_phi)^2
  + obstacle_weight    * d(obstacle_phi)^2
)
```

`TrustRegionSolver` was updated to use the same coordinate split in its
Gauss-Newton graph Hessian and optional graph-residual Hessian.

The compatibility fields `phi_per_frame` and `grad_phi_per_frame` still exist
for older callers. They are sqrt-weighted scalar fallbacks, so they preserve
the new weighting when exactly one graph coordinate is active. New
interpolation code should use the coordinate-specific arrays.

## Paper Preset Alignment

The released config is
`../repulsive-shells/data/Interpolation_SphereThroughTunnel.yaml`:

```text
numSteps: 4
numLevels: 2
TPE alpha: 6
TPE beta: 12
innerWeight: 1e-6
theta: 0.5
thetaNear: 10
bendingWeight: 1e-2
elasticWeight: 1e-4
tpeWeight: 1e-2
obstacleWeight: 1e-2
barycenterWeight: 1
rotationWeight: 1e-3
```

Our `--paper-fig4` preset now maps those into our pipeline:

```text
num_frames = 5
init_mode = start-stack
temporal_refinement_levels = 2
graph_beta = 1
tpe_inner_weight = 1e-6
self_tpe_weight = 1e-2
tpe_barrier_weight = 1e-2
rigid_translation_weight = 1
rigid_rotation_weight = 1e-3
shell_membrane_weight = 1e-4
shell_bending_weight = 1e-6 / 3
shell_bending_model = simple_angle
```

The `1/3` bending-weight compensation is because our bending kernel divides
by one-third adjacent face area, while GOAST's `SimpleBendingEnergy` divides
by the full adjacent-area sum.

The geometry is also scaled from the released assets. The official sphere
radius is about `1.1`, the hole radius is about `0.94887`, and the wall spans
the middle `4` units of an `8.2` unit translation. For our default
`ball_radius = 0.30`, this gives:

```text
hole_radius ~= 0.258783
endpoint_x ~= 1.11818
wall half-thickness ~= 0.545455
wall half-width/height ~= 2.72727
```

That is still a squeeze, but it is much closer to the released demo than the
old `hole_radius = 0.22`, `endpoint_x = 1.80` setup.

## Remaining Differences

We are still not vendoring GOAST or Repulsor. Known differences remain:

- procedural sphere and wall meshes instead of the exact PLY assets,
- our trust-region solver is simpler than their full reduced-Hessian stack,
- our obstacle mesh is much coarser than `Wall_7.ply`,
- no dynamic remeshing in this path solve yet,
- no continuous collision detection, consistent with the project plan.

The main paper-structure mismatch that caused the teleporting investigation
is fixed.

## Validation

These commands pass after the fix:

```bash
./build.sh demo_phase3_ball_tube_interp
./build.sh test_phase3 && ./build/test_phase3
./build.sh test_phase1 && ./build/test_phase1
./build.sh test_phase2 && ./build/test_phase2
./build/demo_phase3_ball_tube_interp --paper-fig4 --check-init-only
```

The init-only paper preset reports a feasible start-stack path:

```text
frames=5
ball_r=0.3
tube_inner_r=0.258783
tube_half_length=0.545455
initial_min_phi_min=0.272727
centroid_x_monotone=1
```

A bounded diagnostic run:

```bash
./build/demo_phase3_ball_tube_interp --paper-fig4 \
  --temporal-refinement-levels 0 --max-tr-iters 30 --max-cg-iters 30
```

now moves the intermediate frames from the start side instead of keeping the
old baked-in midpoint jump. It is still not a clean final solve on the coarse
procedural assets: the last diagnostic run ended with `min_phi=-0.03947` and
`max_centroid_x_gap=1.44244`, so more work remains on robustness / fidelity.

## Relevant Files

```text
src/core/PathEnergy.h
src/core/PathEnergy.cpp
src/core/TrustRegionSolver.cpp
tests/demo_phase3_ball_tube_interp.cpp
```
