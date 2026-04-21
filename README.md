# Limitless: Infinity — Repulsive Shells on the GPU

CS184 Spring 2026 final project. A C++/(soon-)CUDA implementation of the pipeline from
*Repulsive Shells* (Sassen, Schumacher, Rumpf, Crane — SIGGRAPH 2024) and its
predecessor *Repulsive Surfaces* (Yu, Brakensiek, Schumacher, Crane — SIGGRAPH Asia
2021). The end goal is collision-free shape interpolation, extrapolation, and averaging
with the tangent-point energy acting as a Riemannian barrier against self-intersection.

**Team:** Atharv Sampath, Michael Pham, Daniel Liu, Elen Papyan.

Reference PDFs live at the repo root (`RepulsiveShells.pdf`, `RepulsiveSurfaces.pdf`);
the original project proposal is `proposal.html`.

## Status

- **Done:** infrastructure (SoA mesh representation, OpenMesh bridge, finite-difference
  gradient checker, procedural test meshes) and CPU tangent-point energy + BVH
  (brute-force and Barnes-Hut versions of both the TPE energy and its gradient, at
  `O(n log n)` via a block-cluster tree). The test binaries gate every piece against
  FD checks, scale covariance, translation invariance, and BVH-aggregate conservation.
- **In progress:** H^s preconditioner, constraints, and remeshing.
- **Next:** shell energy, path-energy trust-region solver, CUDA port.

## Repo layout

```
src/core/     rsh_core library — MeshData, GradCheck, TestMeshes, FaceGeom, BVH, BCT, TPE
src/viewer/   NanoGUI + OpenGL viewer (single mesh or frame_XXXX.obj sequence)
tests/        test_phase0, test_phase1, dump_test_meshes, demo_phase1_flow, plot_phase1_flow.py
assets/       small test meshes (teapot + procedural icospheres / torus)
build.sh      cmake wrapper (works around the devenv-nix missing `-dev` includes)
run.sh        viewer launcher (works around the devenv-nix missing runtime `LD_LIBRARY_PATH`)
devenv.nix    reproducible dev shell (cmake, eigen, openmesh, X11/wayland/GL runtime libs)
```

## One-time setup

```bash
git clone <repo> && cd CS184-Final-Project
git submodule update --init --recursive         # pulls ext/nanogui for the viewer
devenv shell                                     # https://devenv.sh — optional but recommended
```

`devenv` gives you cmake, clang, Eigen, OpenMesh, and the X11/wayland runtime libs
pinned to known-good versions. On a non-nix host any system with cmake, a C++17
compiler, Eigen 3.4+, OpenMesh, and the usual X11 / GL dev packages will also work;
the `build.sh` / `run.sh` wrappers are nix-specific and can be bypassed with plain
`cmake` invocations there.

## Build

```bash
./build.sh                      # configure (if needed) + full build
./build.sh test_phase1          # rebuild a single target
```

Headless builds (skip the viewer — useful on CI or machines without a GL stack):

```bash
cmake -S . -B build -DRSH_ENABLE_VIEWER=OFF
cmake --build build
```

## Test the pieces we have working

### 1. Smoke tests (no GL needed)

```bash
./build/test_phase0             # 20/20 checks — MeshData, GradCheck, TestMeshes
./build/test_phase1             # 47/47 checks — FaceGeom, TPE brute, BVH, BCT, BH energy/grad
```

Both run in well under a second and are the regression gate for everything built on top of them.

### 2. View a single mesh

```bash
./build/dump_test_meshes        # writes assets/icosphere_{0..3}.obj, assets/torus.obj
./run.sh assets/icosphere_3.obj
```

Viewer controls: left-click-drag orbits around the target, right-click-drag pans,
scroll zooms, WASD moves the camera. Blinn-Phong shading with a fixed directional
light.

### 3. Run the gradient-flow demo and play the result back

This is the end-to-end demo of the CPU TPE pipeline: a Barnes-Hut TPE gradient drives
an Armijo-line-searched `L^2` descent on an **ellipsoid** (icosphere(2) with axis
scales `(1.7, 0.75, 0.75)` — 162 verts / 320 tris, procedurally generated, deterministic).
A `frame_XXXX.obj` per iteration and an `energy.csv` are written to `out/phase1_flow/`,
and the viewer can load the directory directly to play the sequence back.

```bash
./build/demo_phase1_flow        # writes out/phase1_flow/ (run from repo root)
./run.sh out/phase1_flow        # viewer plays the sequence with a scrubber + play/pause
```

Playback controls on top of the normal camera controls:

- `Space` — play / pause
- `←` / `→` — step one frame
- `Home` / `End` — jump to first / last frame
- `fps` slider — playback speed (default 24)
- scrubber — seek anywhere in the sequence

What you should see playing back: the ellipsoid rounds toward a sphere (the
genus-0 TPE minimizer) over ~95 iterations. Energy drops several fold then
stalls well above the round-sphere minimum, with the Armijo step size τ
collapsing to ~1e-12. The stall is the *expected* `L^2` failure mode (RSu
Fig. 5) — exactly what motivates the H^s preconditioner that comes next.

The demo runs from the repo root in under a minute on a single CPU core
(320 triangles through a `θ = 0.5` Barnes-Hut eval each iteration).

Optional 4-panel summary plot (energy log, grad norm, step size, XZ silhouettes at
0 / 25 / 50 / 75% of the run):

```bash
python3 tests/plot_phase1_flow.py out/phase1_flow/energy.csv
```

## References

- Sassen, Schumacher, Rumpf, Crane. *Repulsive Shells.* SIGGRAPH 2024.
  (`RepulsiveShells.pdf`)
- Yu, Brakensiek, Schumacher, Crane. *Repulsive Surfaces.* SIGGRAPH Asia 2021.
  (`RepulsiveSurfaces.pdf`)
- Project proposal: `proposal.html`.
- Reference CPU implementation (comparison only, not vendored):
  [HenrikSchumacher/Repulsor](https://github.com/HenrikSchumacher/Repulsor).
