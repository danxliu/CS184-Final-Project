# Limitless: Infinity — Repulsive Shells from Scratch

CS184 Spring 2026 final project. This repository is a C++ implementation of major
pieces of the *Repulsive Shells* pipeline (Sassen, Schumacher, Rumpf, Crane —
SIGGRAPH 2024), with background from *Repulsive Surfaces* (Yu, Brakensiek,
Schumacher, Crane — SIGGRAPH Asia 2021).

The current codebase is **CPU/OpenMP-first**. A CUDA/GPU backend is still future
work.

**Team:** Atharv Sampath, Michael Pham, Daniel Liu, Elen Papyan.

Reference PDFs are in `references/`:
- `references/RepulsiveShells.pdf`
- `references/RepulsiveSurfaces.pdf`

Project proposal: `websites/proposal.html`.

## Current status (May 2026)

### Implemented in `rsh_core`

- Mesh infrastructure: `MeshData`, OpenMesh bridge, procedural test meshes, FD gradient checking.
- TPE stack: brute-force and Barnes-Hut TPE energy/gradient, BVH + block-cluster tree, adaptive far-field cache.
- Shell/path energies: `ShellEnergy`, `PathEnergy`.
- Solvers: trust-region interpolation (`TrustRegionSolver`), extrapolation (`ExtrapolationSolver`), H^s-preconditioned descent (`HsPreconditioner`), and optimization wrapper (`OptimizeTPE`).
- Constraints/remeshing: barycenter/pin-mask constraints and split-collapse + Delaunay + tangential smoothing remeshing.

### Tests and demos currently present

- Tests: `test_phase0`, `test_phase1`, `test_phase2`, `test_hs_kernel_exponent`.
- Demos: `demo_phase1_flow`, `demo_phase2_hs_flow`, `demo_canonical_genus0`, `demo_gallery_genus{0..5}`, `demo_compression`.
- Viewer: NanoGUI/OpenGL viewer supports single meshes and `frame_XXXX.obj` sequences.

### Partially wired / optional surfaces

- `demo_interpolation` and `demo_extrapolation` targets are currently commented out in `CMakeLists.txt`.
- Canonical embedding demo is currently explicit genus-0 smoke (`demo_canonical_genus0`).
- Optional tooling:
  - `RSH_ENABLE_POLYSCOPE=ON` for `polyscope_viewer`
  - `RSH_HAVE_REPULSOR=ON` for `cross_validate_repulsor` (requires sibling `../Repulsor`)

### Not yet implemented

- CUDA/GPU acceleration backend.

## Repo layout

```text
src/core/     rsh_core library (MeshData, TPE, BVH/BCT, Shell/Path energy, H^s, remeshing, solvers)
src/viewer/   NanoGUI + OpenGL viewer
tests/        phase tests + demo binaries + plot helper(s)
tools/        diagnostics + optional viewers/cross-validation
assets/       example meshes
references/   paper PDFs
websites/     proposal and project web docs
build.sh      cmake wrapper (nix include/lib path convenience)
run.sh        viewer launcher (nixGL wrapper)
```

## One-time setup

```bash
git clone <repo> && cd CS184-Final-Project
git submodule update --init --recursive
devenv shell    # optional but recommended: https://devenv.sh
```

## Build

```bash
./build.sh
./build.sh test_phase2 demo_phase2_hs_flow
```

Headless build (skip NanoGUI viewer):

```bash
cmake -S . -B build -DRSH_ENABLE_VIEWER=OFF
cmake --build build
```

Optional targets:

```bash
cmake -S . -B build -DRSH_ENABLE_POLYSCOPE=ON
cmake --build build --target polyscope_viewer
```

```bash
cmake -S . -B build -DRSH_HAVE_REPULSOR=ON
cmake --build build --target cross_validate_repulsor
```

## Tests

Run the core regression suite (no GUI required):

```bash
./build/test_phase0
./build/test_phase1
./build/test_phase2
./build/test_hs_kernel_exponent
```

`test_hs_kernel_exponent` supports a lighter mode:

```bash
./build/test_hs_kernel_exponent --skip-heavy
# or: RSH_SKIP_HEAVY_TESTS=1 ./build/test_hs_kernel_exponent
```

## Demos

### 1) Viewer sanity check

```bash
./build/dump_test_meshes
./run.sh assets/icosphere_3.obj
```

### 2) L2 flow baseline (expected to stall)

```bash
./build.sh demo_phase1_flow
./build/demo_phase1_flow          # writes out/phase1_flow/{frame_XXXX.obj,energy.csv}
./run.sh out/phase1_flow
python3 tests/plot_phase1_flow.py out/phase1_flow/energy.csv
```

### 3) H^s-preconditioned flow

```bash
./build.sh demo_phase2_hs_flow
./build/demo_phase2_hs_flow       # writes out/phase2_hs_flow/{frame_XXXX.obj,energy.csv}
./run.sh out/phase2_hs_flow
```

### 4) Canonical genus-0 smoke

```bash
./build.sh demo_canonical_genus0
./build/demo_canonical_genus0     # writes out/canonical_genus0/{frame_XXXX.obj,energy.csv}
./run.sh out/canonical_genus0
```

### 5) Gallery demos (fixed genus families)

```bash
./build.sh demo_gallery_genus0 demo_gallery_genus1 demo_gallery_genus2 demo_gallery_genus3 demo_gallery_genus4 demo_gallery_genus5
./build/demo_gallery_genus0       # writes out/gallery_genus0/
./build/demo_gallery_genus1       # writes out/gallery_genus1/
./build/demo_gallery_genus2       # writes out/gallery_genus2/
./build/demo_gallery_genus3       # writes out/gallery_genus3/
./build/demo_gallery_genus4       # writes out/gallery_genus4/
./build/demo_gallery_genus5       # writes out/gallery_genus5/
```

### 6) Compression demo

```bash
./build.sh demo_compression
./build/demo_compression          # writes out/compress_sequence/
./run.sh out/compress_sequence
```

Viewer controls: orbit/pan/zoom camera + sequence scrubber/playback for frame directories.

## References

- Sassen, Schumacher, Rumpf, Crane. *Repulsive Shells.* SIGGRAPH 2024. (`references/RepulsiveShells.pdf`)
- Yu, Brakensiek, Schumacher, Crane. *Repulsive Surfaces.* SIGGRAPH Asia 2021. (`references/RepulsiveSurfaces.pdf`)
- Project proposal: `websites/proposal.html`
- Reference CPU implementation (comparison only, not vendored): [HenrikSchumacher/Repulsor](https://github.com/HenrikSchumacher/Repulsor)
