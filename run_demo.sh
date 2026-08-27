#!/usr/bin/env bash
# Launch the demo windows in Polyscope: genus 3 gallery, then ball-through-tube.
# Close the genus 3 window to launch the ball-through-tube window.

set -e

cd "$(dirname "$0")"

if [[ ! -x ./build/polyscope_viewer ]]; then
  echo "Build polyscope_viewer first:  ./build.sh polyscope_viewer" >&2
  exit 1
fi

if [[ ! -x ./build/precompute_tpe_flow ]]; then
  echo "Build precompute_tpe_flow first:  ./build.sh precompute_tpe_flow" >&2
  exit 1
fi

if [[ -d out/gallery_genus3 ]]; then
  n_frames=$(find out/gallery_genus3 -maxdepth 1 -name 'frame_*.obj' | wc -l)
  n_flows=$(find out/gallery_genus3 -maxdepth 1 -name 'flow_*.vec' | wc -l)
  if [[ "$n_frames" -gt 0 && "$n_flows" -lt "$n_frames" ]]; then
    echo "precomputing TPE flow sidecars for out/gallery_genus3 ($n_flows/$n_frames present)"
    ./build/precompute_tpe_flow out/gallery_genus3 --flow hs
  fi
fi

# Per-window args: directory followed by extra flags for that window.
WINDOWS=(
  "out/gallery_genus3 --fps 60 --flow hs --zoom-out 1.5"
  "out/ball_tube_interp_paper30 --zoom-out 1.5 --flow off"
)

for w in "${WINDOWS[@]}"; do
  read -r -a args <<<"$w"
  d="${args[0]}"
  if [[ ! -d $d ]]; then
    echo "missing: $d" >&2
    continue
  fi
  if [[ "$d" == "out/ball_tube_interp_paper30" ]]; then
    cfg="$d/paper_config.txt"
    if [[ ! -f "$cfg" ]] ||
       ! grep -qx 'obstacle_geometry=slab-hole' "$cfg" ||
       ! grep -qx 'tpe_barrier_enabled=1' "$cfg" ||
       ! grep -Eq '^sdf_guard_weight=0(\.0+)?$' "$cfg" ||
       ! grep -qx 'init_mode=start-stack' "$cfg"; then
      echo "skipping stale/non-paper ball-through-tube output in $d" >&2
      echo "regenerate it with: ./build/demo_phase3_ball_tube_interp --paper-fig4" >&2
      continue
    fi
  fi
  echo "launching ${args[*]}"
  nixGLIntel ./build/polyscope_viewer "${args[@]}" >/dev/null 2>&1
done
