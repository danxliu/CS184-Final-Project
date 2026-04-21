#!/usr/bin/env bash
# Convenience wrapper for running the viewer with the right GL stack.
# Usage:
#   ./run.sh assets/icosphere_3.obj     # single mesh
#   ./run.sh out/phase1_flow            # frame_XXXX.obj sequence (play/pause UI)
#
set -e

exec nixGLIntel ./build/RepulsiveShells "$@"
