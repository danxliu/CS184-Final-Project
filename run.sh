#!/usr/bin/env bash
# Convenience wrapper for running the viewer with the right GL stack.
# Usage:
#   ./run.sh assets/icosphere_3.obj     # single mesh
#   ./run.sh out/phase1_flow            # frame_XXXX.obj sequence (play/pause UI)
#
# The devenv profile doesn't expose libxcb at runtime (glfw dlopens it for the
# X11 backend), so we grab it from the nix store and prepend to LD_LIBRARY_PATH.
set -e

EXTRA_LIBS=$(ls -d /nix/store/*libxcb*[^-]/lib 2>/dev/null | head -1)
export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$HOME/.nix-profile/bin/nixGLIntel" ./build/RepulsiveShells "$@"
