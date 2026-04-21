#!/usr/bin/env bash
# Build wrapper. The devenv profile doesn't pull in -dev outputs for X11 /
# wayland / libglvnd, so nanogui's glfw platform layer can't find its headers
# out of the box. This script collects the right /nix/store/*-dev/include
# directories onto CPATH before invoking cmake / make.
#
# Usage:
#   ./build.sh                 # configure if needed, build all
#   ./build.sh <target> ...    # build specific target(s), e.g. RepulsiveShells
set -e

EXTRA_INCLUDES=$(ls -d \
    /nix/store/*libx11*-dev/include \
    /nix/store/*libxrender*-dev/include \
    /nix/store/*libxfixes*-dev/include \
    /nix/store/*libxinput*-dev/include \
    /nix/store/*libxcursor*-dev/include \
    /nix/store/*libxinerama*-dev/include \
    /nix/store/*libxext*-dev/include \
    /nix/store/*libxi*-dev/include \
    /nix/store/*libxrandr*-dev/include \
    /nix/store/*libglvnd*-dev/include \
    /nix/store/*xorgproto*/include \
    /nix/store/*wayland*-dev/include \
    /nix/store/*libxkbcommon*-dev/include \
    2>/dev/null | tr '\n' ':' | sed 's/:$//')

export CPATH="${EXTRA_INCLUDES}${CPATH:+:$CPATH}"

if [ ! -f build/Makefile ] && [ ! -f build/build.ninja ]; then
    cmake -S . -B build -DRSH_ENABLE_VIEWER=ON
fi

if [ $# -eq 0 ]; then
    cmake --build build
else
    cmake --build build --target "$@"
fi
