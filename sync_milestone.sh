#!/usr/bin/env bash
# Sync milestone.html -> coherentmonkey.github.io/184/project/index.html
# (and the Phase 1.8 plot image). Keeps the project repo and the website
# repo byte-identical so publishing to GH Pages stays a one-command op.
#
# Usage:  ./sync_milestone.sh [path-to-coherentmonkey.github.io]
# Default website path: ~/coherentmonkey.github.io

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="${1:-$HOME/coherentmonkey.github.io}"
SITE_PROJECT_DIR="$SITE_DIR/184/project"

if [[ ! -d "$SITE_DIR" ]]; then
    echo "error: website checkout not found at $SITE_DIR" >&2
    echo "       pass the path as the first argument, e.g." >&2
    echo "       ./sync_milestone.sh /path/to/coherentmonkey.github.io" >&2
    exit 1
fi

mkdir -p "$SITE_PROJECT_DIR"

cp "$PROJECT_DIR/milestone.html" "$SITE_PROJECT_DIR/index.html"
echo "  synced milestone.html -> $SITE_PROJECT_DIR/index.html"

# The Phase 1.8 energy plot is generated into out/phase1_flow/ (gitignored in
# the project repo). Only the website needs the binary; copy it if present.
PLOT_SRC="$PROJECT_DIR/out/phase1_flow/energy.png"
if [[ -f "$PLOT_SRC" ]]; then
    cp "$PLOT_SRC" "$SITE_PROJECT_DIR/energy.png"
    echo "  synced energy.png       -> $SITE_PROJECT_DIR/energy.png"
else
    echo "  note: $PLOT_SRC missing; run demo_phase1_flow + plot_phase1_flow.py to regenerate" >&2
fi

echo "  done. Review and commit in both repos:"
echo "    cd $PROJECT_DIR && git add milestone.html sync_milestone.sh && git commit"
echo "    cd $SITE_DIR    && git add 184/project 184/index.md && git commit"
