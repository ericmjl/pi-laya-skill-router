#!/usr/bin/env bash
# Install local services for the laya skill-router pi plugin.
#
# The pi extension itself is wired by `pi install git:github.com/ericmjl/pi-laya-skill-router`
# (see package.json). This script sets up everything on THIS machine that a
# plugin install can't do for you:
#   1. uv sync                 (python deps for the sidecar)
#   2. skills.json             (scans your installed skills into the router catalog)
#   3. launchd: sidecar        (serves the router on 127.0.0.1:8787, KeepAlive)
#   4. launchd: nightly loop   (optional, --with-nightly: auto distill+finetune at 3 AM)
#
# Plists in this repo are templates (__HOME__/__UV__); this script substitutes
# real paths and writes them to ~/Library/LaunchAgents.
#
# Usage:
#   ./install.sh              # sidecar only
#   ./install.sh --with-nightly
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
WITH_NIGHTLY=false
[ "${1:-}" = "--with-nightly" ] && WITH_NIGHTLY=true

command -v uv >/dev/null || { echo "error: uv not found — install from https://docs.astral.sh/uv/"; exit 1; }
UV="$(command -v uv)"

echo "==> uv sync"
(cd "$REPO" && "$UV" sync)

echo "==> scanning skills into skills.json"
(cd "$REPO" && "$UV" run python scripts/scan_skills.py)

install_plist() {
  local src="$REPO/$1" name; name="$(basename "$1")"
  mkdir -p "$LAUNCH_AGENTS"
  sed -e "s|__UV__|$UV|g" -e "s|__HOME__|$HOME|g" "$src" > "$LAUNCH_AGENTS/$name"
  plutil -lint "$LAUNCH_AGENTS/$name"
  launchctl unload "$LAUNCH_AGENTS/$name" 2>/dev/null || true
  launchctl load "$LAUNCH_AGENTS/$name"
  echo "    installed $name"
}

echo "==> installing sidecar launchd job"
install_plist launchd/com.ericmjl.laya-sidecar.plist

if $WITH_NIGHTLY; then
  echo "==> installing nightly auto-learning launchd job"
  install_plist launchd/com.ericmjl.laya-nightly.plist
fi

echo "==> waiting for sidecar health"
for _ in $(seq 1 20); do
  if curl -s --max-time 3 http://127.0.0.1:8787/health | grep -q '"loaded":true'; then
    curl -s http://127.0.0.1:8787/health; echo
    echo "done. now: pi install git:github.com/ericmjl/pi-laya-skill-router"
    exit 0
  fi
  sleep 3
done
echo "error: sidecar did not become healthy — check /tmp/laya-sidecar.log" >&2
exit 1
