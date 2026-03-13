#!/usr/bin/env bash
# Cron-friendly script: check remote VERSION (same as Python/PM2 autoupdate),
# and if newer, git pull + rebuild image + docker compose up.
# Usage: run from repo root, or set REPO_DIR. Optional: BRANCH=main, ENV_FILE=.env.validator.
# Example crontab (every 5 min): */5 * * * * /path/to/bitmind-subnet/docker/autoupdate.sh >> /var/log/bitmind-docker-update.log 2>&1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${BRANCH:-main}"
ENV_FILE="${ENV_FILE:-.env.validator}"
VERSION_URL="https://raw.githubusercontent.com/BitMind-AI/bitmind-subnet/${BRANCH}/VERSION"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

cd "$REPO_DIR"
LOCAL_VERSION_FILE="$REPO_DIR/VERSION"

if [[ ! -f "$LOCAL_VERSION_FILE" ]]; then
  log "ERROR: VERSION not found at $LOCAL_VERSION_FILE"
  exit 1
fi

local_version=$(cat "$LOCAL_VERSION_FILE" | tr -d '\n\r ')
remote_version=$(curl -sL --connect-timeout 10 --max-time 30 \
  -H "Cache-Control: no-cache, no-store, must-revalidate" \
  -H "Pragma: no-cache" \
  "$VERSION_URL?t=$(date +%s)" | tr -d '\n\r ')

if [[ -z "$remote_version" ]]; then
  log "WARN: Could not fetch remote VERSION from $VERSION_URL"
  exit 0
fi

# Compare x.y.z (same semantics as Python autoupdater: tuple comparison)
latest=$(printf '%s\n%s\n' "$local_version" "$remote_version" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
if [[ "$latest" != "$remote_version" ]] || [[ "$local_version" == "$remote_version" ]]; then
  log "No update: local=$local_version remote=$remote_version"
  exit 0
fi

log "New version: $remote_version (local=$local_version). Pulling and rebuilding..."
git pull

# Verify VERSION after pull
new_local=$(cat "$LOCAL_VERSION_FILE" | tr -d '\n\r ')
if [[ "$new_local" != "$remote_version" ]]; then
  log "ERROR: After git pull, local VERSION=$new_local != remote=$remote_version"
  exit 1
fi

log "Tearing down current stack (including orphans)..."
docker compose --env-file "$ENV_FILE" down --remove-orphans

log "Rebuilding image and bringing stack up..."
# Use --no-cache to ensure a clean build; omit for faster incremental builds.
docker compose --env-file "$ENV_FILE" build --no-cache
docker compose --env-file "$ENV_FILE" up -d

log "Update to $remote_version complete."
exit 0
