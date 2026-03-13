#!/bin/bash
# Persist ~/.claude.json across devcontainer rebuilds.
# The named volume at ~/.claude/ survives rebuilds, but ~/.claude.json
# (which lives OUTSIDE that directory) does not. This script backs it
# up into the volume and restores it on container start.

BACKUP="$HOME/.claude/.claude.json.bak"
CONFIG="$HOME/.claude.json"

case "${1:-}" in
  restore)
    if [ -f "$BACKUP" ]; then
      cp "$BACKUP" "$CONFIG"
      echo "claude-persist: restored $CONFIG from volume backup"
    else
      # First run — create minimal config so Claude doesn't show fresh-install flow
      cat > "$CONFIG" <<'EOF'
{"hasCompletedOnboarding":true,"installMethod":"devcontainer","autoUpdates":false}
EOF
      echo "claude-persist: created minimal $CONFIG (first run)"
    fi
    ;;
  save)
    if [ -f "$CONFIG" ]; then
      cp "$CONFIG" "$BACKUP"
      echo "claude-persist: saved $CONFIG to volume backup"
    fi
    ;;
  *)
    echo "Usage: $0 {restore|save}" >&2
    exit 1
    ;;
esac
