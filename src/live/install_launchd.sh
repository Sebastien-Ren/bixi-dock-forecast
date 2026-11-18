#!/bin/sh
# Install (or remove) the logger as a macOS login job that runs every 5 minutes.
#   sh src/live/install_launchd.sh            install and start
#   sh src/live/install_launchd.sh uninstall  stop and remove
# It only writes one file in ~/Library/LaunchAgents and logs to live/logger.log.
# The Mac must be awake: while it sleeps no polls happen, and the gaps show up in the polls table.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL=com.bixi-forecast.logger
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
if [ "$1" = "uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL"
  exit 0
fi
mkdir -p "$ROOT/live"
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$ROOT/.venv/bin/python</string><string>$ROOT/src/live/log_status.py</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>StandardOutPath</key><string>$ROOT/live/logger.log</string>
  <key>StandardErrorPath</key><string>$ROOT/live/logger.log</string>
</dict></plist>
PLIST_EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL. Check with: .venv/bin/python src/live/log_status.py --summary"
