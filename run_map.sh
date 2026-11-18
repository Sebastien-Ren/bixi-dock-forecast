#!/bin/sh
# Serve the map on this laptop and open it. Stop with Ctrl+C.
cd "$(dirname "$0")/web"
PORT=${PORT:-8765}
( sleep 1; open "http://localhost:$PORT" 2>/dev/null || true ) &
exec python3 -m http.server "$PORT" --bind 127.0.0.1
