#!/usr/bin/env bash
# stethos launcher — one command that always results in a working app window.
#
# stages, each verified before continuing:
#   1. sidecar: adopt a healthy instance on :8765, else start uvicorn with the
#      bundled models (the uvicorn path is the proven-good mode; the pyinstaller
#      bundle still has a weights-resolution bug)
#   2. app: launch the tauri binary under x11 (wayland kills webkitgtk), then
#   3. watch: if the app dies while this script lives, relaunch it.
#
# usage: run-stethos.sh [--no-watch]

set -u
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$APP_DIR/src-tauri/target/release/stethos"
PY="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
LOG_DIR="/tmp/opencode"
mkdir -p "$LOG_DIR"
SIDECAR_LOG="$LOG_DIR/stethos-sidecar.log"
APP_LOG="$LOG_DIR/stethos-app.log"
NO_WATCH="${1:-}"

log() { echo "[stethos] $*"; }
export RUST_LOG="${RUST_LOG:-info}"

# ---- stage 1: sidecar -------------------------------------------------------
probe() { curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:8765/health" 2>/dev/null; }

if [ "$(probe)" = "200" ]; then
    log "sidecar already healthy on :8765 — adopting"
else
    log "no healthy sidecar — starting uvicorn sidecar (models dir: $APP_DIR/python/models)"
    ( cd "$APP_DIR/python" && setsid nohup "$PY" -m uvicorn api.main:app --host 127.0.0.1 --port 8765 \
        >> "$SIDECAR_LOG" 2>&1 < /dev/null & )
    # wait for readiness: weights load on cuda can take a few seconds
    for i in $(seq 1 30); do
        [ "$(probe)" = "200" ] && break
        sleep 1
    done
    if [ "$(probe)" = "200" ]; then
        log "sidecar ready on :8765"
    else
        log "ERROR: sidecar did not become healthy — see $SIDECAR_LOG"
        exit 1
    fi
fi

# ---- stage 2: app -----------------------------------------------------------
if [ ! -x "$BIN" ]; then
    log "ERROR: app binary missing at $BIN (run: cd src-tauri && cargo build --release)"
    exit 1
fi

# --restart: kill any pre-existing instance first so there's exactly one window
if [ "${1:-}" = "--restart" ]; then
    for p in $(pgrep -x stethos); do
        [ "$p" != "$$" ] && kill "$p" 2>/dev/null
    done
    sleep 1
    log "existing instances stopped"
fi

launch_app() {
    GDK_BACKEND=x11 WEBKIT_DISABLE_DMABUF_RENDERER=1 setsid nohup "$BIN" \
        >> "$APP_LOG" 2>&1 < /dev/null &
    APP_PID=$!
    log "app launching (pid $APP_PID)"
}

launch_app

# ---- stage 3: watchdog ------------------------------------------------------
if [ "$NO_WATCH" = "--no-watch" ]; then
    sleep 5
    if kill -0 "$APP_PID" 2>/dev/null; then
        log "app alive after 5s — done"
        exit 0
    else
        log "ERROR: app died immediately — last log lines:"
        tail -5 "$APP_LOG"
        exit 1
    fi
fi

log "watching: app will be relaunched if it exits (ctrl-c to stop watching)"
while true; do
    sleep 10
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        log "app exited (status unknown) — relaunching in 3s"
        sleep 3
        launch_app
    fi
done
