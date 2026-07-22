#!/bin/bash -l
#SBATCH --job-name=iag-code
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=code-%j.out

set -euo pipefail
umask 0007

EXTERNAL_PORT="${EXTERNAL_PORT:-8080}"
LOGIN_HOST="${IAG_LOGIN_HOST:-CURIOSITY_LOGIN_HOST_NOT_SET}"
NODE="$(hostname -f 2>/dev/null || hostname)"

TEAM="${IAG_TEAM:-}"
if [[ -z "$TEAM" ]]; then
    TEAM="$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)"
fi

if [[ -z "${TEAM_SCRATCH:-}" && -n "$TEAM" ]]; then
    TEAM_SCRATCH="${HOME}/${TEAM}"
fi

if [[ -z "${TEAM_SCRATCH:-}" || ! -d "$TEAM_SCRATCH" || ! -w "$TEAM_SCRATCH" ]]; then
    echo "Could not detect writable team storage." >&2
    echo "Set TEAM_SCRATCH=\$HOME/iag-team<N> before submitting." >&2
    exit 1
fi

cd "$TEAM_SCRATCH"

RUNTIME_DIR="${PWD}/.iag/${USER:-user}"
mkdir -p \
    "${RUNTIME_DIR}/code-server/config" \
    "${RUNTIME_DIR}/code-server/data" \
    "${RUNTIME_DIR}/code-server/extensions" \
    "${RUNTIME_DIR}/cache"
if [[ -n "$TEAM" ]]; then
    chgrp "$TEAM" "${PWD}/.iag" "${RUNTIME_DIR}" 2>/dev/null || true
fi
chmod -R g+rwX "${PWD}/.iag" 2>/dev/null || true
chmod 2770 "${PWD}/.iag" "${RUNTIME_DIR}" 2>/dev/null || true

export XDG_CONFIG_HOME="${RUNTIME_DIR}/code-server/config"
export XDG_DATA_HOME="${RUNTIME_DIR}/code-server/data"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export PATH="${HOME}/.local/bin:${PATH}"

while true; do
    PORT="$(shuf -n 1 -i 10000-10999)"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn | awk '{print $4}' | grep -q ":${PORT}$" || break
    elif command -v netstat >/dev/null 2>&1; then
        netstat -atun | grep -q ":${PORT}" || break
    else
        break
    fi
done

echo "VS Code job started"
echo "Node: ${NODE}"
echo "Internal code-server port: ${PORT}"
echo "Local browser port: ${EXTERNAL_PORT}"
echo "Workspace: ${PWD}"
echo

nvidia-smi -L || true
echo

docker info >/dev/null
docker version
echo

if ! command -v code-server >/dev/null 2>&1; then
    echo "code-server was not found on PATH."
    echo "Installing standalone code-server into ~/.local for ${USER:-this user}..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install code-server, but curl was not found." >&2
        exit 127
    fi
    curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v code-server >/dev/null 2>&1; then
    echo "code-server install completed but code-server is still not on PATH." >&2
    echo "Expected it at: ${HOME}/.local/bin/code-server" >&2
    exit 127
fi

echo "code-server: $(command -v code-server)"
code-server --version
echo
echo "Installing recommended VS Code extensions if needed..."
code-server --install-extension ms-python.python || echo "warning: failed to install ms-python.python"
code-server --install-extension ms-toolsai.jupyter || echo "warning: failed to install ms-toolsai.jupyter"
echo

HASHNUMBER="${UNIQUE_AXIS_HASH:-${USER:-<UNIQUE_AXIS_HASH>}}"
TUNNEL_FILE="${RUNTIME_DIR}/port_forwarding_command.code"
echo "From your laptop, run:"
echo "ssh -L localhost:${EXTERNAL_PORT}:${NODE}:${PORT} ${LOGIN_HOST} -l ${HASHNUMBER}"
echo
echo "Then open:"
echo "http://localhost:${EXTERNAL_PORT}"
echo
echo "ssh -L localhost:${EXTERNAL_PORT}:${NODE}:${PORT} ${LOGIN_HOST} -l ${HASHNUMBER}" > "${TUNNEL_FILE}"
echo "Port forwarding command was also written to: ${TUNNEL_FILE}"
echo

exec code-server \
    --bind-addr "0.0.0.0:${PORT}" \
    --auth none \
    --disable-telemetry \
    --user-data-dir "${RUNTIME_DIR}/code-server/data" \
    --extensions-dir "${RUNTIME_DIR}/code-server/extensions" \
    "${PWD}"
