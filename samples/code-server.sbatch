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
LOGIN_HOST="${IAG_LOGIN_HOST:-ssh.axisapps.io}"
NODE="$(hostname -f 2>/dev/null || hostname)"

TEAM="${IAG_TEAM:-}"
if [[ -z "$TEAM" ]]; then
    TEAM="$(id -Gn | tr ' ' '\n' | grep -E '^iag-team[0-9]+$' | head -n 1 || true)"
fi

if [[ -n "$TEAM" ]]; then
    TEAM_SCRATCH="${TEAM_SCRATCH:-/lustre/fs01/hackathons/teams/${TEAM}}"
    cd "$TEAM_SCRATCH"
fi

RUNTIME_DIR="${PWD}/.iag/${USER:-user}"
mkdir -p \
    "${RUNTIME_DIR}/code-server/config" \
    "${RUNTIME_DIR}/code-server/data" \
    "${RUNTIME_DIR}/code-server/extensions" \
    "${RUNTIME_DIR}/cache"
chmod -R g+rwX "${PWD}/.iag" 2>/dev/null || true

export XDG_CONFIG_HOME="${RUNTIME_DIR}/code-server/config"
export XDG_DATA_HOME="${RUNTIME_DIR}/code-server/data"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"

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

module load rootless-docker
if command -v start_rootless_docker >/dev/null 2>&1; then
    start_rootless_docker
else
    start_rootless_docker.sh
fi
docker version
echo

if ! command -v code-server >/dev/null 2>&1; then
    echo "code-server is not available in this environment." >&2
    echo "Ask the support team to install code-server as a module or put it on PATH." >&2
    exit 127
fi

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
