#!/bin/bash -l
#SBATCH --job-name=iag-jupyter
#SBATCH --partition=primary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=jupyter-%j.out

set -euo pipefail
umask 0007

EXTERNAL_PORT="${EXTERNAL_PORT:-8001}"
LOGIN_HOST="${IAG_LOGIN_HOST:-CURIOSITY_LOGIN_HOST_NOT_SET}"
VENV_NAME="${IAG_JUPYTER_VENV_NAME:-iag-jupyter}"
UV_MODULE="${IAG_UV_MODULE:-py-uv/0.6.8}"
UV_HTTP_TIMEOUT="${IAG_UV_HTTP_TIMEOUT:-300}"
PIP_DEFAULT_TIMEOUT="${IAG_PIP_DEFAULT_TIMEOUT:-300}"
UV_LINK_MODE="${IAG_UV_LINK_MODE:-hardlink}"
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

RUNTIME_DIR="${PWD}/.${USER:-user}"
mkdir -p \
    "${RUNTIME_DIR}/venvs" \
    "${RUNTIME_DIR}/jupyter/config" \
    "${RUNTIME_DIR}/jupyter/data" \
    "${RUNTIME_DIR}/jupyter/runtime" \
    "${RUNTIME_DIR}/ipython" \
    "${RUNTIME_DIR}/cache/uv" \
    "${RUNTIME_DIR}/cache/pip" \
    "${RUNTIME_DIR}/tmp"
chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true

export JUPYTER_CONFIG_DIR="${RUNTIME_DIR}/jupyter/config"
export JUPYTER_DATA_DIR="${RUNTIME_DIR}/jupyter/data"
export JUPYTER_RUNTIME_DIR="${RUNTIME_DIR}/jupyter/runtime"
export IPYTHONDIR="${RUNTIME_DIR}/ipython"
export UV_CACHE_DIR="${RUNTIME_DIR}/cache/uv"
export PIP_CACHE_DIR="${RUNTIME_DIR}/cache/pip"
export TMPDIR="${RUNTIME_DIR}/tmp"
export UV_HTTP_TIMEOUT
export PIP_DEFAULT_TIMEOUT
export UV_LINK_MODE

while true; do
    PORT="$(shuf -n 1 -i 9000-9999)"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn | awk '{print $4}' | grep -q ":${PORT}$" || break
    elif command -v netstat >/dev/null 2>&1; then
        netstat -atun | grep -q ":${PORT}" || break
    else
        break
    fi
done

echo "Jupyter job started"
echo "Node: ${NODE}"
echo "Internal Jupyter port: ${PORT}"
echo "Local browser port: ${EXTERNAL_PORT}"
echo

nvidia-smi -L || true
echo

docker info >/dev/null
docker version
echo

if ! command -v uv >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "${UV_MODULE}" >/dev/null 2>&1 || module load uv >/dev/null 2>&1 || true
fi
VENV_DIR="${RUNTIME_DIR}/venvs/${VENV_NAME}"
if [[ ! -d "$VENV_DIR" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv "$VENV_DIR"
    else
        python3 -m venv "$VENV_DIR"
    fi
fi
source "${VENV_DIR}/bin/activate"
if ! command -v jupyter >/dev/null 2>&1; then
    install_status=0
    if command -v uv >/dev/null 2>&1; then
        echo "Using $(uv --version)"
        uv pip install jupyterlab || install_status=$?
    else
        echo "Using $(python --version) with pip"
        python -m ensurepip --upgrade >/dev/null 2>&1 || true
        python -m pip install jupyterlab || install_status=$?
    fi
    if [[ "$install_status" -ne 0 ]]; then
        echo "JupyterLab installation failed while downloading packages." >&2
        echo "Try preparing the environment first with: iag-setup-jupyter" >&2
        exit "$install_status"
    fi
fi
echo "Jupyter venv: ${VENV_DIR}"
echo

HASHNUMBER="${UNIQUE_AXIS_HASH:-${USER:-<UNIQUE_AXIS_HASH>}}"
TUNNEL_FILE="${RUNTIME_DIR}/port_forwarding_command.jupyter"
echo "From your laptop, run:"
echo "ssh -L localhost:${EXTERNAL_PORT}:${NODE}:${PORT} ${LOGIN_HOST} -l ${HASHNUMBER}"
echo
echo "Then open:"
echo "http://localhost:${EXTERNAL_PORT}"
echo
echo "ssh -L localhost:${EXTERNAL_PORT}:${NODE}:${PORT} ${LOGIN_HOST} -l ${HASHNUMBER}" > "${TUNNEL_FILE}"
echo "Port forwarding command was also written to: ${TUNNEL_FILE}"
echo

exec jupyter lab \
    --no-browser \
    --allow-root \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --ServerApp.token="" \
    --NotebookApp.token="" \
    --notebook-dir="${PWD}"
