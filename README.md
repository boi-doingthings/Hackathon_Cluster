# Hackathon Cluster Starter Kit - Curiosity

Helper commands and examples for developer teams using the Curiosity Slurm cluster for the hackathon.

Curiosity is the A100 backup cluster. The participant workflow stays the same: SSH to the login node, work from the team shared storage symlink in `$HOME`, and use `iag-*` commands for GPU shells, JupyterLab, browser VS Code, health checks, and batch jobs.

## Quick Start

Clone this Curiosity branch on the cluster login node:

```bash
git clone -b codex/curiosity-cluster https://github.com/boi-doingthings/Hackathon_Cluster.git
cd Hackathon_Cluster
export PATH="$PWD/bin:$PATH"
```

Set your team storage path. Replace `7` with your team number:

```bash
export IAG_TEAM=iag-team7
export TEAM_SCRATCH="$HOME/$IAG_TEAM"
cd "$TEAM_SCRATCH"
```

Run the healthcheck:

```bash
iag-healthcheck
```

Start an interactive GPU shell:

```bash
iag-shell --gpus 1
```

Start JupyterLab on a GPU node:

```bash
iag-setup-jupyter
iag-jupyter --gpus 1
```

Start VS Code in the browser beta:

```bash
iag-code --gpus 1
```

Read the full participant guide:

[START_HERE.md](START_HERE.md)

## Storage Model

Each team has shared fast storage visible from the user's home directory through a symlink:

```bash
$HOME/iag-team<N>
```

Examples:

```bash
export IAG_TEAM=iag-team7
export TEAM_SCRATCH="$HOME/$IAG_TEAM"
```

JupyterLab, VS Code, and interactive shells require a writable team storage path. They will not silently fall back to `$HOME`.

Shared project files should live directly under `$TEAM_SCRATCH`.

Per-user runtime files live under:

```bash
$TEAM_SCRATCH/.$USER/
```

This keeps five teammates on the same team from colliding over virtual environments, logs, and tunnel command files.

## Included Commands

- `iag-healthcheck`: checks Slurm, team storage, GPU visibility, Docker daemon access, and Python virtualenv tooling.
- `iag-shell`: starts an interactive GPU shell and checks Docker.
- `iag-setup-jupyter`: prepares the per-user JupyterLab environment from the login node.
- `iag-jupyter`: starts JupyterLab on a GPU node from team storage.
- `iag-code`: beta; starts `code-server` on a GPU node from team storage and installs it into `~/.local` if missing.
- `iag-status`: shows cluster status and the current user's jobs.
- `iag-cancel`: cancels one or more Slurm jobs.
- `iag-submit`: submits a sample or custom `sbatch` file.
- `iag-storage`: shows the resolved team storage path and diagnostics.

## Samples

The `samples/` directory contains:

- `gpu-smoke-test.sbatch`
- `python-venv-job.sbatch`
- `docker-job.sbatch`
- `jupyter.sbatch`
- `code-server.sbatch`

Submit a sample with:

```bash
iag-submit samples/gpu-smoke-test.sbatch
```

## Notes

- The default partition is `primary`.
- Memory is not requested by default. Pass `--mem` only if a job needs a specific amount.
- Docker is available through the compute-node Docker daemon. No `rootless-docker` module is used on Curiosity.
- The default Curiosity uv module is `py-uv/0.6.8`. Override with `IAG_UV_MODULE` if the module name changes.
- `iag-shell` and `iag-setup-jupyter` place uv/pip caches under `$TEAM_SCRATCH/.$USER/cache` and use `UV_LINK_MODE=hardlink` to avoid slow cross-filesystem wheel copies.
- JupyterLab is installed into a per-user virtual environment on team storage. Run `iag-setup-jupyter` once if first-time package downloads are slow from a GPU job.
- `iag-code` is beta. If `code-server` is missing, it installs the standalone user version into `~/.local`.
- If local port `8001` or `8080` is already in use, pass `--port` to `iag-jupyter` or `iag-code`.
- Set `IAG_LOGIN_HOST` before starting JupyterLab or VS Code so the printed tunnel command uses the Curiosity SSH gateway.
