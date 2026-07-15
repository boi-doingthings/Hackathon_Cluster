# Hackathon Cluster Starter Kit

Small helper scripts and examples for developer teams using a Slurm-based H100 cluster during the hackathon.

The goal is to keep participants focused on building: SSH in, work from fast team storage, start JupyterLab or a browser-based VS Code session on GPU nodes, and avoid running heavy work on the login node.

## Quick Start

Clone on the cluster login node:

```bash
git clone https://github.com/boi-doingthings/Hackathon_Cluster.git
cd Hackathon_Cluster
export PATH="$PWD/bin:$PATH"
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
iag-jupyter --gpus 1
```

Start VS Code in the browser beta:

```bash
iag-code --gpus 1
```

Read the full participant guide:

[START_HERE.md](START_HERE.md)

## Storage Model

Each team has fast Lustre storage:

```bash
/lustre/fs01/hackathons/teams/iag-team<N>
```

The helper scripts detect the team from Linux groups when possible. You can set it explicitly:

```bash
export IAG_TEAM=iag-team7
export TEAM_SCRATCH=/lustre/fs01/hackathons/teams/$IAG_TEAM
```

JupyterLab, VS Code, and interactive shells require a writable team storage path. They will not silently fall back to `$HOME`.

Shared project files should live directly under `$TEAM_SCRATCH`.

Per-user runtime files live under:

```bash
$TEAM_SCRATCH/.iag/$USER/
```

This keeps five teammates on the same team from colliding over virtual environments, logs, and tunnel command files.

## Included Commands

- `iag-healthcheck`: checks Slurm, team storage, GPU visibility, Docker startup, and `uv`.
- `iag-shell`: starts an interactive GPU shell and initializes rootless Docker.
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

- Memory is not requested by default. Pass `--mem` only if a job needs a specific amount.
- Docker is started automatically in the provided GPU allocation paths.
- JupyterLab is installed into a per-user virtual environment on fast storage.
- `iag-code` is beta. If `code-server` is missing, it installs the standalone user version into `~/.local`.
- If local port `8001` or `8080` is already in use, pass `--port` to `iag-jupyter` or `iag-code`.
