# Hackathon Cluster Start Here

This cluster uses Slurm to allocate GPU nodes. You do not need to learn Slurm in depth to get started. Use the helper commands in `bin/` for the common workflows.

## 1. Connect

From your laptop:

```bash
ssh ssh.axisapps.io -l <UNIQUE_AXIS_HASH>
```

Optional SSH config entry:

```sshconfig
Host iag-cluster
    HostName ssh.axisapps.io
    User <UNIQUE_AXIS_HASH>
```

Then connect with:

```bash
ssh iag-cluster
```

## 2. Add The Helper Commands

On the cluster login node, from this starter-kit directory:

```bash
export PATH="$PWD/bin:$PATH"
```

## 3. Know Your Team Storage

Each team has a Lustre directory:

```bash
/lustre/fs01/hackathons/teams/iag-team<N>
```

The helper scripts try to detect your team from your Linux groups. You can also set it explicitly:

```bash
export IAG_TEAM=iag-team7
export TEAM_SCRATCH=/lustre/fs01/hackathons/teams/$IAG_TEAM
cd "$TEAM_SCRATCH"
```

Put datasets, virtual environments, checkpoints, logs, and cloned repos under this directory.

## 4. Important Warning

Do not train models, run benchmarks, build large Docker images, or start Jupyter kernels on the login node.

Use the login node only for:

- SSH access
- editing files
- copying data
- submitting jobs
- checking job status

All GPU work should run inside a Slurm allocation.

## 5. Run A Healthcheck

Run this first:

```bash
iag-healthcheck
```

It checks:

- Slurm commands are available
- your team storage is visible
- a tiny 1 GPU job can start
- `nvidia-smi` works on the compute node
- rootless Docker can be loaded and started
- the `uv` module is available for virtual environment setup

If something fails, share the full output with the support team.

## 6. Start An Interactive GPU Shell

For 1 GPU:

```bash
iag-shell --gpus 1
```

For 2 or 4 GPUs:

```bash
iag-shell --gpus 2
iag-shell --gpus 4
```

Inside that shell, you are on a compute node and can run GPU work:

```bash
hostname
nvidia-smi
docker version
```

Leave the allocation with:

```bash
exit
```

## 7. Start JupyterLab On GPUs

Submit a Jupyter job:

```bash
iag-jupyter --gpus 1
```

or:

```bash
iag-jupyter --gpus 4 --time 08:00:00
```

The command prints the Slurm job id and tells you where the log file will appear. Once the job starts, the log shows the SSH tunnel command and the local browser URL.

`iag-jupyter` creates a virtual environment under your team storage and installs JupyterLab inside it. It also starts rootless Docker automatically.

The first launch can take a few minutes while the virtual environment is created. Later launches reuse the same environment.

If your login username is not your Axis hash, set it explicitly before starting Jupyter:

```bash
export UNIQUE_AXIS_HASH=<UNIQUE_AXIS_HASH>
iag-jupyter --gpus 1
```

After the job starts, you can also see the tunnel command with:

```bash
cat "$TEAM_SCRATCH/port_forwarding_command"
```

## 8. Submit A Batch Job

Use the sample scripts in `samples/`:

```bash
iag-submit samples/gpu-smoke-test.sbatch
iag-submit samples/python-venv-job.sbatch
iag-submit samples/docker-job.sbatch
```

Check your jobs:

```bash
iag-status
```

Cancel a job:

```bash
iag-cancel <jobid>
```

## 9. Use Rootless Docker On Compute Nodes

Docker is available only on compute nodes, inside a Slurm allocation. The provided helper commands and sample scripts start it automatically.

If you write your own `sbatch` file, include this block near the top:

```bash
module load rootless-docker
if command -v start_rootless_docker >/dev/null 2>&1; then
    start_rootless_docker
else
    start_rootless_docker.sh
fi
docker info
```

## 10. Copy Data

From your laptop to the cluster:

```bash
scp data.tar.gz <UNIQUE_AXIS_HASH>@ssh.axisapps.io:/lustre/fs01/hackathons/teams/iag-team<N>/
```

For larger folders, prefer `rsync`:

```bash
rsync -avP ./dataset/ <UNIQUE_AXIS_HASH>@ssh.axisapps.io:/lustre/fs01/hackathons/teams/iag-team<N>/dataset/
```

From the cluster back to your laptop:

```bash
rsync -avP <UNIQUE_AXIS_HASH>@ssh.axisapps.io:/lustre/fs01/hackathons/teams/iag-team<N>/outputs/ ./outputs/
```

## 11. Minimal Slurm Cheat Sheet

You only need these commands most of the time:

```bash
sinfo                         # cluster/node summary
squeue -u "$USER"             # your jobs
scancel <jobid>               # cancel a job
sbatch script.sbatch          # low-level submit for a batch job
srun --gres=gpu:1 --pty bash  # low-level interactive GPU shell
```

Prefer the `iag-*` helpers unless you already know what you are doing.
