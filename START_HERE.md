# Hackathon Cluster Start Here - Curiosity

Curiosity uses Slurm to allocate GPU nodes. You do not need to learn Slurm in depth to get started. Use the helper commands in `bin/` for the common workflows.

Curiosity is the A100 backup cluster for this hackathon. The important changes from the original cluster are:

- Team storage is visible in your home directory, for example `$HOME/iag-team7`.
- Docker is already provided by the compute-node Docker daemon.
- There is no `rootless-docker` module to load.

## 1. Connect

Use the Curiosity SSH command provided by the support team.

If you need an SSH config entry, use the host and username shared by the organizers:

```sshconfig
Host iag-curiosity
    HostName <CURIOSITY_LOGIN_HOST>
    User <UNIQUE_AXIS_HASH>
```

Then connect with:

```bash
ssh iag-curiosity
```

If the support team gives you a direct one-line command, use that instead.

## 2. Get The Curiosity Helper Commands

On the Curiosity login node:

```bash
git clone -b codex/curiosity-cluster https://github.com/boi-doingthings/Hackathon_Cluster.git
cd Hackathon_Cluster
export PATH="$PWD/bin:$PATH"
```

## 3. Set Your Team Storage

Each team has a shared storage symlink in the home directory:

```bash
$HOME/iag-team<N>
```

Set it once per shell. Replace `7` with your team number:

```bash
export IAG_TEAM=iag-team7
export TEAM_SCRATCH="$HOME/$IAG_TEAM"
cd "$TEAM_SCRATCH"
```

Check what the helper scripts resolve with:

```bash
iag-storage
```

Put datasets, checkpoints, notebooks, and cloned repos under this directory.

The helper scripts start in this team directory by default so teammates can collaborate in the same fast-storage workspace. Per-user runtime files such as virtual environments, logs, and tunnel commands are stored under:

```bash
$TEAM_SCRATCH/.iag/$USER/
```

This avoids conflicts between the five user accounts on the same team.

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
- your team storage is visible and writable
- a tiny 1 GPU job can start
- `nvidia-smi` works on the compute node
- Docker daemon access works on the compute node
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

For testing a specific team path:

```bash
iag-jupyter --gpus 1 --workspace "$HOME/iag-team<N>"
```

or:

```bash
iag-jupyter --gpus 4 --time 08:00:00
```

The command prints the Slurm job id and tells you where the log file will appear. Once the job starts, the log shows the SSH tunnel command and the local browser URL.

`iag-jupyter` starts in your team storage directory. It creates a per-user virtual environment under `$TEAM_SCRATCH/.iag/$USER/venvs/iag-jupyter` and installs JupyterLab inside it.

The first launch can take a few minutes while the virtual environment is created. Later launches reuse the same environment.

If your login username is not your Axis hash, set it explicitly before starting Jupyter:

```bash
export UNIQUE_AXIS_HASH=<UNIQUE_AXIS_HASH>
iag-jupyter --gpus 1
```

Set the Curiosity SSH gateway before starting Jupyter:

```bash
export IAG_LOGIN_HOST="<CURIOSITY_LOGIN_HOST>"
iag-jupyter --gpus 1
```

After the job starts, you can also see the tunnel command with:

```bash
cat "$TEAM_SCRATCH/.iag/$USER/port_forwarding_command.jupyter"
```

## 8. Start VS Code In The Browser Beta

This path is beta and less tested than JupyterLab.

Start a VS Code-like browser session on a GPU node:

```bash
iag-code --gpus 1
```

For testing a specific team path:

```bash
iag-code --gpus 1 --workspace "$HOME/iag-team<N>"
```

or:

```bash
iag-code --gpus 4 --time 08:00:00
```

The editor opens in your team storage directory. Docker is checked on the compute node.

If `code-server` is not already installed for your account, `iag-code` installs the standalone user version into:

```bash
$HOME/.local
```

The job also adds this to `PATH` for the session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

It installs the recommended Python and Jupyter extensions on first use.

After the job starts, see the tunnel command with:

```bash
cat "$TEAM_SCRATCH/.iag/$USER/port_forwarding_command.code"
```

If the user-level install fails, use JupyterLab and share the `code-*.out` log with the support team.

If port `8080` is already in use on your laptop, choose another local port:

```bash
iag-code --gpus 1 --port 8081
```

## 9. Submit A Batch Job

Use the sample scripts in `samples/`:

```bash
iag-submit samples/gpu-smoke-test.sbatch
iag-submit samples/python-venv-job.sbatch
iag-submit samples/docker-job.sbatch
iag-submit samples/code-server.sbatch
```

Check your jobs:

```bash
iag-status
```

Cancel a job:

```bash
iag-cancel <jobid>
```

## 10. Use Docker On Compute Nodes

Docker is available only on compute nodes, inside a Slurm allocation. On Curiosity, the compute node already has access to a Docker daemon.

If you write your own `sbatch` file, include this quick check near the top:

```bash
docker info >/dev/null
docker version
```

## 11. Copy Data

From your laptop to the cluster, copy into your team storage path. Replace the host and username with the Curiosity values shared by the support team:

```bash
scp data.tar.gz <UNIQUE_AXIS_HASH>@<CURIOSITY_LOGIN_HOST>:~/iag-team<N>/
```

For larger folders, prefer `rsync`:

```bash
rsync -avP ./dataset/ <UNIQUE_AXIS_HASH>@<CURIOSITY_LOGIN_HOST>:~/iag-team<N>/dataset/
```

From the cluster back to your laptop:

```bash
rsync -avP <UNIQUE_AXIS_HASH>@<CURIOSITY_LOGIN_HOST>:~/iag-team<N>/outputs/ ./outputs/
```

## 12. Minimal Slurm Cheat Sheet

You only need these commands most of the time:

```bash
sinfo                         # cluster/node summary
squeue -u "$USER"             # your jobs
scancel <jobid>               # cancel a job
sbatch script.sbatch          # low-level submit for a batch job
srun --gres=gpu:1 --pty bash  # low-level interactive GPU shell
```

Prefer the `iag-*` helpers unless you already know what you are doing.
