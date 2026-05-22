# AML v2 job submission & debugging

Self-contained YAML alternatives to the SDK `command(...)` cells in the notebook.
Use these when you want a tight edit → submit → fail-fast loop without re-running
notebook cells.

## Files

| File | Purpose |
|---|---|
| `env.yml`                 | Training environment (CUDA 12.4 / cuDNN 9, pins from `../conda.yml`). |
| `data.yml`                | Registers the training CSV as a versioned data asset. |
| `job.smoke.yml`           | **1-GPU smoke job.** Runs on a compute instance. Few minutes per run. |
| `job.cluster.single.yml`  | 1× T4 production job on `Standard_NC16as_T4_v3`. Fits a 20-core quota. |
| `job.cluster.yml`         | 4× T4 DDP production job on `Standard_NC64as_T4_v3`. Needs ≥ 64-core cluster quota. |

## Prerequisites (one-off)

```bash
# 1. CLI extension + workspace defaults
az extension add -n ml -y
az configure --defaults group=<rg> workspace=<workspace>

# 2. Register the environment (only when conda.yml changes)
az ml environment create -f aml/env.yml

# 3. Register / bump the data asset (only when the CSV changes)
az ml data create -f aml/data.yml --set version=$(date +%Y%m%d%H%M)
```

## Debug loop — run on your compute instance (no cluster quota needed)

`ComputeInstance` is a valid v2 Command-Job target. It runs one job at a time,
no multi-node, no extra quota — ideal for debugging.

```bash
# Replace <ci-name> with the compute instance you SSH into today.
az ml job create -f aml/job.smoke.yml \
    --set compute=azureml:<ci-name> \
    --web --stream
```

`--web` opens AML Studio in your browser; `--stream` tails `std_log.txt` in the
terminal. Detach with `Ctrl+C` — the job keeps running. Re-attach with:

```bash
az ml job stream -n <job-name>
```

### Fast iteration tips

- **Edit + resubmit only.** `code: ../` re-uploads the working dir as a snapshot
  per submission, so you do not need to bump environment or data versions for
  code changes.
- **Cache HF weights.** `HF_HOME=/tmp/hf` keeps downloads inside the job
  container; the curated image already ships transformers but not the model
  weights. Move to a mounted output if you want them to survive across runs.
- **Drop to a shell.** If a job fails during environment build, use
  `az ml job connect-ssh -n <job-name>` (CI-attached jobs support interactive
  debug via the Studio "Open Terminal" button).
- **Tail individual log files**:
  ```bash
  az ml job download -n <job-name> --download-path ./_job_logs
  ls _job_logs/user_logs/      # std_log.txt, std_log_process_0.txt, ...
  ```

## Production run — once cluster quota lands

```bash
az ml job create -f aml/job.cluster.yml --web --stream
```

If `gpu-t4-cluster` does not exist yet:

```bash
az ml compute create \
  --name gpu-t4-cluster \
  --type AmlCompute \
  --size Standard_NC64as_T4_v3 \
  --min-instances 0 --max-instances 1 \
  --idle-time-before-scale-down 300
```

### Quota error on cluster creation but CI works?

Two things to check, in order:

1. **Per-SKU core count vs. available quota.** A `Standard_NC64as_T4_v3` node
   needs 64 cores; if the customer only has e.g. 20 cores of `Standard NCASv3_T4
   Family Cluster Dedicated vCPUs`, the cluster *cannot* be created regardless
   of how many nodes you ask for. Either request more quota, or drop to
   `Standard_NC16as_T4_v3` (16 cores, 1× T4) — see `job.cluster.single.yml`.
2. **CI quota ≠ Cluster quota.** Compute Instance and AmlCompute draw on
   **different** quota buckets:

| Quota name (Portal → Subscriptions → Usage + quotas) | Used by |
|---|---|
| `Standard NCASv3_T4 Family vCPUs`                    | Compute Instance |
| `Standard NCASv3_T4 Family Cluster Dedicated vCPUs`  | AmlCompute cluster (dedicated) |
| `Standard NCASv3_T4 Family Cluster LowPriority vCPUs`| AmlCompute cluster (low-pri / spot) |

Two ways forward:

1. **Request the cluster-dedicated quota.** Same form, different line item.
   Typical SLA: same business day. Ask for ≥ 64 cores for `Standard_NC64as_T4_v3`.
2. **Use low-priority for now.** Often already at 64 cores by default:
   ```bash
   az ml compute create --name gpu-t4-cluster-lp \
       --type AmlCompute --size Standard_NC64as_T4_v3 \
       --min-instances 0 --max-instances 1 \
       --tier low_priority \
       --idle-time-before-scale-down 300
   ```
   Then submit with `--set compute=azureml:gpu-t4-cluster-lp`. Low-pri nodes can
   be pre-empted; HF `Trainer` checkpoints under `--output-dir` survive retries.

## Common failure modes

| Symptom in `std_log.txt` | Likely cause | Fix |
|---|---|---|
| `RuntimeError: NCCL ... unhandled cuda error` early in epoch 1 | DDP rank mismatch between `--nproc_per_node` and `process_count_per_instance` | Keep them equal in `job.cluster.yml`. |
| Job stuck in `Preparing` for >10 min | Image build cold start | Normal first run. Subsequent runs reuse the cached image. |
| `Insufficient quota` on submit | Requesting dedicated cores you don't have | See quota table above; switch to low-priority. |
| `FileNotFoundError: dataset_onlyWithImoUntil2023.csv` inside the job | Forgot to register data asset, or `mode: ro_mount` not propagating | `az ml data list -o table` to confirm asset, re-bump version. |
| `Trainer` crashes with `mlflow ... no active run` on a non-zero rank | Logging on non-main rank | Already guarded in `train.py` via `is_main_process`; check no new `mlflow.log_*` calls slipped in. |
