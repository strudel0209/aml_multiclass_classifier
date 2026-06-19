# Azure Machine Learning - Email to IMO Classifier

## Corpus language profile

Before picking encoder candidates we ran a `langdetect` diagnostic over a 1000-row random sample (seed 42) of `dataset_onlyWithImoUntil2023.csv` (114 181 total rows). Bracket tokens (`[SUBJECT]` / `[FROM]` / `[ATTACH]` / `[BODY]`), email addresses and URLs were stripped before detection to avoid ASCII bias.

| Language | Count | Share |
|---|---:|---:|
| en (English)    | 925 | 92.50 % |
| no (Norwegian)  |  38 |  3.80 % |
| fi (Finnish)    |  12 |  1.20 % |
| fr (French)     |  10 |  1.00 % |
| it (Italian)    |   5 |  0.50 % |
| da (Danish)     |   4 |  0.40 % |
| et, sv, pt, nl, es, ru (1 each) | 6 | 0.60 % |
| **Non-English total** | **75** | **7.50 %** |

Sample texts confirmed the non-English rows are legitimate operational content (Norwegian alarm message, Finnish ABB technician note, French quote request, Italian printer issue, Russian Norilsk Nickel inquiry) rather than detector noise.

**Decision (borderline 5–15 % zone):** an English-only encoder is probably acceptable, but the dataset has enough multilingual tail to justify comparing one multilingual candidate. The bake-off in the AML notebook ("Architecture bake-off" section) trains four candidates on identical splits and aggregates results in a single MLflow comparison table:

| Candidate | HF model ID | Role | Notes |
|---|---|---|---|
| ModernBERT-base   | `answerdotai/ModernBERT-base`   | Baseline           | Current production model (149 M, 8192 ctx) |
| ModernBERT-large  | `answerdotai/ModernBERT-large`  | Scale comparison   | ~395 M params; tests whether capacity helps |
| DeBERTa-v3-base   | `microsoft/deberta-v3-base`     | Short-ctx control  | 184 M, 512 ctx — strong even with truncation |
| EuroBERT-210M     | `EuroBERT/EuroBERT-210M`        | Multilingual probe | Covers the 7.5 % non-English tail |

On the default single-GPU `gpu-t4-single` cluster the four jobs run **serially** (`max_instances=1`), so wall-clock ≈ 4× a single run. Estimated bake-off cost: ~25–40 USD. Raise the cluster `max_instances` (and request more T4 quota) to run them in parallel at the same total cost.

---

## Prerequisites

Everything that must be provisioned **before** opening the notebook. Roughly 30 minutes of one-off setup if the Azure subscription is new; near-zero if the workspace already exists.

### A. Azure subscription & identity

| Requirement | Purpose | How to check |
|---|---|---|
| Active Azure subscription | Hosts the AML workspace and GPU quotas | `az account show` |
| Owner or Contributor on the target resource group | Create workspace + storage + ACR + endpoint | `az role assignment list --assignee <upn> -g <rg>` |
| Subscription registered for `Microsoft.MachineLearningServices` | Required to create AML resources | `az provider show -n Microsoft.MachineLearningServices --query registrationState` |

### B. Azure ML workspace

A single AML workspace in a region where T4 quota is available (e.g. `westeurope`, `eastus`, `eastus2`, `northeurope`). The workspace must have the four standard companion resources attached — they are auto-created by `az ml workspace create` if missing:

| Companion resource | Used for |
|---|---|
| Storage Account (blob)                | Dataset upload + job code snapshot + MLflow artifacts |
| Key Vault                             | Endpoint scoring auth keys |
| Application Insights                  | Job + endpoint telemetry |
| Azure Container Registry (ACR), Premium SKU recommended | Stores the built training/inference Docker images |

One-shot creation (skip if already present):

```bash
az group create -n <rg> -l <region>
az ml workspace create -n <workspace> -g <rg> -l <region>
```

### C. GPU quota

The notebook now defaults to a **single-GPU** training cluster, so you only need a modest T4 quota. Each SKU below must show a non-zero `Limit` in your target region. Request quota via **Azure Portal → Subscriptions → Usage + quotas → Compute** (typical SLA: same business day):

| SKU | Used for | vCPU quota family | Min cores needed | Required? |
|---|---|---|---:|---|
| `Standard_NC16as_T4_v3` | **Default** training cluster `gpu-t4-single` (1× T4) **and** inference endpoint (1× T4) | `Standard NCASv3_T4 Family Cluster Dedicated vCPUs` | 16–20 | ✅ required |
| `Standard_NC64as_T4_v3` | Optional 4× T4 DDP training upgrade | `Standard NCASv3_T4 Family Cluster Dedicated vCPUs` | 64 | ⬜ optional |

> **You can skip cluster quota entirely** by setting `AML_TRAIN_COMPUTE` in `.env` to the name of a GPU **compute instance** you already own (see Step 3 below). In that case the job runs on your existing compute and no cluster is created.

A quick check from the CLI:

```bash
az vm list-usage -l <region> --query "[?contains(name.value,'standardNCASv3T4Family')]"
```

### D. Compute instance (for running the notebook)

The orchestration notebook runs on an AML **compute instance** — not on the GPU cluster. The compute instance does **not** need a GPU:

| Item | Recommendation |
|---|---|
| SKU                  | `Standard_DS3_v2` (4 vCPU / 14 GB) or larger CPU SKU |
| Image                | Default AML Python 3.10 image (ships `azure-ai-ml`, `azureml-mlflow`, `mlflow`, `ipykernel` pre-installed) |
| Managed identity     | System-assigned, with **AzureML Data Scientist** role on the workspace |
| Network              | Public endpoint OK for first run; behind workspace-managed VNet is also supported |

Create from the AML Studio UI ("Compute → Compute instances → + New") or via CLI:

```bash
az ml compute create -n <ci-name> --type ComputeInstance --size Standard_DS3_v2
```

### E. Python packages

Three dependency surfaces, each pinned independently because they target different runtimes:

| File | Where it runs | What it installs |
|---|---|---|
| **First cell of the notebook** (`%pip install ...`) | Compute instance kernel | Same set as `requirements.txt` — enough to execute every notebook cell end-to-end |
| [`requirements.txt`](requirements.txt) | Compute instance (fallback / local dev) | `torch>=2.10`, `transformers>=4.46` (5.x OK), `accelerate>=1.13`, `datasets>=4.7`, `scikit-learn>=1.7`, `mlflow>=3.10.1`, `azure-ai-ml`, `azure-identity`, `azureml-mlflow`, `safetensors`, `sentencepiece`, `protobuf>=4.25`, `numpy<2`, `ipykernel` |
| [`conda.yml`](conda.yml) | AML training/inference Docker image (built once, reused by every job) | Same packages on `python=3.10`, layered on the base image `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest`. The CUDA runtime comes from the self-contained `torch>=2.10` pip wheel, so the base image's own CUDA version is irrelevant. |
| [`requirements.inference.txt`](requirements.inference.txt) | Online endpoint container | Inference-only subset — no `azure-ai-ml`, no `datasets` |

The notebook's first cell is idempotent (skips packages that are already at the right version), so re-running it after a kernel restart is safe.

### F. Optional but recommended for the bake-off

| Item | Why |
|---|---|
| Cluster `max_instances=4` instead of `1` (in the AML Command Job cell) | Lets the four bake-off candidates train in parallel; same total cost, ~4× faster wall-clock |
| `langdetect` in the kernel | Only needed if you want to re-run the corpus-language diagnostic; not required for training or deployment |

### G. Files shipped separately from git

These are **not** in the repo but must exist in the working directory before submitting the AML Command Job:

- `dataset_onlyWithImoUntil2023.csv` — training set (`;`-separated)
- `dataset_onlyWithImoFrom2023.csv` — temporal holdout (`;`-separated), evaluation only

Upload via AML Studio's file browser, `azcopy`, or drag-and-drop into the compute instance.

---

## Step-by-Step Run Guide

This is the complete, beginner-friendly walkthrough — from a freshly cloned repo to a deployed, tested endpoint. It assumes **no prior knowledge of this codebase**. Every cloud action happens from the notebook [`imo_extractor_pipeline_CSV.ipynb`](imo_extractor_pipeline_CSV.ipynb); you only touch the terminal for one-time setup.

> **The golden rule:** training does **not** run inside the notebook kernel. The notebook *orchestrates* — it submits a job to an Azure ML GPU compute and streams the logs back. So your notebook can run on a cheap CPU compute instance while a T4 GPU does the heavy lifting in the cloud.

```
You (notebook on CPU)  ──submit──▶  AML Command Job  ──runs train.py──▶  T4 GPU
        ▲                                                                    │
        └──────────────── streams logs + auto-registers model ◀─────────────┘
```

### Step 0 — Where to run the notebook

Open the notebook on an **AML compute instance** (recommended) or your local machine. A GPU is **not** required here — the orchestration kernel runs fine on CPU. You need:

- The repo cloned and the two dataset CSVs present (Step 1).
- Logged in to Azure: run `az login` in the terminal (you can see this was the last command in your terminal — that's exactly right).

### Step 1 — Get the code and data in place

```bash
git clone <repo-url>
cd "CM6 AML Multi-class Classifier"
```

Two CSV files ship **separately from git**. Place both in the repo root, next to `train.py`:

| File | Role |
|---|---|
| `dataset_onlyWithImoUntil2023.csv` | Training set (pre-2023, `;`-separated) — **required** |
| `dataset_onlyWithImoFrom2023.csv`  | Temporal holdout (post-2023) — evaluation only, optional |

Drag-and-drop them via the AML Studio file browser, or `azcopy`/`scp` them onto the compute instance. The notebook's *file preflight* cell will fail fast and tell you exactly what is missing if either is absent.

### Step 2 — Fill in the `.env` file correctly  ⭐

This is the single most important configuration step. The notebook reads your Azure ML workspace coordinates from a `.env` file in the repo root. Create it (the *environment-variables* cell also writes a blank template if it's missing) and fill in **all four** lines:

```ini
# .env  — sits in the repo root, never commit real values to git
AZUREML_SUBSCRIPTION_ID=<your-subscription-guid>
AZUREML_RESOURCE_GROUP=<your-resource-group>
AZUREML_WORKSPACE=<your-workspace-name>
AML_TRAIN_COMPUTE=<your-gpu-compute-name>      # optional — see below
```

How to find each value:

| Variable | What it is | How to get it |
|---|---|---|
| `AZUREML_SUBSCRIPTION_ID` | The GUID of the Azure subscription holding your AML workspace | `az account show --query id -o tsv`, or Azure Portal → Subscriptions |
| `AZUREML_RESOURCE_GROUP` | The resource group containing the workspace | AML Studio → top-right workspace menu, or `az ml workspace list -o table` |
| `AZUREML_WORKSPACE` | The Azure ML workspace name | AML Studio header, or `az ml workspace list -o table` |
| `AML_TRAIN_COMPUTE` | **(optional)** Name of an *existing* GPU compute to train on. **Leave blank** to let the notebook auto-create a `gpu-t4-single` cluster. **Set it** to a GPU compute instance/cluster name to skip cluster-quota provisioning entirely. | AML Studio → Compute → your GPU instance's name |

> **Which `AML_TRAIN_COMPUTE` choice is right for you?**
> - **Leave it blank** if you have ≥ 16–20 cores of `Standard NCASv3_T4 Family Cluster Dedicated vCPUs` quota. The notebook auto-provisions a 1× T4 cluster that scales to zero when idle (you only pay while a job runs).
> - **Set it to an existing GPU compute** (e.g. `CM6-Training-GPU`) if you already have one, or if you have **no cluster quota**. This bypasses cluster creation completely.

Security note: `.env` should be in `.gitignore` — never commit your real subscription ID. The values are masked when printed by the notebook.

### Step 3 — (Optional) Run the offline unit tests

A ~3-second sanity check that exercises the regex pre-filter, IMO checksum, request parsing and label encoding — **no Azure, no GPU, no cost**:

```bash
python3 -m pip install pytest
python3 -m pytest
```

Expected: `64 passed`. If these fail, fix that before spending money on cloud compute.

### Step 4 — Run the setup & preprocessing cells

Open the notebook and run from the top through tokenization. These run **in the notebook kernel only** — zero cloud cost:

1. **Install dependencies** — picks a hardware-matched PyTorch wheel, installs the rest from `requirements.txt`. Re-running is a no-op if torch already works.
2. **Resolve environment variables** — loads your `.env`, validates the three required keys, prints them masked. If anything is missing it raises here with a clear message — fix `.env` and re-run.
3. **File & data preflight** — confirms `train.py`, `score.py`, `conda.yml`, both requirements files and the training CSV are present.
4. **Imports & version check / GPU sanity check** — prints library versions. *"No CUDA GPU"* here is **expected and fine** on a CPU compute instance.
5. **Bind the Azure ML workspace** — creates the authenticated `MLClient`. Success looks like `✓ MLClient bound to workspace '...'`.
6. **Load & Preprocess Data** — merges email fields with `[SUBJECT]`/`[FROM]`/`[ATTACH]`/`[BODY]` tokens, collapses rare vessels (< 20 emails) into `UNKNOWN`, and does a stratified train/val split. Note the printed `num_labels` — that is how many vessel classes the model will learn.
7. **Regex pre-filter** — reports the high-confidence fast-path coverage and its precision on its own hits (target ≥ 99 %).
8. **Tokenization** — tokenizes train/val with ModernBERT's tokenizer.

> **Do NOT run** the in-kernel *"Fine-tune transformer model"* (`trainer.train()`) cell unless you are on a GPU compute instance and just want a quick dev iteration. It is guarded to refuse to run on CPU. Full training happens in Step 5.

### Step 5 — Submit the training job to the AML GPU  ⭐

Run the **"AML Command Job: Full-Scale GPU Training"** cell. It does four things:

1. **Resolves the compute** — uses `AML_TRAIN_COMPUTE` if you set it, otherwise auto-creates the `gpu-t4-single` cluster (`Standard_NC16as_T4_v3`, 1× T4, scales to zero).
2. **Registers the training CSV** as an immutable versioned data asset (`email-imo-training-data:1`).
3. **Builds the training environment** from `conda.yml` on the `openmpi4.1.0-cuda11.8` base image (the CUDA runtime itself ships inside the `torch>=2.10` wheel).
4. **Submits the job** — launches `torchrun --standalone --nproc_per_node=1 train.py ... --epochs 15 --batch-size 8 --lr 3e-5 --max-seq-len 2048 --min-examples 20`, then streams the logs into the notebook.

You'll see `✓ Job submitted: <name>` and a **Studio URL** — click it to follow along in the browser.

### Step 6 — Verify the job is really running on the GPU

Confirm it's on a GPU and healthy, not silently stuck or on CPU:

1. **In AML Studio** (the printed Studio URL) → **Jobs → imo-extractor-experiment → modernbert-imo-1gpu**.
2. **Status** should move `Queued → Preparing → Running`. *Preparing* can take several minutes the first time while the Docker image builds — this is normal.
3. Open the **`Outputs + logs` tab → `user_logs/std_log.txt`**. A GPU run prints lines like:
   - `Profile: T4/Turing (Tesla T4)` (or `A100/Ampere` on a bigger GPU)
   - per-epoch `{'eval_f1': ..., 'eval_accuracy': ...}` blocks appearing every few minutes.
4. **Monitoring tab → Metrics** shows `eval_f1` / `eval_accuracy` curves climbing epoch over epoch.
5. *(Optional)* On the compute → **Monitoring**, GPU utilisation should be non-zero while *Running*.

> **Red flags:** `Profile: CPU (no GPU detected)` in the log means the job landed on a non-GPU target — check that `AML_TRAIN_COMPUTE` points at a GPU compute (or is blank so the T4 cluster is used). A job stuck in *Preparing* for >20 min usually means an environment/image build error — open `system_logs/` to see the build failure.

The cell streams logs to the notebook; press `Ctrl+C` to detach — **the job keeps running** in the cloud. Re-attach any time with the Studio URL.

### Step 7 — Interpret the metrics

When the job finishes, the tail of `std_log.txt` (and the **Metrics** tab) shows the final validation scores. `train.py` logs these to MLflow:

| Metric | Meaning | What "good" looks like |
|---|---|---|
| `final_eval_f1` | Weighted F1 across all vessel classes — **the primary metric** (`load_best_model_at_end` picks the best-F1 checkpoint) | **≥ 0.85** is the acceptance gate; ≥ 0.90 on named classes is the project target |
| `final_eval_accuracy` | Overall fraction correct | Tracks F1; large gap below F1 suggests class imbalance issues |
| `final_eval_precision_weighted` | Weighted precision | High = few false vessel assignments |
| `final_eval_recall_weighted` | Weighted recall | High = few missed assignments |
| `final_eval_mcc` | Matthews correlation coefficient — robust on imbalanced labels | > 0 means better than chance; closer to 1 is better |
| `prefilter_coverage_pct` | Share of emails the Stage-1 regex resolves without the model | Informational (≈ 3–5 % after hardening) |

How to read them:
- **Primary check:** is `final_eval_f1 ≥ 0.85`? If yes, the model passes the acceptance gate and is safe to deploy.
- **F1 ≫ accuracy or accuracy ≫ F1:** the long-tailed classes are dragging one metric — consider raising `--min-examples` (fewer, better-supported classes) or gathering more data for rare vessels.
- **Low MCC despite decent accuracy:** the model may be leaning on the dominant `UNKNOWN` class. The class-weighted loss already counteracts this; inspect the per-class behaviour before deploying.
- **Where to view:** AML Studio → the job's **Metrics** tab (charts), or **Models → imo-extractor-model** for the registered version's tags.

> *Optional, stronger evidence:* run [`evaluate_on_holdout.py`](evaluate_on_holdout.py) against `dataset_onlyWithImoFrom2023.csv` to measure performance on emails the model has **never seen** (a true temporal holdout) rather than the in-distribution validation split.

### Step 8 — Retrieve the registered model

`train.py` auto-registers the best checkpoint as **`imo-extractor-model`** at the end of the job (via `mlflow.transformers.log_model(...)`). Run the **"Retrieve the registered model"** cell — it fetches `imo-extractor-model:latest` and prints its version. If this raises, the job hasn't finished registering yet (or you only ran in-kernel dev training, which does not register).

### Step 9 — Deploy the online endpoint

Run the **"Deploy as Managed Online Endpoint"** cell. It:

1. Creates the endpoint `imo-extractor-endpoint` (key auth).
2. Builds the inference environment and deploys the registered model on a **1× T4 `Standard_NC16as_T4_v3`** instance with `score.py` wiring **Stage 1 (regex) → Stage 2 (neural)**.
3. Routes 100 % of traffic to the `default` deployment.

Tunable env vars baked into the deployment: `MIN_CONFIDENCE=0.70` (below this → `UNKNOWN` → human review), `MAX_SEQ_LEN=2048`. Provisioning takes several minutes; success prints `✓ Endpoint 'imo-extractor-endpoint' deployed`.

> The endpoint SKU is the **same Turing (sm_75) T4 generation** as the training cluster, so the checkpoint runs unchanged.

### Step 10 — Test the endpoint

Run the **"Test the endpoint"** cell. It sends two payloads:

1. **Explicit IMO** email → expect `predicted_imo="IMO9319466"`, `confidence` ≈ 1.0, `source="regex_prefilter"`, `requires_human_review=false`.
2. **Implicit-signal-only** email → handled by the neural model; `source="neural_classifier"`, and `requires_human_review=true` if `confidence < 0.70` or the prediction is `UNKNOWN`.

See [API Usage](#api-usage) below for the exact request/response shapes.

### Step 11 — Clean up (avoid surprise costs)

The training cluster scales to zero on its own, but the **endpoint bills continuously** while it exists. Delete it when you're done:

```bash
az ml online-endpoint delete --name imo-extractor-endpoint --yes
```

### Acceptance gates

| Stage | Pass criterion |
|---|---|
| Unit tests | 64 / 64 pass |
| AML Command Job | Finishes without OOM; `final_eval_f1` ≥ 0.85 on the validation split |
| Model registry | `imo-extractor-model:latest` resolves with a non-zero version |
| Endpoint | Both smoke-test calls return valid JSON; `requires_human_review` reflects the 70 % confidence threshold |

### Cost notes

- **Training:** `Standard_NC16as_T4_v3` (1× T4) ≈ 1.20 USD/hour, scales to zero when idle. A 15-epoch run typically completes in a few hours → a few USD per run. The optional 4× T4 `Standard_NC64as_T4_v3` upgrade is faster but needs ≥ 64-core cluster quota.
- **Inference:** `Standard_NC16as_T4_v3` ≈ 1.20 USD/hour, **billed continuously** while the deployment exists (it does **not** scale to zero). Delete the endpoint when not in use (Step 11).

---

## Overview

This solution converts incoming emails into vessel IMO number predictions for automatic ticket population. The model learns patterns from historical email data to identify which vessel an email is related to.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Incoming Email │───▶│ AML Online       │───▶│ Ticketing       │
│  (Subject,Body, │    │ Endpoint         │    │ System          │
│   Attachments)  │    │ (Real-time API)  │    │ (Auto-populate) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Human Review     │ (if confidence < 70%)
                       │ Queue            │
                       └──────────────────┘
```

## Components

### 1. Data Assets
- `email-imo-training-data`: Registered dataset with labeled emails

### 2. Training Pipeline
- **Data Preprocessing**: Merges email fields with structured `[SUBJECT]`/`[FROM]`/`[ATTACH]`/`[BODY]` tokens; collapses rare vessels into `UNKNOWN`
- **Model Training**: Fine-tunes `answerdotai/ModernBERT-base` (149 M params, 8192-token context) with class-weighted loss
- **MLflow Tracking**: Logs metrics, parameters, artifacts; auto-registers the best checkpoint

### 3. Model Registry
- Versioned models with metadata
- Tags for framework, task, domain

### 4. Online Endpoint
- Managed deployment with auto-scaling
- Key-based authentication
- Real-time inference API

## API Usage

The endpoint is served by [`score.py`](score.py). It accepts a **single email object**, a **list** of email objects, or `{"inputs": [ ... ]}`. Each email object uses the same field names as the training CSV.

### Request Format

Single email:
```json
{
  "emailSubject": "Parts Request for WEST AURIGA",
  "emailBody": "Please quote for spare parts...",
  "Attachments": "RFQ-12345.pdf;Invoice.pdf",
  "emailAddresses": "buyer@company.com supplier@company.com"
}
```

Batch of emails:
```json
{ "inputs": [ { "emailSubject": "...", "emailBody": "..." }, { "emailSubject": "..." } ] }
```

### Response Format

A single request returns one object; a batch returns a list of objects. `confidence` is a float in **[0, 1]** (not a percentage). `source` tells you which stage answered.

```json
{
  "predicted_imo": "IMO9609392",
  "confidence": 0.93,
  "raw_prediction": "IMO9609392",
  "min_confidence": 0.7,
  "requires_human_review": false,
  "source": "neural_classifier"
}
```

| Field | Meaning |
|---|---|
| `predicted_imo` | Predicted vessel IMO (e.g. `IMO9609392`) or `UNKNOWN` |
| `confidence` | Softmax confidence in [0, 1]; always `1.0` for regex-prefilter hits |
| `raw_prediction` | The model's top label before the confidence/`UNKNOWN` gate is applied |
| `min_confidence` | The active threshold (from `MIN_CONFIDENCE`, default `0.70`) |
| `requires_human_review` | `true` when `confidence < min_confidence` or the prediction is `UNKNOWN` |
| `source` | `regex_prefilter` (Stage 1, explicit IMO) or `neural_classifier` (Stage 2) |

## Model Performance

`train.py` logs these validation metrics to MLflow at the end of each run (visible in AML Studio → the job's **Metrics** tab):

| Metric (MLflow key) | Meaning | Target |
|---|---|---|
| `final_eval_f1` | Weighted F1 across all classes — **primary metric** | ≥ 0.85 (gate); ≥ 0.90 on named classes (goal) |
| `final_eval_accuracy` | Overall fraction correct | Tracks F1 |
| `final_eval_precision_weighted` | Weighted precision | High = few false assignments |
| `final_eval_recall_weighted` | Weighted recall | High = few missed assignments |
| `final_eval_mcc` | Matthews correlation coefficient (robust on imbalance) | Closer to 1 is better |
| `prefilter_coverage_pct` | Share resolved by the Stage-1 regex fast path | Informational (≈ 3–5 %) |

See [Step 7 — Interpret the metrics](#step-7--interpret-the-metrics) for how to read these together.

## Deployment Commands

```bash
# View endpoint
az ml online-endpoint show --name imo-extractor-endpoint

# Test endpoint
az ml online-endpoint invoke --name imo-extractor-endpoint \
  --request-file sample_emails.jsonl

# Scale deployment
az ml online-deployment update --name default \
  --endpoint-name imo-extractor-endpoint \
  --instance-count 3

# Delete the endpoint when done (it bills continuously)
az ml online-endpoint delete --name imo-extractor-endpoint --yes
```

## Human-in-the-Loop Workflow

1. Email arrives → API called
2. If confidence ≥ 70%: Auto-populate ticket
3. If confidence < 70%: Route to review queue
4. Human approves/corrects → Feedback stored for retraining
