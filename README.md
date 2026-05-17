# Azure Machine Learning - Email to IMO Classifier

## Customer Quickstart

This section is the minimum set of steps to reproduce the pipeline on a fresh Azure ML workspace. Estimated wall time: ~10 minutes of setup, then 2–4 hours for the training job on 4× T4.

### 1. Prerequisites

- Azure ML workspace in a region where you have GPU quota for `Standard_NC64as_T4_v3` (training, 4× T4) **and** `Standard_NC16as_T4_v3` (inference, 1× T4).
- An AML **compute instance** (any CPU SKU — e.g. `Standard_DS3_v2`) to host the orchestration notebook. The compute instance does **not** need a GPU; the GPUs are provisioned on demand by the AML Command Job and the online endpoint.

### 2. Clone the repository

On the compute instance terminal:

```bash
git clone <repo-url>
cd <repo-folder>
```

### 3. Upload the datasets

Two CSV files are shipped separately (not in git):

- `dataset_onlyWithImoUntil2023.csv` — training set (pre-2023, `;`-separated)
- `dataset_onlyWithImoFrom2023.csv` — temporal holdout (post-2023, `;`-separated), used only for evaluation

Drag-and-drop both files into the compute instance's home directory via AML Studio's file browser, or upload them to the workspace default datastore.

### 4. Set workspace environment variables

The notebook reads workspace identity from three env vars. On a compute instance the first three are usually set automatically; if not, export them once:

```bash
export AZUREML_SUBSCRIPTION_ID="<your-subscription-id>"
export AZUREML_RESOURCE_GROUP="<your-resource-group>"
export AZUREML_WORKSPACE="<your-workspace-name>"
```

Then restart the Jupyter kernel so the values are picked up.

### 5. (Optional) Run the unit tests

A 3-second smoke test that exercises the regex pre-filter, IMO checksum, request parsing and label encoding without touching Azure:

```bash
python3 -m pip install pytest
python3 -m pytest
```

Expected: `64 passed`.

### 6. Run the notebook

Open `imo_extractor_pipeline_CSV.ipynb` and run cells top to bottom **with one exception**: skip the in-kernel `trainer.train()` cell under "Fine-tune transformer model" — that cell is for local dev iteration only. The AML Command Job cell further down does the real, full-scale training on 4× T4.

Order of operations:

1. Setup & Configuration → Load & Preprocess Data → Regex Pre-filter → Tokenization (these run in the notebook kernel, no cloud cost).
2. **AML Command Job cell** → submits training on `Standard_NC64as_T4_v3`. The cluster auto-scales from zero, so cost only accrues while the job runs. Streams logs back to the notebook.
3. **Retrieve the registered model cell** → resolves `imo-extractor-model:latest` once training finishes and `train.py` has auto-registered the checkpoint.
4. **Deploy as Managed Online Endpoint** → provisions a 1× T4 endpoint and routes 100 % traffic to it.
5. **Test the endpoint** → smoke-test with two sample payloads.

### 7. Acceptance gates

| Stage | Pass criterion |
|---|---|
| Unit tests | 64 / 64 pass |
| AML Command Job | Finishes without OOM/DDP hang; `eval_f1` ≥ 0.85 on validation split |
| Model registry | `imo-extractor-model:latest` resolves with non-zero version |
| Endpoint | Both smoke-test calls return valid JSON; `requires_human_review` reflects the confidence threshold |

### 8. Cost notes

- Training cluster: `Standard_NC64as_T4_v3` ≈ 4.35 USD/hour, scales to zero when idle. A 15-epoch run on the full dataset should complete in 2–4 hours → roughly 10–20 USD per training run.
- Inference endpoint: `Standard_NC16as_T4_v3` ≈ 1.20 USD/hour, **billed continuously** while the deployment exists (it does not scale to zero). Delete the endpoint when not in use.

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
- **Data Preprocessing**: Cleans emails, creates text features
- **Model Training**: Trains classifier (BERT/RNN/CNN/FastText)
- **MLflow Tracking**: Logs metrics, parameters, artifacts

### 3. Model Registry
- Versioned models with metadata
- Tags for framework, task, domain

### 4. Online Endpoint
- Managed deployment with auto-scaling
- Key-based authentication
- Real-time inference API

## API Usage

### Request Format
```json
{
  "emails": [
    {
      "subject": "Parts Request for WEST AURIGA",
      "from": "buyer@company.com",
      "attachments": "RFQ-12345.pdf;Invoice.pdf",
      "body": "Please quote for spare parts..."
    }
  ],
  "confidence_threshold": 0.7
}
```

### Response Format
```json
{
  "predictions": [
    {
      "predicted_imo": "9609392",
      "predicted_vessel": "WEST AURIGA",
      "confidence": 92.5,
      "requires_review": false
    }
  ]
}
```

## Model Performance

| Model    | MCC   | F1-Score | Accuracy |
|----------|-------|----------|----------|
| FastText | ~0.65 | ~0.70    | ~75%     |
| CNN      | ~0.70 | ~0.75    | ~80%     |
| RNN      | ~0.72 | ~0.78    | ~82%     |
| BERT     | ~0.80 | ~0.85    | ~88%     |

## Deployment Commands

```bash
# View endpoint
az ml online-endpoint show --name email-imo-endpoint-xxx

# Test endpoint
az ml online-endpoint invoke --name email-imo-endpoint-xxx \
  --request-file test_request.json

# Scale deployment
az ml online-deployment update --name blue \
  --endpoint-name email-imo-endpoint-xxx \
  --instance-count 3
```

## Human-in-the-Loop Workflow

1. Email arrives → API called
2. If confidence ≥ 70%: Auto-populate ticket
3. If confidence < 70%: Route to review queue
4. Human approves/corrects → Feedback stored for retraining
