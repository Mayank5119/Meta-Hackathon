# Submission Guide - Construction Superintendent OpenEnv

Step-by-step instructions to validate and deploy the environment for the OpenEnv hackathon.

## Prerequisites

```bash
# Python 3.11+
python --version

# Docker Desktop (must be running)
docker --version

# HF CLI
pip install huggingface_hub

# OpenEnv validator
pip install openenv-core
```

## Step 1 - Local smoke test (no API key needed)

Verify the core environment logic before touching any deployment tooling.

```bash
# Install server + inference deps only
pip install -r requirements.txt

# Run smoke test (uses NOOP + heuristic actions, no LLM)
python test_env.py
```

Expected output:
```text
[easy] reset OK - 5 tasks, original_end=21d, budget=$90,000
steps=6 reward=... score=0.78xx passed=True
[medium] reset OK - 8 tasks...
[hard] reset OK - 10 tasks...
All levels OK.
```

If this fails, the problem is in the core env code - fix before continuing.

## Step 2 - Run `openenv validate`

```bash
# From the repo root directory
openenv validate
```

This reads `openenv.yaml` and checks schema compliance. It will also attempt to call the `/reset` endpoint, so you need the server running in a separate terminal:

```bash
# Terminal 1 - start the server
uvicorn api.server:app --host 0.0.0.0 --port 7060
# or
python -m uvicorn server.app:app --host 0.0.0.0 --port 7060

# Terminal 2 - run the validator
openenv validate
```

**Expected:** "All checks passed."

*If it fails:* Check the error message carefully. Common issues:
- Missing required field in `openenv.yaml` -> add it
- `/reset` returns unexpected schema -> check `Observation` Pydantic model

## Step 3 - Docker build and local test

```bash
# Build image (takes 2-3 minutes first time - torch is NOT included)
docker build -t construction-env .

# Run locally
docker run -p 7060:7060 construction-env
```

Verify the server is live:
```bash
curl http://localhost:7060/
# Expected: {"status":"ok","environment":"construction-superintendent-openenv","version":"1.0.0"}
```

Test `/reset` endpoint:
```bash
curl -X POST http://localhost:7060/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy", "seed": 42}'
```
or

Invoke-RestMethod -Uri "http://localhost:7060/reset" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"task_level": "easy", "seed": 42}'

Test `/step` endpoint:
```bash
curl -X POST http://localhost:7060/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "noop"}'
```
or

Invoke-RestMethod -Uri "http://localhost:7060/step" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"action_type": "noop"}'

Test `/grade` endpoint:
```bash
curl -X POST http://localhost:7060/grade
```
or

Invoke-RestMethod -Uri "http://localhost:7060/grade" -Method POST

If `/reset` returns a valid `Observation` JSON, the Docker image is correct.

## Step 4 - Deploy to Hugging Face Spaces (Docker Space)

### 4a. Create the Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Set:
   - **Space name:** `construction-superintendent`
   - **License:** MIT or Apache 2.0
   - **Select the Space SDK:** Docker -> Blank
   - **Space Hardware:** Free (CPU) is fine.
   - **Visibility:** Public
4. Click "Create Space"

### 4b. Push the code

```bash
# Login to HF
huggingface-cli login
# Paste your HF token when prompted

# Clone the empty space repo (replace YOUR_USERNAME)
git clone [https://huggingface.co/spaces/YOUR_USERNAME/construction-superintendent](https://huggingface.co/spaces/YOUR_USERNAME/construction-superintendent)
cd construction-superintendent

# Copy all project files into it
# (or set the remote on your existing repo)
git remote add hf [https://huggingface.co/spaces/YOUR_USERNAME/construction-superintendent](https://huggingface.co/spaces/YOUR_USERNAME/construction-superintendent)

# or just 1 instead of all above 3
python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='.', repo_id='Mayank3290/construction-superintendent', repo_type='space')"

# Push
git add .
git commit -m "Initial submission"
git push hf main
```

### 4c. Monitor the build

- Go to your Space URL -> "Logs" tab
- Wait for `Application is running on port 7060`
- Your Space URL will be: `https://YOUR_USERNAME-construction-superintendent.hf.space`

### 4d. Verify the Space is live

```bash
curl -X POST https://YOUR_USERNAME-construction-superintendent.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy", "seed": 42}'
```
<!-- or -->
Invoke-RestMethod -Uri "https://mayank3290-construction-superintendent.hf.space/" -Method GET

Should return an `Observation` JSON with `"current_day": 0`.

### Step 4e: Required README Metadata
Every Space needs a YAML header in README.md. Ensure the following is at the top:

YAML
---
title: Construction Superintendent
sdk: docker
app_port: 7060
---

## Step 5 - Run the pre-validation script

```bash
chmod +x validate-submission.sh  # Linux/Mac only
./validate-submission.sh YOUR_USERNAME-construction-superintendent.hf.space
```

On Windows, run each check manually:

```bash
# Check 1: Space responds
curl -s -o /dev/null -w "%{http_code}" -X POST \
  https://YOUR_USERNAME-construction-superintendent.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}'
# Expected: 200

# Check 2: Docker build
docker build .
# Expected: exit code 0

# Check 3: OpenEnv validate
openenv validate
# Expected: All checks passed
```

All three must pass before submitting.

## Step 6 - Run baseline inference (requires API key)

```bash
# Set credentials
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Run all three task levels
python inference.py --task_level all --seed 42
```

Expected stdout format:
```text
[START] task=easy env=construction-superintendent model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"action_type": "noop"} reward=0.0 done=False error=null
...
[END] success=True steps=6 score=0.78 rewards=[...]
[START] task=medium...
...
JSON_SCORES: {"easy": 0.78, "medium": 0.61, "hard": 0.52}
```

The `JSON_SCORES` line at the end is your reproducible baseline.

## Step 7 - Final submission checklist

- [ ] `openenv validate` passes locally
- [ ] Docker build succeeds with no errors
- [ ] HF Space URL responds to `/POST /reset` with HTTP 200
- [ ] Pre-validation script shows All 3/3 checks passed
- [ ] `inference.py --seed 42` produces `JSON_SCORES` with scores > 0 for all levels
- [ ] README has baseline scores filled in
- [ ] Space is tagged with "openenv" in the HF Space settings

## Troubleshooting

### `openenv validate` fails with schema error

The validator likely checks that `Observation` fields match `openenv.yaml`. If a field name changed in code but not in the YAML (or vice versa), update both. Run the server first, then `openenv validate`.

### Docker build fails on `uvicorn[standard]`

```bash
# Explicitly test the requirements install
docker run --rm python:3.11-slim pip install -r requirements.txt
```

### HF Space stuck on "Building"

- Check the **"Logs"** tab on HF Spaces for the actual error
- Common cause: `requirements.txt` has a package that fails to install (e.g., C extension)
- Since we removed `torch` from `requirements.txt`, this should no longer be an issue

### `/reset` returns 422 Unprocessable Entity

The request body doesn't match `ResetRequest`. Send `{}` or `{"task_level": "easy"}`.

### Inference script produces no `[START]` line

Make sure you're running the updated `inference.py`, not a cached `.pyc` file:
```bash
find . -name "*.pyc" -delete
python inference.py --task_level easy --seed 42
```

## Local dev setup (with agent + UI)

```bash
# Full dev environment (includes torch, gradio, etc.)
pip install -r requirements-dev.txt

# Run gradio UI locally
python gradio_app.py

# Train DQN agent
python agent/train.py
```

*Team Cockroach - Mayank Vaishya - Harsh Pundir - Dhruvv*