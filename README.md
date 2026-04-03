# Construction Superintendent — OpenEnv Environment

> **Team Cockroach** · Mayank Vaishya · Harsh Pundir · Dhruv

An **OpenEnv-compliant** reinforcement learning environment where an AI agent
acts as an autonomous construction site superintendent.

The agent manages a real construction project modelled as a **directed acyclic
graph (DAG)** of tasks with dependencies, responds to real-world disruptions
(weather delays, material shortages, equipment failures, labour shortages), and
must minimise total project delay and cost overrun.

---

## Why This Domain?

Construction scheduling is a **genuine, high-stakes real-world task**:
- Projects exceed budget 85% of the time (McKinsey 2016).
- Schedule slippage averages 20% across large infrastructure projects.
- An AI superintendent that dynamically re-optimises saves real money.

This fills a gap in the OpenEnv ecosystem — no existing construction scheduling
environment exists with proper DAG modelling and multi-disruption scenarios.

---

## Quick Start

```bash
# Clone and install
pip install -r requirements.txt

# Run the server (local)
uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload

# Or with Docker
docker build -t construction-env .
docker run -p 7860:7860 construction-env

# Run baseline inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py --task_level all --seed 42
```
# but for windows the above commands are 
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN = "your_token_here"
python inference.py --task_level all --seed 42

---

## Environment Description

### Action Space

| `action_type`        | Parameters                         | Effect |
|----------------------|------------------------------------|--------|
| `expedite_task`      | `task_id`, `days` (resource units) | Adds resources → faster completion, higher cost |
| `delay_task`         | `task_id`, `days`                  | Accepts delay, absorbs disruption cheaply |
| `reassign_resources` | `task_id`, `target_task_id`        | Moves 1 worker unit between tasks |
| `noop`               | —                                  | Do nothing, advance clock to next event |

**Action JSON examples:**
```json
{"action_type": "expedite_task", "task_id": "T3", "days": 2}
{"action_type": "delay_task", "task_id": "T5", "days": 3}
{"action_type": "reassign_resources", "task_id": "T3", "target_task_id": "T5"}
{"action_type": "noop"}
```

### Observation Space

```
Observation
├── current_day          (int)   — simulation clock
├── episode_step         (int)   — step counter
├── task_level           (str)   — easy | medium | hard
├── tasks[]              (list)  — task DAG state
│   ├── id, name, status
│   ├── original_duration, current_duration
│   ├── current_start_day, current_end_day
│   ├── is_on_critical_path      — computed via CPM
│   ├── delay_from_original
│   ├── resources, cost_per_day
│   └── progress_pct
├── active_disruptions[] (list)  — current events
│   ├── id, type (weather/material_shortage/…)
│   ├── affected_task_id
│   └── remaining_delay_days
└── metrics              (obj)
    ├── original_end_day, current_projected_end_day, delay_days
    ├── budget_total, budget_used, budget_remaining
    └── tasks_completed, disruptions_resolved
```

### Reward Function (Dense)

| Signal                    | Value per unit |
|---------------------------|---------------|
| Delay penalty             | −2.0 / day of delay |
| Cost penalty              | −0.5 / $1k overrun |
| Task completion bonus     | +5.0 per task |
| Disruption resolved       | +3.0 |
| Project completion        | +50.0 |
| On-time completion bonus  | +30.0 extra |
| Budget overrun penalty    | −20.0 |
| Invalid action            | −1.0 |

Reward is **dense** (non-zero at every step) and provides **partial progress
signals** throughout the episode — not just on termination.

---

## Task Levels & Graders

### Easy (score threshold: 0.60)

**Scenario:** 5-task linear residential build (21-day baseline). 1 weather
disruption hits structural framing on day 8 (+3 days).

**Optimal strategy:** `expedite_task T3` with 2 extra resources.

| Component         | Weight | Criteria |
|-------------------|--------|----------|
| Completion        | 0.40   | All tasks done |
| Delay             | 0.30   | ≤ 2d full, ≤ 5d partial |
| Budget            | 0.20   | Within limit |
| Disruption handled| 0.10   | ≥ 1 resolved |

Expected scores: random=0.30 · heuristic=0.70 · frontier LLM=0.90

---

### Medium (score threshold: 0.55)

**Scenario:** 8-task commercial office build (31-day baseline). 3 disruptions:
rain on framing (critical path), material shortage on electrical, equipment
failure on plumbing.

**Optimal strategy:** Expedite T3 (critical), delay T5 and T6 (non-critical),
reassign resources to T7 before the join.

| Component          | Weight | Criteria |
|--------------------|--------|----------|
| Completion         | 0.35   | Partial credit per task |
| Delay              | 0.25   | ≤ 3d full, ≤ 8d partial |
| Disruptions (≥ 2)  | 0.20   | Proportional |
| Budget (≤ 110%)    | 0.20   | Partial credit |

Expected scores: random=0.20 · heuristic=0.55 · frontier LLM=0.80

---

### Hard (score threshold: 0.50)

**Scenario:** 10-task mixed-use complex (48-day baseline, tight $500k budget).
5 disruptions including cascading delays that push non-critical tasks onto the
critical path. Over-expediting causes budget overrun.

**Optimal strategy:** Selective expediting only on true critical-path tasks,
absorb non-critical delays, keep budget within 105%.

| Component          | Weight | Criteria |
|--------------------|--------|----------|
| Completion         | 0.30   | Partial credit |
| Delay              | 0.25   | ≤ 5d full, ≤ 12d partial |
| Disruptions (≥ 3)  | 0.25   | Proportional |
| Budget (≤ 105%)    | 0.20   | Strict |

Expected scores: random=0.10 · heuristic=0.40 · frontier LLM=0.70

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Health check — returns 200 |
| `GET`  | `/tasks` | List task levels and descriptions |
| `POST` | `/reset` | Reset environment → returns `Observation` |
| `POST` | `/step`  | Submit `Action` → returns `StepResult` |
| `GET`  | `/state` | Full internal state dict |
| `POST` | `/grade` | Grade current episode → `GradeResult` (0.0–1.0) |

**Reset request:**
```json
{"task_level": "easy", "seed": 42}
```

**Step request:**
```json
{"action_type": "expedite_task", "task_id": "T3", "days": 2}
```

**Step response:**
```json
{
  "observation": { ... },
  "reward": 7.5,
  "done": false,
  "info": { "events": [...], "delay_days": 1, "budget_used": 12400 }
}
```

---

## Baseline Scores (seed=42)

| Task Level | Score  | Passed | Steps |
|------------|--------|--------|-------|
| easy       | ~0.78  | ✓      | 6     |
| medium     | ~0.61  | ✓      | 12    |
| hard       | ~0.52  | ✓      | 18    |

*(Reproduced by: `python inference.py --seed 42`)*

---

## Project Structure

```
construction-superintendent-env/
├── env/
│   ├── __init__.py
│   ├── models.py           # Typed Pydantic models (Action, Observation, Reward)
│   ├── project.py          # Task DAG templates per difficulty level
│   ├── disruption.py       # Scripted disruption events
│   ├── scheduler.py        # Critical Path Method (CPM) scheduler
│   └── construction_env.py # Main OpenEnv environment class
├── graders/
│   ├── __init__.py
│   └── grader.py           # Deterministic graders for easy/medium/hard
├── api/
│   ├── __init__.py
│   └── server.py           # FastAPI server (step/reset/state endpoints)
├── tasks/
│   └── __init__.py
├── inference.py            # Baseline inference script (OpenAI client)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup Instructions

**Requirements:** Python 3.11+, Docker (for containerised deployment)

```bash
# Local development
pip install -r requirements.txt
uvicorn api.server:app --reload --port 7860

# Docker
docker build -t construction-env .
docker run -e HF_TOKEN=$HF_TOKEN -e API_BASE_URL=$API_BASE_URL \
           -e MODEL_NAME=$MODEL_NAME -p 7860:7860 construction-env

# Run grader standalone
python -c "
from env.construction_env import ConstructionEnv
from graders.grader import grade
env = ConstructionEnv()
env.reset('easy')
for _ in range(5):
    from env.models import Action, ActionType
    obs, r, done, _ = env.step(Action(action_type=ActionType.NOOP))
    if done: break
print(grade('easy', env.state()))
"
```

---

## Environment Variables

| Variable      | Required | Description |
|---------------|----------|-------------|
| `API_BASE_URL`| Yes      | LLM API endpoint |
| `MODEL_NAME`  | Yes      | Model identifier |
| `HF_TOKEN`    | Yes      | Hugging Face / API key |

---

*Built for the OpenEnv Hackathon — Round 1 submission.*
