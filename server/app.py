"""
FastAPI server exposing the Construction Superintendent OpenEnv environment.

Endpoints:
- GET /        : health check (returns 200)
- POST /reset  : reset environment, returns Observation
- POST /step   : take action, returns StepResult
- GET /state   : return current internal state
- POST /grade  : run grader on current state, return GradeResult
- GET /tasks   : list available task levels
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.construction_env import ConstructionEnv
from env.models import (
    Action,
    GradeResult,
    Observation,
    ResetRequest,
    StepResult,
)
from graders.grader import grade

app = FastAPI(
    title="Construction Superintendent OpenEnv",
    description=(
        "An OpenEnv-compliant RL environment where an AI agent manages a "
        "real construction project schedule, responding to disruptions "
        "(weather, material shortages, equipment failures) to minimise "
        "delay and cost overrun."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful server)
env = ConstructionEnv()
last_reset_level: str = "easy"

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": "construction-superintendent-openenv", "version": "1.0.0"}

@app.get("/tasks", tags=["Info"])
def list_tasks() -> Dict[str, Any]:
    return {
        "task_levels": ["easy", "medium", "hard"],
        "descriptions": {
            "easy": "5-task linear residential build, 1 disruption. Good for quick evaluation.",
            "medium": "8-task commercial build with parallel MEP tracks, 3 disruptions.",
            "hard": "10-task mixed-use complex, 5 disruptions, tight budget. Challenges frontier models.",
        }
    }

@app.post("/reset", response_model=Observation, tags=["OpenEnv"])
def reset(request: ResetRequest = ResetRequest()) -> Observation:
    """Reset environment to initial state and return first observation."""
    global last_reset_level
    last_reset_level = request.task_level
    try:
        obs = env.reset(task_level=request.task_level, seed=request.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
def step(action: Action) -> StepResult:
    """Submit an action and advance the simulation."""
    obs, reward, done, info = env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    """Return the full internal environment state."""
    return env.state()

@app.post("/grade", response_model=GradeResult, tags=["Grading"])
def grade_current() -> GradeResult:
    """Grade the current episode against the task-level rubric."""
    current_state = env.state()
    return grade(last_reset_level, current_state)

@app.post("/grade/{task_level}", response_model=GradeResult, tags=["Grading"])
def grade_task(task_level: str) -> GradeResult:
    """Grade for a specific task level using the current environment state."""
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="task_level must be easy | medium | hard")
    return grade(task_level, env.state())

def main():
    import uvicorn
    # This matches the port from the original Dockerfile
    uvicorn.run("server.app:app", host="0.0.0.0", port=7060)

if __name__ == "__main__":
    main()