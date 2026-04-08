# Baseline Inference script - Construction Superintendent OpenEnv
# MANDATORY stdout format (one episode block per task level):
#
# [START] task=easy env=construction-superintendent model=my-model
# [STEP] step=1 action={"action_type": "noop"} reward=0.0 done=False error=null
# ...
# [END] success=True steps=15 score=0.78 rewards=[...]
# JSON_SCORES: {"easy": 0.78}

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType, Observation
from graders.grader import grade

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
BENCHMARK = "construction-superintendent"
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.50 # grader score in [0, 1]

if not API_KEY:
    print("[WARNING] HF_TOKEN / API_KEY not set. LLM calls will fail.", file=sys.stderr)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

# =============================================================================
# Mandatory stdout logging helpers
# =============================================================================

def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str] = None
):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards=[{rewards_str}]",
        flush=True,
    )

def action_to_str(action: Action) -> str:
    """Compact single-line JSON for the [STEP] action field."""
    return json.dumps(
        {k: v for k, v in action.dict().items() if v is not None},
        separators=(',', ':')
    )

# =============================================================================
# Prompt engineering
# =============================================================================

SYSTEM_PROMPT = """
You are an expert construction project manager (Site Superintendent).
You manage a construction schedule and must respond to disruptions to minimise delays and cost.

At each step you receive the current project state and must choose ONE action.

AVAILABLE ACTIONS:
1. expedite_task - Add extra resources to a task to finish it faster (costs money)
   JSON: {"action_type": "expedite_task", "task_id": "T1", "days": 2}
   (days = number of extra resource units to add, 1-5)

2. delay_task - Accept a delay on a task (absorbs the disruption cheaply)
   JSON: {"action_type": "delay_task", "task_id": "T3", "days": 3}

3. reassign_resources - Move 1 resource from one task to another
   JSON: {"action_type": "reassign_resources", "task_id": "T2", "target_task_id": "T5"}

4. noop - Do nothing, let the simulation advance
   JSON: {"action_type": "noop"}

DECISION RULES (apply in order):
- If a disruption hits a CRITICAL PATH task -> expedite_task (fix it fast)
- If a disruption hits a NON-CRITICAL task -> delay_task (cheaper to absorb)
- If a task is blocking the critical path -> reassign_resources from non-critical tasks
- If no disruptions are active -> noop

Respond with ONLY valid JSON matching one of the action formats above.
No explanation, no markdown - just the raw JSON object on a single line.
"""

def build_user_message(obs: Observation) -> str:
    m = obs.metrics
    lines = [
        f"--- Day {obs.current_day} | Step {obs.episode_step} | {obs.task_level} ---",
        f"Project end: original day {m.original_end_day}, projected day {m.current_projected_end_day} (delay: {m.delay_days}d)",
        f"Budget: ${m.budget_used:.0f} used / ${m.budget_total:.0f} total",
        f"Tasks: {m.tasks_completed}/{m.tasks_total} complete",
        "",
        "TASKS:"
    ]
    for t in obs.tasks:
        cp = " [CRITICAL PATH]" if t.is_on_critical_path else ""
        lines.append(
            f"- {t.id} ({t.name}){cp}: {t.status.value}, "
            f"days: {t.current_start_day}-{t.current_end_day}, "
            f"delay: {t.delay_from_original}d, resources: {t.resources}"
        )
    if obs.active_disruptions:
        lines.append("\nACTIVE DISRUPTIONS:")
        for d in obs.active_disruptions:
            lines.append(
                f"- [{d.id}] {d.type.value} affects {d.affected_task_id}: "
                f"{d.description} (+{d.remaining_delay_days}d)"
            )
    lines.append("\nChoose ONE action (respond with JSON only):")
    return "\n".join(lines)

def llm_select_action(obs: Observation) -> Tuple[Action, Optional[str]]:
    """
    Call the LLM with the current observation and parse its action.
    Returns (action, error_string_or_none).
    On any failure, falls back to noop and returns the error string.
    """
    raw = ""
    user_msg = build_user_message(obs)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
        
        # Strip markdown code fences if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        return Action(**json.loads(raw)), None
    except Exception as exc:
        err = str(exc)
        print(f"[WARN] LLM parse error: {err} | Raw: {raw}", file=sys.stderr)
        return Action(action_type=ActionType.NOOP), err

# =============================================================================
# Episode runner
# =============================================================================

def run_episode(task_level: str, seed: Optional[int]) -> float:
    """
    Run one full episode for the given task level.
    Emits the mandatory [START] / [STEP] / [END] lines to stdout.
    Returns the grader score in [0.0, 1.0].
    """
    env = ConstructionEnv()
    obs = env.reset(task_level=task_level, seed=seed)
    
    log_start(task_level, env_name=BENCHMARK, model=MODEL_NAME)
    
    steps_taken = 0
    rewards: List[float] = []
    score = 0.0
    success = False
    
    done = False
    
    try:
        while not done:
            action, parse_error = llm_select_action(obs)
            action_str = action_to_str(action)
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            # Detect env-level invalid actions (separate from LLM parse errors)
            step_error = parse_error
            if not step_error and info.get("reward_breakdown", {}).get("invalid_action_penalty", 0) < 0:
                step_error = f"invalid_action: {action.action_type}"
                
            log_step(
                step=steps_taken,
                action_str=action_str,
                reward=reward,
                done=done,
                error=step_error,
            )
            steps_taken += 1
            
        # Grade the completed episode
        final_state = env.state()
        result = grade(task_level, final_state)
        score = result.score
        success = score >= SUCCESS_SCORE_THRESHOLD
            
    except Exception as exc:
        err_msg = str(exc)
        print(f"[ERROR] Episode exception: {err_msg}", file=sys.stderr)
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
    return score

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construction Superintendent OpenEnv - baseline inference script."
    )
    parser.add_argument(
        "--task_level",
        type=str,
        default="easy",
        help="Task difficulty level to run (default: all three in sequence).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()
    
    levels = ["easy", "medium", "hard"] if args.task_level == "all" else [args.task_level]
    scores: Dict[str, float] = {}
    
    for level in levels:
        score = run_episode(task_level=level, seed=args.seed)
        scores[level] = score
        
    # Machine-readable summary line for CI / eval pipelines
    print(f"JSON_SCORES: {json.dumps(scores)}", flush=True)

if __name__ == "__main__":
    main()