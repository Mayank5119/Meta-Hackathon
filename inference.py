"""
Baseline Inference Script — Construction Superintendent OpenEnv
===============================================================
MANDATORY configuration (set as environment variables):
  API_BASE_URL  — LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier   (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      — Hugging Face / API key

Usage:
  python inference.py [--task_level easy|medium|hard] [--max_steps 20] [--seed 42]

The script runs ALL three task levels in sequence and prints a score table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType, Observation
from graders.grader import grade

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
TEMPERATURE = 0.0
MAX_TOKENS = 512

if not API_KEY:
    print("[WARNING] HF_TOKEN / API_KEY not set. LLM calls will fail.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert construction project manager (Site Superintendent).
You manage a construction schedule and must respond to disruptions to minimise delays and cost.

At each step you receive the current project state and must choose ONE action.

AVAILABLE ACTIONS:
1. expedite_task   — Add extra resources to a task to finish it faster (costs money)
   JSON: {"action_type": "expedite_task", "task_id": "T3", "days": 2}
   (days = number of extra resource units to add, 1–5)

2. delay_task      — Accept a delay on a task (absorbs the disruption)
   JSON: {"action_type": "delay_task", "task_id": "T3", "days": 3}

3. reassign_resources — Move 1 resource from one task to another
   JSON: {"action_type": "reassign_resources", "task_id": "T3", "target_task_id": "T5"}

4. noop            — Do nothing, let the simulation advance
   JSON: {"action_type": "noop"}

DECISION RULES (apply in order):
- If a disruption hits a CRITICAL PATH task → expedite_task (fix it fast)
- If a disruption hits a NON-CRITICAL task → delay_task (cheaper to absorb)
- If a task is blocking the critical path → reassign_resources from non-critical tasks
- If no disruptions are active → noop

Respond with ONLY valid JSON matching one of the action formats above.
No explanation, no markdown — just the JSON object.
"""


def build_user_message(obs: Observation) -> str:
    m = obs.metrics
    lines = [
        f"=== Day {obs.current_day} | Step {obs.episode_step} | Task: {obs.task_level} ===",
        f"Project end: original day {m.original_end_day}, projected day {m.current_projected_end_day} (delay: {m.delay_days}d)",
        f"Budget: ${m.budget_used:,.0f} used / ${m.budget_total:,.0f} total",
        f"Tasks: {m.tasks_completed}/{m.tasks_total} complete",
        "",
        "TASKS:",
    ]
    for t in obs.tasks:
        cp = " [CRITICAL PATH]" if t.is_on_critical_path else ""
        lines.append(
            f"  {t.id} {t.name}: {t.status.value}, start={t.current_start_day}, end={t.current_end_day}, "
            f"delay={t.delay_from_original}d, resources={t.resources}{cp}"
        )

    if obs.active_disruptions:
        lines.append("")
        lines.append("ACTIVE DISRUPTIONS:")
        for d in obs.active_disruptions:
            lines.append(f"  {d.id} [{d.type.value}] affects {d.affected_task_id}: {d.description} (+{d.remaining_delay_days}d)")

    lines.append("")
    lines.append("Choose ONE action (respond with JSON only):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def llm_select_action(obs: Observation) -> Action:
    """Call the LLM with the current observation and parse its action response."""
    user_msg = build_user_message(obs)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action_dict = json.loads(raw)
        return Action(**action_dict)
    except Exception as e:
        print(f"  [LLM error] {e} — falling back to noop")
        return Action(action_type=ActionType.NOOP)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_level: str, seed: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run one full episode and return results dict."""
    env = ConstructionEnv()
    obs = env.reset(task_level=task_level, seed=seed)

    total_reward = 0.0
    step_count = 0
    done = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE: {task_level.upper()} | Seed: {seed}")
        print(f"Original project end: day {obs.metrics.original_end_day}")
        print(f"Budget: ${obs.metrics.budget_total:,.0f}")
        print(f"{'='*60}")

    while not done:
        action = llm_select_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        if verbose:
            events = info.get("events", [])
            print(f"\nStep {step_count}: action={action.action_type} task={action.task_id or '-'}")
            for evt in events:
                print(f"  EVENT: {evt}")
            print(f"  Reward: {reward:.2f} | Total: {total_reward:.2f}")
            print(f"  Day: {obs.current_day} | Delay: {obs.metrics.delay_days}d | Budget used: ${obs.metrics.budget_used:,.0f}")

    final_state = env.state()
    result = grade(task_level, final_state)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE COMPLETE — {task_level.upper()}")
        print(f"Score: {result.score:.4f} ({'PASS' if result.passed else 'FAIL'})")
        print(f"Explanation: {result.explanation}")
        print(f"{'='*60}")

    return {
        "task_level": task_level,
        "total_reward": round(total_reward, 2),
        "steps": step_count,
        "grade": result.dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Construction Superintendent OpenEnv baseline inference.")
    parser.add_argument("--task_level", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    levels = ["easy", "medium", "hard"] if args.task_level == "all" else [args.task_level]
    results: List[Dict] = []

    for level in levels:
        result = run_episode(task_level=level, seed=args.seed, verbose=not args.quiet)
        results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("BASELINE SCORES SUMMARY")
    print("="*60)
    print(f"{'Task':<10} {'Score':>8} {'Pass':>6} {'Reward':>10} {'Steps':>6}")
    print("-"*60)
    for r in results:
        g = r["grade"]
        print(
            f"{r['task_level']:<10} {g['score']:>8.4f} {str(g['passed']):>6} "
            f"{r['total_reward']:>10.2f} {r['steps']:>6}"
        )
    print("="*60)

    # Machine-readable output
    print("\nJSON_SCORES:", json.dumps({r["task_level"]: r["grade"]["score"] for r in results}))


if __name__ == "__main__":
    main()
