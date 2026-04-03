"""
Gradio UI — Construction Superintendent AI
==========================================
Three tabs:
  1. Run Episode    — run a full episode with Heuristic or PyTorch DQN agent
  2. Train DQN      — configure + launch training with live progress curves
  3. Manual Explorer — step through the environment by hand

Launch:
    python gradio_app.py
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType, Observation
from graders.grader import grade
from agent.pytorch_agent import DQNAgent, encode_observation
from agent.train import run_training, DEFAULT_CHECKPOINT

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic baseline agent (no model required)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_action(obs: Observation) -> Action:
    """Rule-based agent: expedite critical-path disruptions, delay others."""
    for d in obs.active_disruptions:
        if d.resolved:
            continue
        task = next((t for t in obs.tasks if t.id == d.affected_task_id), None)
        if task and task.is_on_critical_path:
            return Action(action_type=ActionType.EXPEDITE_TASK, task_id=task.id, days=2)
        elif task:
            return Action(action_type=ActionType.DELAY_TASK, task_id=task.id, days=d.total_delay_days)
    return Action(action_type=ActionType.NOOP)


# ─────────────────────────────────────────────────────────────────────────────
# Visual helpers
# ─────────────────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "pending":     "#aec6cf",
    "in_progress": "#fdfd96",
    "completed":   "#77dd77",
    "disrupted":   "#ff6961",
    "blocked":     "#cfcfc4",
}

BG_DARK  = "#1a1a2e"
BG_MID   = "#16213e"
ACCENT   = "#e94560"
CYAN     = "#00b4d8"
BORDER   = "#0f3460"


def _ax_dark(ax):
    ax.set_facecolor(BG_MID)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def build_gantt(obs: Observation) -> plt.Figure:
    """Horizontal bar chart showing current task schedule."""
    tasks = sorted(obs.tasks, key=lambda t: t.current_start_day)
    fig, ax = plt.subplots(figsize=(10, max(3, len(tasks) * 0.65 + 1)))
    fig.patch.set_facecolor(BG_DARK)
    _ax_dark(ax)

    for i, t in enumerate(tasks):
        status_str = t.status.value if hasattr(t.status, "value") else str(t.status)
        color = STATUS_COLORS.get(status_str, "#aaaaaa")
        ax.barh(
            i, t.current_duration, left=t.current_start_day,
            color=color, edgecolor=BORDER, height=0.6, alpha=0.9,
        )
        label = f"{t.id}: {t.name}"
        if t.is_on_critical_path:
            label += " ★"
        ax.text(
            t.current_start_day + 0.2, i, label,
            va="center", ha="left", fontsize=8, color="white",
        )

    ax.axvline(obs.current_day, color=ACCENT, linewidth=2, label=f"Day {obs.current_day}", zorder=5)
    ax.set_yticks([])
    ax.set_xlabel("Day", color="white")
    ax.set_title("Project Schedule  (★ = Critical Path)", color="white", fontsize=11)
    ax.xaxis.label.set_color("white")

    legend_patches = [mpatches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    legend_patches.append(mpatches.Patch(color=ACCENT, label=f"Day {obs.current_day}"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              facecolor=BORDER, labelcolor="white", framealpha=0.9)
    plt.tight_layout()
    return fig


def build_episode_fig(metrics_log: List[Dict]) -> plt.Figure:
    """Reward-per-step and delay-per-step chart for a completed episode."""
    if not metrics_log:
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor(BG_DARK)
        _ax_dark(ax)
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color="white")
        return fig

    steps   = [m["step"]   for m in metrics_log]
    rewards = [m["reward"] for m in metrics_log]
    delays  = [m["delay"]  for m in metrics_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2):
        _ax_dark(ax)

    ax1.plot(steps, rewards, color=CYAN, linewidth=1.5, marker="o", markersize=3)
    ax1.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Step Reward", color="white")

    ax2.bar(steps, delays, color=ACCENT, alpha=0.75, width=0.6)
    ax2.set_ylabel("Delay (days)", color="white")
    ax2.set_xlabel("Step", color="white")

    fig.suptitle("Episode Metrics", color="white", fontsize=12)
    plt.tight_layout()
    return fig


def build_training_fig(rewards: List[float], losses: List[float]) -> plt.Figure:
    """Smoothed reward curve + loss curve during DQN training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax1, ax2):
        _ax_dark(ax)

    if rewards:
        ep = list(range(1, len(rewards) + 1))
        window = max(1, len(rewards) // 20)
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(ep, rewards, color=CYAN, alpha=0.25, linewidth=0.8)
        ax1.plot(ep[window - 1:], smooth, color=CYAN, linewidth=2, label="smoothed")
        ax1.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Episode Reward", color="white")
        ax1.legend(facecolor=BORDER, labelcolor="white", fontsize=8)

    if losses:
        ep = list(range(1, len(losses) + 1))
        ax2.plot(ep, losses, color="#ff9f43", alpha=0.6, linewidth=0.9)
        ax2.set_ylabel("Avg Loss", color="white")
        ax2.set_xlabel("Episode", color="white")

    fig.suptitle("DQN Training Progress", color="white", fontsize=12)
    plt.tight_layout()
    return fig


def obs_to_task_df(obs: Observation) -> pd.DataFrame:
    rows = []
    for t in obs.tasks:
        status_str = t.status.value if hasattr(t.status, "value") else str(t.status)
        rows.append({
            "ID":        t.id,
            "Name":      t.name,
            "Status":    status_str,
            "Start":     t.current_start_day,
            "End":       t.current_end_day,
            "Delay (d)": t.delay_from_original,
            "Progress":  f"{t.progress_pct:.0f}%",
            "Critical":  "★" if t.is_on_critical_path else "",
            "Resources": t.resources,
        })
    return pd.DataFrame(rows)


def obs_to_disruption_df(obs: Observation) -> pd.DataFrame:
    if not obs.active_disruptions:
        return pd.DataFrame(columns=["ID", "Type", "Affects", "Delay (d)", "Resolved"])
    rows = []
    for d in obs.active_disruptions:
        rows.append({
            "ID":       d.id,
            "Type":     d.type.value if hasattr(d.type, "value") else str(d.type),
            "Affects":  d.affected_task_id,
            "Delay (d)": d.total_delay_days,
            "Resolved": "Yes" if d.resolved else "No",
        })
    return pd.DataFrame(rows)


def metrics_md(obs: Observation) -> str:
    m = obs.metrics
    budget_pct = m.budget_used / max(m.budget_total, 1) * 100
    cp_flag = " | **Critical path delayed!**" if m.on_critical_path_delayed else ""
    return (
        f"**Day:** {obs.current_day}  |  "
        f"**Step:** {obs.episode_step}  |  "
        f"**Level:** `{obs.task_level}`  |  "
        f"**Delay:** {m.delay_days}d  |  "
        f"**Budget:** ${m.budget_used:,.0f} / ${m.budget_total:,.0f} "
        f"({budget_pct:.1f}%)  |  "
        f"**Tasks:** {m.tasks_completed}/{m.tasks_total}"
        f"{cp_flag}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared mutable state
# ─────────────────────────────────────────────────────────────────────────────

_training_state: Dict = {
    "running":    False,
    "stop":       [False],
    "rewards":    [],
    "losses":     [],
    "grades":     [],
    "episode":    0,
    "total":      0,
    "task_level": "easy",
}

_manual_env = ConstructionEnv()
_manual_obs: Optional[Observation] = None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Run Episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_ui(task_level: str, seed: int, agent_type: str):
    """Run a complete episode and return all display components."""
    env = ConstructionEnv()
    obs = env.reset(task_level=task_level, seed=int(seed))
    task_ids = [t.id for t in obs.tasks]

    # Load DQN if selected
    dqn_agent: Optional[DQNAgent] = None
    if agent_type == "PyTorch DQN":
        ckpt = DEFAULT_CHECKPOINT.format(level=task_level)
        if not os.path.exists(ckpt):
            no_ckpt = (
                f"**No checkpoint found** for `{task_level}`.  \n"
                f"Please go to the **Train DQN** tab and train first, then retry."
            )
            return no_ckpt, pd.DataFrame(), pd.DataFrame(), None, None
        dqn_agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)
        dqn_agent.load(ckpt)

    steps_log: List[Dict] = []
    metrics_log: List[Dict] = []
    done = False

    while not done:
        if agent_type == "Heuristic":
            action = heuristic_action(obs)
        else:  # PyTorch DQN
            state = encode_observation(obs, task_ids)
            _, action = dqn_agent.greedy_action(state, task_ids)  # type: ignore[union-attr]

        next_obs, reward, done, info = env.step(action)

        steps_log.append({
            "Step":      obs.episode_step,
            "Day":       obs.current_day,
            "Action":    (
                f"{action.action_type} "
                f"{action.task_id or ''} "
                f"{'days='+str(action.days) if action.days else ''}"
            ).strip(),
            "Reward":    round(float(reward), 2),
            "Delay (d)": obs.metrics.delay_days,
            "Budget $":  f"{obs.metrics.budget_used:,.0f}",
            "Events":    " | ".join(info.get("events", [])),
        })
        metrics_log.append({
            "step":   obs.episode_step,
            "reward": float(reward),
            "delay":  obs.metrics.delay_days,
        })
        obs = next_obs

    final_state = env.state()
    grade_result = grade(task_level, final_state)

    passed_str = "PASS" if grade_result.passed else "FAIL"
    breakdown_md = "\n".join(
        f"- **{k}**: {v:.4f}" for k, v in grade_result.breakdown.items()
    )
    grade_md_out = (
        f"## Score: {grade_result.score:.4f} — {passed_str}\n\n"
        f"{grade_result.explanation}\n\n"
        f"### Breakdown\n{breakdown_md}"
    )

    gantt = build_gantt(obs)
    ep_fig = build_episode_fig(metrics_log)
    steps_df = pd.DataFrame(steps_log)
    task_df = obs_to_task_df(obs)

    return grade_md_out, steps_df, task_df, gantt, ep_fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Train DQN
# ─────────────────────────────────────────────────────────────────────────────

def start_training(task_level, num_episodes, lr, eps_decay, batch_size):
    if _training_state["running"]:
        return "Training already running. Click **Stop** first.", build_training_fig([], [])

    _training_state.update({
        "running":    True,
        "stop":       [False],
        "rewards":    [],
        "losses":     [],
        "grades":     [],
        "episode":    0,
        "total":      int(num_episodes),
        "task_level": task_level,
    })

    def _callback(ep, reward, epsilon, loss):
        _training_state["episode"] = ep
        _training_state["rewards"].append(reward)
        _training_state["losses"].append(loss)

    def _thread():
        run_training(
            task_level=task_level,
            num_episodes=int(num_episodes),
            lr=float(lr),
            epsilon_decay=float(eps_decay),
            batch_size=int(batch_size),
            progress_callback=_callback,
            stop_flag=_training_state["stop"],
        )
        _training_state["running"] = False

    threading.Thread(target=_thread, daemon=True).start()
    msg = (
        f"Training started: **{int(num_episodes)} episodes** on `{task_level}`  \n"
        f"Click **Refresh** below to see live progress."
    )
    return msg, build_training_fig([], [])


def stop_training():
    _training_state["stop"][0] = True
    return "Stop signal sent. Training will finish the current episode and save."


def poll_training():
    ep      = _training_state["episode"]
    total   = _training_state["total"]
    rewards = list(_training_state["rewards"])
    losses  = list(_training_state["losses"])
    running = _training_state["running"]

    if total > 0:
        pct   = ep / total * 100
        last_r = f"{rewards[-1]:.2f}" if rewards else "—"
        last_l = f"{losses[-1]:.4f}" if losses else "—"
        status = (
            f"{'Running' if running else 'Done'} | "
            f"Episode {ep}/{total} ({pct:.0f}%) | "
            f"Reward: {last_r} | Loss: {last_l}"
        )
    else:
        status = "Not started — configure parameters and click **Start Training**."

    return status, build_training_fig(rewards, losses)


def checkpoint_status(task_level):
    ckpt = DEFAULT_CHECKPOINT.format(level=task_level)
    if os.path.exists(ckpt):
        size_kb = os.path.getsize(ckpt) // 1024
        return f"Checkpoint found: `{ckpt}` ({size_kb} KB)"
    return f"No checkpoint yet for `{task_level}`. Train to create one."


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Manual Explorer
# ─────────────────────────────────────────────────────────────────────────────

def manual_reset(task_level, seed):
    global _manual_obs
    _manual_obs = _manual_env.reset(task_level=task_level, seed=int(seed))
    return (
        metrics_md(_manual_obs),
        obs_to_task_df(_manual_obs),
        obs_to_disruption_df(_manual_obs),
        build_gantt(_manual_obs),
        "",
    )


def manual_step(action_type, task_id, target_task_id, days):
    global _manual_obs
    if _manual_obs is None:
        return "Reset the environment first.", pd.DataFrame(), pd.DataFrame(), None, ""

    # Parse days safely
    try:
        days_int = int(days) if days and str(days).strip().isdigit() and int(days) >= 1 else None
    except (TypeError, ValueError):
        days_int = None

    action = Action(
        action_type=action_type,
        task_id=task_id.strip() if task_id and task_id.strip() else None,
        target_task_id=target_task_id.strip() if target_task_id and target_task_id.strip() else None,
        days=days_int,
    )

    try:
        next_obs, reward, done, info = _manual_env.step(action)
    except Exception as exc:
        return (
            f"Error: {exc}",
            obs_to_task_df(_manual_obs),
            obs_to_disruption_df(_manual_obs),
            build_gantt(_manual_obs),
            "",
        )

    _manual_obs = next_obs
    events_str = "\n".join(info.get("events", ["(no events)"])) or "(none)"
    log = f"**Step reward:** {reward:.2f}  |  **Done:** {done}\n\n**Events:**\n{events_str}"

    if done:
        final_state = _manual_env.state()
        gr_result = grade(_manual_obs.task_level, final_state)
        log += (
            f"\n\n---\n**EPISODE DONE** — "
            f"Score: {gr_result.score:.4f} "
            f"({'PASS' if gr_result.passed else 'FAIL'})\n\n"
            f"{gr_result.explanation}"
        )

    return (
        metrics_md(_manual_obs),
        obs_to_task_df(_manual_obs),
        obs_to_disruption_df(_manual_obs),
        build_gantt(_manual_obs),
        log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Construction Superintendent AI",
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    body { background: #0f0e17; }
    .gradio-container { background: #0f0e17 !important; }
    """,
) as demo:

    gr.Markdown(
        """# Construction Superintendent AI
Reinforcement learning environment for autonomous construction project management.
Compare the **Heuristic** rule-based agent vs a trained **PyTorch DQN** agent.

| Level  | Tasks | Disruptions | Budget    |
|--------|-------|-------------|-----------|
| Easy   | 5     | 1           | $90 000   |
| Medium | 8     | 3           | $250 000  |
| Hard   | 10    | 5           | $500 000  |
"""
    )

    # ── Tab 1: Run Episode ───────────────────────────────────────────────────
    with gr.Tab("Run Episode"):
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                task_level_ep = gr.Dropdown(
                    ["easy", "medium", "hard"], value="easy", label="Task Level"
                )
                seed_ep = gr.Number(value=42, label="Random Seed", precision=0)
                agent_type = gr.Radio(
                    ["Heuristic", "PyTorch DQN"],
                    value="Heuristic",
                    label="Agent",
                    info="Train the DQN first (Train tab) before selecting it here.",
                )
                run_btn = gr.Button("Run Episode", variant="primary")

            with gr.Column(scale=3):
                grade_out = gr.Markdown("*Configure and click Run Episode.*")

        with gr.Row():
            gantt_ep = gr.Plot(label="Final Gantt Chart")
            ep_fig   = gr.Plot(label="Reward & Delay per Step")

        steps_df_ep = gr.Dataframe(label="Step Log", wrap=True)
        task_df_ep  = gr.Dataframe(label="Final Task States")

        run_btn.click(
            run_episode_ui,
            inputs=[task_level_ep, seed_ep, agent_type],
            outputs=[grade_out, steps_df_ep, task_df_ep, gantt_ep, ep_fig],
        )

    # ── Tab 2: Train DQN ─────────────────────────────────────────────────────
    with gr.Tab("Train DQN"):
        gr.Markdown(
            """Train a **Double Dueling DQN** agent from scratch.
The checkpoint is auto-saved to `agent/checkpoints/dqn_<level>.pt` and used
by **Run Episode → PyTorch DQN**."""
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                task_level_tr  = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Level")
                num_eps_tr     = gr.Slider(50, 1000, value=300, step=50, label="Episodes")
                lr_tr          = gr.Number(value=1e-3, label="Learning Rate")
                eps_decay_tr   = gr.Slider(0.90, 0.999, value=0.99, step=0.001, label="Epsilon Decay")
                batch_tr       = gr.Slider(32, 256, value=64, step=32, label="Batch Size")
                ckpt_status_md = gr.Markdown("")

                with gr.Row():
                    train_btn = gr.Button("Start Training", variant="primary")
                    stop_btn  = gr.Button("Stop", variant="stop")
                refresh_btn = gr.Button("Refresh Progress")

            with gr.Column(scale=3):
                train_status = gr.Markdown("*Configure and click Start Training.*")
                train_fig    = gr.Plot(label="Training Curves")

        # Update checkpoint status when level changes
        task_level_tr.change(checkpoint_status, inputs=[task_level_tr], outputs=[ckpt_status_md])

        train_btn.click(
            start_training,
            inputs=[task_level_tr, num_eps_tr, lr_tr, eps_decay_tr, batch_tr],
            outputs=[train_status, train_fig],
        )
        stop_btn.click(stop_training, outputs=[train_status])
        refresh_btn.click(poll_training, outputs=[train_status, train_fig])

    # ── Tab 3: Manual Explorer ───────────────────────────────────────────────
    with gr.Tab("Manual Explorer"):
        gr.Markdown(
            "Step through the environment one action at a time to understand the dynamics."
        )

        with gr.Row():
            task_level_me = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Level")
            seed_me       = gr.Number(value=42, label="Seed", precision=0)
            reset_me_btn  = gr.Button("Reset Environment", variant="primary")

        metrics_me = gr.Markdown("*Reset to start.*")

        with gr.Row():
            gantt_me = gr.Plot(label="Gantt Chart")
            with gr.Column():
                task_df_me = gr.Dataframe(label="Tasks")
                dis_df_me  = gr.Dataframe(label="Active Disruptions")

        gr.Markdown("### Take Action")
        with gr.Row():
            at_me  = gr.Dropdown(
                ["noop", "expedite_task", "delay_task", "reassign_resources"],
                value="noop",
                label="Action Type",
            )
            tid_me    = gr.Textbox(label="Task ID", placeholder="e.g. T1")
            ttid_me   = gr.Textbox(label="Target Task ID (reassign only)", placeholder="e.g. T2")
            days_me   = gr.Number(value=1, label="Days (expedite/delay)", precision=0)
            step_me   = gr.Button("Step →", variant="primary")

        event_log_me = gr.Markdown("")

        reset_me_btn.click(
            manual_reset,
            inputs=[task_level_me, seed_me],
            outputs=[metrics_me, task_df_me, dis_df_me, gantt_me, event_log_me],
        )
        step_me.click(
            manual_step,
            inputs=[at_me, tid_me, ttid_me, days_me],
            outputs=[metrics_me, task_df_me, dis_df_me, gantt_me, event_log_me],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True,
    )
