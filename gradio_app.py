"""
Interactive Gradio Web UI for the Construction Superintendent OpenEnv.
Allows human players to interact with the environment, visualize the DAG,
and attempt the scenarios manually.
"""

import gradio as gr
import pandas as pd

from env.construction_env import ConstructionEnv
from env.models import Action, ActionType

# =============================================================================
# Global State
# =============================================================================

# We use a single global environment instance for the local UI
env = ConstructionEnv()
current_obs = None

# =============================================================================
# Helper Formatting Functions
# =============================================================================

def format_tasks(tasks):
    """Convert tasks into a pandas DataFrame for the Gradio Dataframe component."""
    if not tasks:
        return pd.DataFrame()
    
    data = []
    for t in tasks:
        data.append({
            "ID": t.id,
            "Name": t.name,
            "Status": t.status.value.replace("_", " ").title(),
            "Start-End": f"Day {t.current_start_day} - {t.current_end_day}",
            "Delay": f"+{t.delay_from_original}d",
            "Resources": t.resources,
            "Critical Path": "⭐ Yes" if t.is_on_critical_path else "No"
        })
    return pd.DataFrame(data)

def format_disruptions(disruptions):
    """Convert active disruptions into a pandas DataFrame."""
    if not disruptions:
        return pd.DataFrame(columns=["ID", "Type", "Affected Task", "Delay Left", "Description"])
        
    data = []
    for d in disruptions:
        data.append({
            "ID": d.id,
            "Type": d.type.value.replace("_", " ").title(),
            "Affected Task": d.affected_task_id,
            "Delay Left": f"{d.remaining_delay_days} days",
            "Description": d.description
        })
    return pd.DataFrame(data)

def format_metrics(metrics, current_day, step):
    """Format the top-level metrics as Markdown."""
    budget_color = "green" if metrics.budget_used <= metrics.budget_total else "red"
    delay_color = "red" if metrics.delay_days > 0 else "green"
    
    return f"""
### Clock: Day {current_day} | Step: {step}
* **Projected End:** Day {metrics.current_projected_end_day} (Original: {metrics.original_end_day})
* **Total Delay:** <span style="color: {delay_color};">**{metrics.delay_days} days**</span>
* **Budget Used:** <span style="color: {budget_color};">**${metrics.budget_used:,.0f}**</span> / ${metrics.budget_total:,.0f}
* **Tasks Done:** {metrics.tasks_completed} / {metrics.tasks_total}
* **Disruptions Resolved:** {metrics.disruptions_resolved}
"""

# =============================================================================
# UI Action Handlers
# =============================================================================

def reset_env(task_level):
    global current_obs
    current_obs = env.reset(task_level=task_level)
    return update_ui_components("Environment reset successfully. Simulation at Day 0.")

def step_env(action_type, task_id, target_task_id, days):
    global current_obs
    
    if env._done:
        return update_ui_components("⚠️ Episode is already finished. Please reset.")
        
    try:
        # Build action object, handling empty strings from Gradio textboxes
        action = Action(
            action_type=action_type,
            task_id=task_id if task_id.strip() else None,
            target_task_id=target_task_id if target_task_id.strip() else None,
            days=int(days) if days else None
        )
        
        current_obs, reward, done, info = env.step(action)
        
        msg = f"**Step applied!** Reward: {reward:.2f}\n\n*Log: {info.get('reward_detail', '')}*"
        
        for event in info.get("events", []):
            msg += f"\n* {event}"
            
        if done:
            final_score = env.state()
            msg += f"\n\n### 🎉 EPISODE COMPLETE! Check terminal for grading."
            
        return update_ui_components(msg)
        
    except Exception as e:
        return update_ui_components(f"❌ Error: {str(e)}")

def update_ui_components(status_msg=""):
    """Returns the updated values for all UI components based on current_obs."""
    if current_obs is None:
        return (
            pd.DataFrame(), 
            "Please reset the environment.", 
            status_msg, 
            pd.DataFrame()
        )
        
    tasks_df = format_tasks(current_obs.tasks)
    metrics_md = format_metrics(current_obs.metrics, current_obs.current_day, current_obs.episode_step)
    disruptions_df = format_disruptions(current_obs.active_disruptions)
    
    return tasks_df, metrics_md, status_msg, disruptions_df

# =============================================================================
# Gradio Layout Definitions
# =============================================================================

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(title="Construction AI Env") as demo:
    gr.Markdown("# 🏗️ Construction Superintendent - OpenEnv")
    gr.Markdown("Play the environment manually to test the DAG logic and disruption events.")
    
    with gr.Row():
        level_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"], 
            value="easy", 
            label="Select Difficulty Level",
            interactive=True
        )
        reset_btn = gr.Button("🔄 Reset / Start New Episode", variant="primary")
        
    with gr.Row():
        # Left Column - Metrics and Disruptions
        with gr.Column(scale=2):
            gr.Markdown("## Dashboard")
            metrics_display = gr.Markdown("Please reset the environment to start.")
            
            gr.Markdown("### ⚠️ Active Disruptions")
            disruptions_display = gr.Dataframe(
                headers=["ID", "Type", "Affected Task", "Delay Left", "Description"],
                interactive=False
            )
            
            gr.Markdown("### 📋 System Logs")
            status_display = gr.Markdown("*Waiting for action...*")
            
        # Right Column - Controls
        with gr.Column(scale=1):
            gr.Markdown("## Take Action")
            gr.Markdown("Submit one scheduling decision for the current step.")
            
            action_dropdown = gr.Dropdown(
                choices=[a.value for a in ActionType],
                value=ActionType.NOOP.value,
                label="Action Type"
            )
            
            task_input = gr.Textbox(
                label="Primary Task ID", 
                placeholder="e.g., T1 (required for expedite/delay/reassign)"
            )
            
            target_input = gr.Textbox(
                label="Target Task ID", 
                placeholder="e.g., T5 (only for reassign_resources)"
            )
            
            days_input = gr.Number(
                value=1, 
                label="Days / Extra Resources",
                precision=0
            )
            
            step_btn = gr.Button("▶️ Submit Action & Step Simulation", variant="secondary")

    gr.Markdown("## Task Schedule (DAG Status)")
    task_display = gr.Dataframe(
        headers=["ID", "Name", "Status", "Start-End", "Delay", "Resources", "Critical Path"],
        interactive=False,
        wrap=True
    )

    # Wire up events
    reset_btn.click(
        fn=reset_env,
        inputs=[level_dropdown],
        outputs=[task_display, metrics_display, status_display, disruptions_display]
    )
    
    step_btn.click(
        fn=step_env,
        inputs=[action_dropdown, task_input, target_input, days_input],
        outputs=[task_display, metrics_display, status_display, disruptions_display]
    )

if __name__ == "__main__":
    print("Starting Gradio server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme)