"""
Prompt Templates for LLM Agents

This module defines the prompt templates used by the Planner and Evaluator
LLM agents in the NAS orchestration system.
"""

PLANNER_PROMPT = """You are a Neural Architecture Search expert optimizing the Informer model for time series forecasting.

Current State:
- Iteration: {iteration}/{max_iterations}
- Best validation MSE: {best_val_mse:.6f}
- Best run ID: {best_run_id}

Recent Runs History:
{recent_runs_summary}

NOTE: For each run above, you can see:
1. Metrics (val_mse, test_mse) and acceptance status
2. **Config changes that were applied** - this shows what was modified from the previous run
3. Training dynamics summary (overfitting signals, convergence behavior)

Current Config (Python module config.py):
```python
{current_config}
```

Task:
Analyze the history INCLUDING the config changes that were tried and their outcomes. Consider:
- Learning dynamics (overfitting, underfitting, instability)
- Model capacity (too small or too large)
- Training hyperparameters (learning rate too high/low, batch size)
- Optimization and scheduler behavior
- Regularization (dropout, weight decay)
- **What changes have already been tried and whether they helped or hurt performance**

Choose step_type:
- "small": Fine-tune hyperparameters (small LR changes, batch size tweaks, dropout adjustments, weight decay).
- "medium": Architecture-level changes (d_model, n_heads, e_layers, d_ff, factor).
- "radical": Major changes (optimizer/scheduler switch, strong regularization, loss function change).
- "stop": If val_mse is already very good (< 0.15) or max_iterations is reached or no more improvements possible.

Fill config_changes with concrete nested key updates, for example:
- {{"training.optimizer_params.lr": 0.0001}}
- {{"model.d_model": 256, "model.e_layers": 3}}
- {{"training.dropout": 0.3}}

IMPORTANT RULES:
1. Only propose changes you can express via config_changes (nested dictionary keys).
2. For optimizer changes, you can change "optimizer_class" to "Adam", "AdamW", "SGD", etc.
3. For scheduler changes, you can change "scheduler_class" to "CosineAnnealingLR", "StepLR", "ReduceLROnPlateau", etc.
4. For loss function changes, you can change "criterion_class" to "MSELoss", "L1Loss", "SmoothL1Loss", etc.
5. Be specific about the values - use actual numbers, not placeholders.
6. Consider the recent history to avoid repeating failed experiments.

Provide your decision with:
- step_type: one of ["small", "medium", "radical", "stop"]
- plan: brief description of what you're trying
- reason: why this makes sense given the history
- config_changes: concrete dictionary of changes to apply
"""


EVALUATOR_PROMPT = """You are evaluating a borderline training run in a Neural Architecture Search system.

New Run ({new_run_id}):
- val_mse: {new_val_mse:.6f}
- test_mse: {new_test_mse:.6f}
- train_mse: {new_train_mse:.6f}
- best_epoch: {new_best_epoch}

Best Run So Far ({best_run_id}):
- val_mse: {best_val_mse:.6f}
- test_mse: {best_test_mse:.6f}

Changes Made:
{changes_summary}

History Summary:
{history_summary}

Context:
Automatic rules found this run to be borderline. The validation MSE improvement is small 
(between -0.5% and +2.0%), or there are mixed signals in the training dynamics.

Your Task:
Decide whether to accept this run as a potential improvement:

ACCEPT if:
- The improvement, even small, seems meaningful and stable
- Training dynamics are healthy (no severe overfitting, reasonable convergence)
- The change represents progress in the right direction

REJECT if:
- Training dynamics look unhealthy (overfitting, instability, weird patterns)
- The model is clearly worse despite similar metrics
- Early stopping triggered too early (< 3 epochs)

Provide feedback on what to try in the next iteration (specific suggestions for hyperparameters or architecture).

Make your decision based on:
1. Validation and test MSE comparison
2. Training dynamics and overfitting signals
3. Convergence behavior (best_epoch)
4. Overall trajectory of the NAS process
"""


def format_recent_runs(runs, k=5):
    """
    Format recent runs for inclusion in the planner prompt.
    
    Args:
        runs: List of RunInfo objects
        k: Number of recent runs to include
        
    Returns:
        Formatted string with recent run summaries
    """
    if not runs:
        return "No previous runs available."
    
    recent = runs[-k:] if len(runs) >= k else runs
    
    lines = []
    for run in recent:
        status = "✓ Accepted" if run.accepted else "✗ Rejected"
        lines.append(
            f"- {run.run_id}: val_mse={run.val_mse:.6f}, test_mse={run.test_mse:.6f} ({status})"
        )
        
        # Show config changes if available
        if run.config_changes:
            import json
            changes_str = json.dumps(run.config_changes, indent=4)
            lines.append(f"  Changes applied: {changes_str}")
        else:
            lines.append("  Changes applied: (not recorded)")
        
        lines.append(f"  Training summary: {run.history_summary[:200]}...")  # Truncate for brevity
        lines.append("")  # Empty line for readability
    
    return "\n".join(lines)


def format_config_section(config_dict):
    """
    Format relevant config sections for the planner prompt.
    
    Args:
        config_dict: The full config dictionary
        
    Returns:
        Formatted string with training and model config
    """
    import json
    
    # Extract relevant sections
    training = config_dict.get("training", {})
    model = config_dict.get("model", {})
    
    # Format nicely
    sections = []
    
    sections.append("# Training Configuration")
    sections.append("training = {")
    for key, value in training.items():
        # Handle class objects
        if hasattr(value, '__name__'):
            sections.append(f'    "{key}": {value.__name__},')
        else:
            sections.append(f'    "{key}": {json.dumps(value)},')
    sections.append("}")
    
    sections.append("\n# Model Configuration")
    sections.append("model = {")
    for key, value in model.items():
        sections.append(f'    "{key}": {json.dumps(value)},')
    sections.append("}")
    
    return "\n".join(sections)
