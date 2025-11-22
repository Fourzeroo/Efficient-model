"""
Executor - Apply config changes and run training

This module implements the executor that modifies the config.py file
based on the Planner's decisions and runs the training script.
"""

import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nas_system.nas_agent import load_metrics, load_history
from . import config as graph_config
from .phoenix_tracing import trace_operation, add_span_attribute, add_span_event


def apply_config_changes(config_path: Path, changes: Dict[str, Any]) -> None:
    """
    Apply configuration changes to the config.py file.
    
    This function modifies the Python config file by updating nested dictionary
    values based on dot-notation keys (e.g., "training.optimizer_params.lr").
    
    Args:
        config_path: Path to the config.py file
        changes: Dictionary of changes with dot-notation keys
        
    Example:
        changes = {
            "training.optimizer_params.lr": 0.0001,
            "model.d_model": 256,
            "training.optimizer_class": "AdamW"
        }
    """
    if not changes:
        return
    
    # Read the current config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply each change
    for key, value in changes.items():
        parts = key.split('.')
        
        if len(parts) == 2:
            # Top-level dict change (e.g., "model.d_model")
            dict_name = parts[0]
            field_name = parts[1]
            
            # Handle special class changes
            if field_name in ["optimizer_class", "scheduler_class", "criterion_class"]:
                content = _replace_class_assignment(content, dict_name, field_name, value)
            else:
                content = _replace_dict_field(content, dict_name, field_name, value)
                
        elif len(parts) == 3:
            # Nested dict change (e.g., "training.optimizer_params.lr")
            dict_name = parts[0]
            nested_dict = parts[1]
            field_name = parts[2]
            content = _replace_nested_dict_field(content, dict_name, nested_dict, field_name, value)
    
    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _replace_dict_field(content: str, dict_name: str, field_name: str, value: Any) -> str:
    """Replace a field in a top-level dictionary."""
    # Pattern to match: "field_name": <any_value>,
    pattern = rf'("{field_name}":\s*)([^,\n]+)(,?)'
    
    # Format the new value
    if isinstance(value, str) and not value.startswith('"'):
        new_value = f'"{value}"'
    elif isinstance(value, bool):
        new_value = str(value)
    elif isinstance(value, (int, float)):
        new_value = str(value)
    else:
        new_value = str(value)
    
    replacement = rf'\g<1>{new_value}\g<3>'
    
    # Find the dictionary block
    dict_pattern = rf'{dict_name}\s*=\s*\{{([^}}]+)}}'
    
    def replacer(match):
        dict_content = match.group(1)
        updated_content = re.sub(pattern, replacement, dict_content)
        return f'{dict_name} = {{{updated_content}}}'
    
    return re.sub(dict_pattern, replacer, content, flags=re.DOTALL)


def _replace_nested_dict_field(content: str, dict_name: str, nested_dict: str, 
                                field_name: str, value: Any) -> str:
    """Replace a field in a nested dictionary."""
    # Pattern to match nested dict section
    nested_pattern = rf'("{nested_dict}":\s*\{{)([^}}]+)(\}})'
    
    # Format the new value
    if isinstance(value, str) and not value.startswith('"'):
        new_value = f'"{value}"'
    elif isinstance(value, bool):
        new_value = str(value)
    elif isinstance(value, (int, float)):
        new_value = str(value)
    else:
        new_value = str(value)
    
    def replacer(match):
        prefix = match.group(1)
        nested_content = match.group(2)
        suffix = match.group(3)
        
        # Replace the field in nested content
        field_pattern = rf'("{field_name}":\s*)([^,\n]+)(,?)'
        field_replacement = rf'\g<1>{new_value}\g<3>'
        updated_nested = re.sub(field_pattern, field_replacement, nested_content)
        
        return f'{prefix}{updated_nested}{suffix}'
    
    # Find the parent dictionary block
    dict_pattern = rf'{dict_name}\s*=\s*\{{(.+?)\n\}}'
    
    def dict_replacer(match):
        dict_content = match.group(1)
        updated_content = re.sub(nested_pattern, replacer, dict_content, flags=re.DOTALL)
        return f'{dict_name} = {{{updated_content}\n}}'
    
    return re.sub(dict_pattern, dict_replacer, content, flags=re.DOTALL)


def _replace_class_assignment(content: str, dict_name: str, field_name: str, value: str) -> str:
    """
    Replace a class assignment (e.g., optimizer_class, scheduler_class).
    
    Args:
        content: File content
        dict_name: Dictionary name (e.g., "training")
        field_name: Field name (e.g., "optimizer_class")
        value: Class name as string (e.g., "AdamW", "SGD", "StepLR")
    """
    # Map string names to actual import paths
    class_mappings = {
        # Optimizers
        "Adam": "optim.Adam",
        "AdamW": "optim.AdamW",
        "SGD": "optim.SGD",
        "RMSprop": "optim.RMSprop",
        # Schedulers
        "CosineAnnealingLR": "optim.lr_scheduler.CosineAnnealingLR",
        "StepLR": "optim.lr_scheduler.StepLR",
        "ReduceLROnPlateau": "optim.lr_scheduler.ReduceLROnPlateau",
        "ExponentialLR": "optim.lr_scheduler.ExponentialLR",
        # Loss functions
        "MSELoss": "nn.MSELoss",
        "L1Loss": "nn.L1Loss",
        "SmoothL1Loss": "nn.SmoothL1Loss",
    }
    
    class_ref = class_mappings.get(value, value)
    
    # Pattern to match the field in the dictionary
    pattern = rf'("{field_name}":\s*)([^,\n]+)(,?)'
    replacement = rf'\g<1>{class_ref}\g<3>'
    
    # Find the dictionary block
    dict_pattern = rf'{dict_name}\s*=\s*\{{(.+?)\n\}}'
    
    def replacer(match):
        dict_content = match.group(1)
        updated_content = re.sub(pattern, replacement, dict_content, flags=re.DOTALL)
        return f'{dict_name} = {{{updated_content}\n}}'
    
    return re.sub(dict_pattern, replacer, content, flags=re.DOTALL)


def run_training(config_path: Path, tag: str) -> str:
    """
    Run the training script.
    
    Args:
        config_path: Path to the config file
        tag: Run tag (e.g., "run_0001")
        
    Returns:
        The run tag
        
    Raises:
        RuntimeError: If training fails
    """
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--config", config_path.name,
        "--tag", tag
    ]
    
    if graph_config.VERBOSE:
        print(f"\nRunning training: {' '.join(cmd)}")
    
    # Run in nas_system directory
    result = subprocess.run(
        cmd,
        cwd=str(graph_config.NAS_SYSTEM_ROOT),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        error_msg = f"Training failed with exit code {result.returncode}\n"
        error_msg += f"STDERR:\n{result.stderr}\n"
        error_msg += f"STDOUT:\n{result.stdout}"
        raise RuntimeError(error_msg)
    
    if graph_config.VERBOSE:
        print(f"Training completed successfully for {tag}")
    
    return tag


def load_run_results(run_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load metrics and history for a completed run.
    
    Args:
        run_id: Run identifier (e.g., "run_0001")
        
    Returns:
        Tuple of (metrics, history) dictionaries
    """
    run_dir = graph_config.RUNS_ROOT / run_id
    metrics = load_metrics(run_dir)
    history = load_history(run_dir)
    return metrics, history


def execute_training(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the training based on planner decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with execution results
    """
    with trace_operation("executor", {
        "iteration": state["iteration"],
        "step_type": state["step_type"]
    }):
        step_type = state["step_type"]
        
        # Check if we should stop
        if step_type == "stop":
            if graph_config.VERBOSE:
                print("\nPlanner decided to stop. No execution needed.")
            add_span_event("execution_skipped", {"reason": "planner_stop"})
            state["training_success"] = False
            return state
        
        config_changes = state["config_changes"]
        iteration = state["iteration"]
        
        # Generate run ID
        run_id = f"run_{iteration:04d}"
        add_span_attribute("run_id", run_id)
        add_span_attribute("num_changes", len(config_changes))
        
        if graph_config.VERBOSE:
            print("\n" + "=" * 80)
            print(f"EXECUTOR - Running iteration {iteration}")
            print("=" * 80)
            print(f"Applying {len(config_changes)} config changes...")
        
        try:
            # Apply config changes
            with trace_operation("apply_config_changes", {"changes": config_changes}):
                apply_config_changes(graph_config.CONFIG_PATH, config_changes)
            
            if graph_config.VERBOSE:
                print("Config changes applied successfully")
            
            add_span_event("config_modified")
            
            # Run training
            with trace_operation("training_run", {"run_id": run_id}):
                run_training(graph_config.CONFIG_PATH, run_id)
            
            add_span_event("training_completed", {"success": True})
            state["current_run_id"] = run_id
            state["training_success"] = True
            
        except Exception as e:
            print(f"\nError during execution: {e}")
            add_span_event("execution_error", {"error": str(e)})
            state["current_run_id"] = run_id
            state["training_success"] = False
        
        return state
