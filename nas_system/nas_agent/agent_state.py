"""
Agent State - Manage global NAS state across multiple training runs.

This module provides dataclasses and utilities for tracking all training runs,
maintaining the best performing model, and enabling LLM agents to make informed
decisions about architecture search progress.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class RunInfo:
    """
    Information about a single training run.
    
    Attributes:
        run_id: Unique identifier for the run (e.g., "run_0000")
        val_mse: Validation MSE achieved by this run
        test_mse: Test MSE achieved by this run
        history_summary: Textual summary of learning dynamics
        accepted: Whether this run was accepted by the NAS agent
        config_changes: Dictionary of config changes applied for this run
    """
    run_id: str
    val_mse: float
    test_mse: float
    history_summary: str
    accepted: bool = True
    config_changes: Optional[dict] = None


@dataclass
class AgentState:
    """
    Global state of the NAS agent across all runs.
    
    Attributes:
        max_runs: Maximum number of runs to perform
        best_run_id: ID of the best performing run (lowest val_mse among accepted runs)
        runs: List of all completed runs
    """
    max_runs: int
    best_run_id: Optional[str]
    runs: List[RunInfo]


def load_agent_state(path: Path = Path("agent_state.json")) -> AgentState:
    """
    Load the global NAS agent state from a JSON file.
    
    If the file does not exist, returns a default empty state.
    
    Args:
        path: Path to the agent state JSON file
        
    Returns:
        AgentState object containing the global NAS state
    
    Example:
        >>> state = load_agent_state()
        >>> print(f"Completed {len(state.runs)} runs")
        Completed 5 runs
    """
    path = Path(path).resolve()
    
    if not path.exists():
        # Return default empty state
        return AgentState(
            max_runs=100,
            best_run_id=None,
            runs=[]
        )
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct RunInfo objects with backward compatibility
    runs = []
    for run_data in data.get("runs", []):
        # Handle old data without config_changes field
        if "config_changes" not in run_data:
            run_data["config_changes"] = None
        runs.append(RunInfo(**run_data))
    
    return AgentState(
        max_runs=data.get("max_runs", 100),
        best_run_id=data.get("best_run_id"),
        runs=runs
    )


def save_agent_state(state: AgentState, path: Path = Path("agent_state.json")) -> None:
    """
    Save the global NAS agent state to a JSON file.
    
    Args:
        state: AgentState object to save
        path: Path to the agent state JSON file
    
    Example:
        >>> state = AgentState(max_runs=100, best_run_id="run_0003", runs=[...])
        >>> save_agent_state(state)
    """
    path = Path(path).resolve()
    
    # Convert to dictionary
    data = {
        "max_runs": state.max_runs,
        "best_run_id": state.best_run_id,
        "runs": [asdict(run) for run in state.runs]
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_run(state: AgentState, run: RunInfo) -> AgentState:
    """
    Add a new run to the agent state and update best_run_id if necessary.
    
    Args:
        state: Current agent state
        run: RunInfo for the new run to add
        
    Returns:
        Updated AgentState with the new run added
    
    Example:
        >>> state = load_agent_state()
        >>> new_run = RunInfo(
        ...     run_id="run_0005",
        ...     val_mse=0.312,
        ...     test_mse=0.328,
        ...     history_summary="...",
        ...     accepted=True
        ... )
        >>> state = add_run(state, new_run)
        >>> save_agent_state(state)
    """
    # Add the run
    state.runs.append(run)
    
    # Update best_run_id if this run is better
    if run.accepted:
        current_best = get_best_run(state)
        if current_best is None or run.val_mse < current_best.val_mse:
            state.best_run_id = run.run_id
    
    return state


def get_best_run(state: AgentState) -> Optional[RunInfo]:
    """
    Get the best performing run (lowest val_mse among accepted runs).
    
    Args:
        state: Current agent state
        
    Returns:
        RunInfo for the best run, or None if no accepted runs exist
    
    Example:
        >>> state = load_agent_state()
        >>> best = get_best_run(state)
        >>> if best:
        ...     print(f"Best run: {best.run_id} with val_mse={best.val_mse}")
    """
    accepted_runs = [run for run in state.runs if run.accepted]
    
    if not accepted_runs:
        return None
    
    return min(accepted_runs, key=lambda r: r.val_mse)


def get_recent_runs(state: AgentState, k: int = 5) -> List[RunInfo]:
    """
    Get the k most recent runs.
    
    Args:
        state: Current agent state
        k: Number of recent runs to return
        
    Returns:
        List of the k most recent RunInfo objects (or fewer if less than k exist)
    
    Example:
        >>> state = load_agent_state()
        >>> recent = get_recent_runs(state, k=3)
        >>> for run in recent:
        ...     print(f"{run.run_id}: val_mse={run.val_mse}")
    """
    return state.runs[-k:] if len(state.runs) >= k else state.runs
