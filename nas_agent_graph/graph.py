"""
LangGraph Workflow - State machine for NAS orchestration

This module defines the state graph that orchestrates the NAS process,
connecting the Planner, Executor, and Evaluator in a loop.
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from nas_system.nas_agent import (
    load_agent_state, save_agent_state, add_run, 
    RunInfo, build_history_summary
)
from .planner import plan_next_iteration
from .executor import execute_training, load_run_results
from .evaluator import evaluate_run
from . import config as graph_config


class NASGraphState(TypedDict):
    """State for the NAS graph workflow."""
    # Iteration tracking
    iteration: int
    max_iterations: int
    
    # Planner outputs
    step_type: Literal["small", "medium", "radical", "stop"]
    plan: str
    reason: str
    config_changes: Dict[str, Any]
    
    # Executor outputs
    current_run_id: str
    training_success: bool
    
    # Evaluator outputs
    accept_run: bool
    evaluation_reason: str
    feedback: str


def planner_node(state: NASGraphState) -> NASGraphState:
    """
    Node for the Planner agent.
    
    Invokes the Planner LLM to decide what configuration changes to try next.
    """
    if graph_config.VERBOSE:
        print("\n" + "=" * 80)
        print("PLANNER NODE")
        print("=" * 80)
    
    return plan_next_iteration(state)


def executor_node(state: NASGraphState) -> NASGraphState:
    """
    Node for the Executor.
    
    Applies config changes and runs the training script.
    """
    if graph_config.VERBOSE:
        print("\n" + "=" * 80)
        print("EXECUTOR NODE")
        print("=" * 80)
    
    return execute_training(state)


def evaluator_node(state: NASGraphState) -> NASGraphState:
    """
    Node for the Evaluator.
    
    Evaluates the training run using deterministic rules and optional LLM.
    """
    if graph_config.VERBOSE:
        print("\n" + "=" * 80)
        print("EVALUATOR NODE")
        print("=" * 80)
    
    return evaluate_run(state)


def update_state_node(state: NASGraphState) -> NASGraphState:
    """
    Node to update global agent state.
    
    Saves the run to agent_state.json and increments the iteration counter.
    """
    if graph_config.VERBOSE:
        print("\n" + "=" * 80)
        print("UPDATE STATE NODE")
        print("=" * 80)
    
    # Load agent state
    agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
    
    # Add run if training was successful
    if state["training_success"]:
        try:
            # Load run results
            metrics, history = load_run_results(state["current_run_id"])
            
            # Build history summary
            history_summary = build_history_summary(history)
            
            # Create RunInfo
            run_info = RunInfo(
                run_id=state["current_run_id"],
                val_mse=metrics["val_mse"],
                test_mse=metrics["test_mse"],
                history_summary=history_summary,
                accepted=state["accept_run"],
                config_changes=state.get("config_changes", {})
            )
            
            # Add to state
            agent_state = add_run(agent_state, run_info)
            
            if graph_config.VERBOSE:
                print(f"Added run {run_info.run_id} to agent state")
                print(f"  Val MSE: {run_info.val_mse:.6f}")
                print(f"  Test MSE: {run_info.test_mse:.6f}")
                print(f"  Accepted: {run_info.accepted}")
        
        except Exception as e:
            print(f"Error updating agent state: {e}")
    
    # Save agent state
    save_agent_state(agent_state, graph_config.AGENT_STATE_PATH)
    
    # Increment iteration
    state["iteration"] += 1
    
    if graph_config.VERBOSE:
        print(f"Iteration incremented to {state['iteration']}")
    
    return state


def should_continue(state: NASGraphState) -> Literal["continue", "stop"]:
    """
    Decide whether to continue the NAS loop or stop.
    
    Args:
        state: Current graph state
        
    Returns:
        "continue" or "stop"
    """
    # Stop if planner says stop
    if state["step_type"] == "stop":
        if graph_config.VERBOSE:
            print("\nStopping: Planner decided to stop")
        return "stop"
    
    # Stop if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        if graph_config.VERBOSE:
            print(f"\nStopping: Max iterations ({state['max_iterations']}) reached")
        return "stop"
    
    return "continue"


def build_nas_graph() -> Any:
    """
    Build the LangGraph workflow for NAS.
    
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(NASGraphState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("update_state", update_state_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add sequential edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    workflow.add_edge("evaluator", "update_state")
    
    # Add conditional edge for loop
    workflow.add_conditional_edges(
        "update_state",
        should_continue,
        {
            "continue": "planner",
            "stop": END
        }
    )
    
    return workflow.compile()
