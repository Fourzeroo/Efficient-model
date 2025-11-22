"""
Planner Agent - LLM-based decision maker for NAS

This module implements the Planner agent that uses an LLM with structured
output to decide what configuration changes to try next based on the history
of previous runs.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from nas_system.nas_agent import load_agent_state, get_best_run, get_recent_runs
from .prompts import PLANNER_PROMPT, format_recent_runs, format_config_section
from . import config as graph_config
from .phoenix_tracing import trace_operation, add_span_attribute, add_span_event


class PlannerDecision(BaseModel):
    """Structured output for the Planner agent."""
    step_type: str = Field(
        description='One of: "small", "medium", "radical", "stop"'
    )
    plan: str = Field(
        description="Human-readable plan for next change"
    )
    reason: str = Field(
        description="Why this plan makes sense given the history"
    )
    config_changes: Dict[str, Any] = Field(
        description='Concrete config changes, e.g. {"training.optimizer_params.lr": 1e-4}'
    )


def create_planner_llm():
    """Create and configure the Planner LLM."""
    return ChatOpenAI(
        base_url=graph_config.OPENROUTER_BASE_URL,
        api_key=graph_config.OPENROUTER_API_KEY,
        model=graph_config.PLANNER_MODEL,
        temperature=graph_config.PLANNER_TEMPERATURE,
    )


def load_current_config():
    """Load the current config.py as a dictionary."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "current_config", 
        graph_config.CONFIG_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {graph_config.CONFIG_PATH}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, 'get_config'):
        return config_module.get_config()
    else:
        return {
            "data": config_module.data,
            "model": config_module.model,
            "training": config_module.training,
            "logging": config_module.logging,
            "seed": config_module.seed,
        }


def plan_next_iteration(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the Planner LLM to decide what to try next.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with planner decision
    """
    with trace_operation("planner_agent", {
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"]
    }):
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        # Load agent state
        agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
        best_run = get_best_run(agent_state)
        recent_runs = get_recent_runs(agent_state, k=graph_config.RECENT_RUNS_WINDOW)
        
        add_span_event("agent_state_loaded", {
            "total_runs": len(agent_state.runs),
            "recent_runs_count": len(recent_runs)
        })
    
        # Prepare context
        if best_run:
            best_val_mse = best_run.val_mse
            best_run_id = best_run.run_id
            add_span_attribute("best_run_id", best_run_id)
            add_span_attribute("best_val_mse", f"{best_val_mse:.6f}")
        else:
            best_val_mse = float('inf')
            best_run_id = "None"
        
        recent_runs_summary = format_recent_runs(recent_runs)
        
        # Load current config
        current_config = load_current_config()
        config_text = format_config_section(current_config)
        
        # Format prompt
        prompt = PLANNER_PROMPT.format(
            iteration=iteration,
            max_iterations=max_iterations,
            best_val_mse=best_val_mse,
            best_run_id=best_run_id,
            recent_runs_summary=recent_runs_summary,
            current_config=config_text,
        )
        
        if graph_config.VERBOSE:
            print("\n" + "=" * 80)
            print(f"PLANNER - Iteration {iteration}/{max_iterations}")
            print("=" * 80)
            print(f"Best so far: {best_run_id} (val_mse={best_val_mse:.6f})")
            print(f"Recent runs: {len(recent_runs)}")
        
        # Create LLM with structured output
        llm = create_planner_llm()
        structured_llm = llm.with_structured_output(PlannerDecision)
        
        # Invoke LLM
        try:
            add_span_event("llm_invocation_start")
            messages = [HumanMessage(content=prompt)]
            decision = structured_llm.invoke(messages)
            
            # Add decision to trace
            add_span_attribute("step_type", decision.step_type)
            add_span_attribute("plan", decision.plan)
            add_span_attribute("num_config_changes", len(decision.config_changes))
            add_span_event("decision_made", {
                "step_type": decision.step_type,
                "changes": str(decision.config_changes)
            })
            
            if graph_config.VERBOSE:
                print(f"\nPlan: {decision.plan}")
                print(f"Step type: {decision.step_type}")
                print(f"Reason: {decision.reason}")
                print(f"Changes: {decision.config_changes}")
            
            # Update state
            state["step_type"] = decision.step_type
            state["plan"] = decision.plan
            state["reason"] = decision.reason
            state["config_changes"] = decision.config_changes
            
        except Exception as e:
            print(f"Error in Planner LLM: {e}")
            add_span_event("planner_error", {"error": str(e)})
            # Fallback to stop
            state["step_type"] = "stop"
            state["plan"] = "Error in planning"
            state["reason"] = str(e)
            state["config_changes"] = {}
        
        return state
