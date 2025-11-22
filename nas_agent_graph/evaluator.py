"""
Evaluator - Hybrid evaluation with deterministic rules and optional LLM

This module implements the evaluator that decides whether to accept or reject
a training run based on deterministic rules first, and falls back to LLM
evaluation for borderline cases.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from nas_system.nas_agent import (
    load_agent_state, get_best_run, add_run, save_agent_state, 
    RunInfo, build_history_summary
)
from .executor import load_run_results
from .prompts import EVALUATOR_PROMPT
from . import config as graph_config


class EvaluatorDecision(BaseModel):
    """Structured output for the Evaluator LLM."""
    accept: bool = Field(
        description="Whether to accept this run"
    )
    reason: str = Field(
        description="Justification for the decision"
    )
    feedback: str = Field(
        description="Suggestions for next iteration"
    )


def create_evaluator_llm():
    """Create and configure the Evaluator LLM."""
    return ChatOpenAI(
        base_url=graph_config.OPENROUTER_BASE_URL,
        api_key=graph_config.OPENROUTER_API_KEY,
        model=graph_config.EVALUATOR_MODEL,
        temperature=graph_config.EVALUATOR_TEMPERATURE,
    )


def evaluate_run_deterministic(
    new_metrics: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
    history: Dict[str, Any]
) -> Tuple[Optional[bool], str]:
    """
    Evaluate a run using deterministic rules.
    
    Args:
        new_metrics: Metrics from the new run
        best_metrics: Metrics from the best run so far (None if first run)
        history: Training history from the new run
        
    Returns:
        Tuple of (decision, reason) where:
            decision: True (accept), False (reject), or None (unclear, ask LLM)
            reason: Explanation for the decision
    """
    new_val = new_metrics.get("val_mse", float('inf'))
    new_train = new_metrics.get("train_mse", float('inf'))
    new_best_epoch = new_metrics.get("best_epoch", 0)
    
    # Handle first run
    if best_metrics is None:
        if new_val < 1.0:  # Reasonable threshold for first run
            return True, "First run with reasonable validation MSE"
        else:
            return False, "First run with poor validation MSE"
    
    best_val = best_metrics.get("val_mse", float('inf'))
    
    # Calculate improvement percentage
    improvement = (best_val - new_val) / best_val * 100  # positive = better
    
    # Rule 1: Clear improvement
    if improvement > 2.0:
        return True, f"Significant improvement: {improvement:.1f}%"
    
    # Rule 2: Clear regression
    if improvement < -0.5:
        return False, f"Performance degraded: {improvement:.1f}%"
    
    # Rule 3: Check for severe overfitting
    if new_train > 0 and new_val > 0:
        train_val_gap = abs(new_train - new_val) / new_val * 100
        if train_val_gap > 50:
            return False, f"Severe overfitting: train/val gap {train_val_gap:.1f}%"
    
    # Rule 4: Training converged too quickly (likely failed)
    if new_best_epoch < 3:
        return False, "Training converged too quickly or failed"
    
    # Rule 5: Borderline improvement
    if 0 < improvement <= 2.0:
        return None, f"Small improvement: {improvement:.1f}% (borderline)"
    
    # Rule 6: Minor regression
    if -0.5 <= improvement <= 0:
        return None, f"Minor regression: {improvement:.1f}% (borderline)"
    
    # Default: reject
    return False, "No clear improvement"


def evaluate_with_llm(
    new_run_id: str,
    new_metrics: Dict[str, Any],
    best_run_id: str,
    best_metrics: Dict[str, Any],
    history: Dict[str, Any],
    changes: Dict[str, Any]
) -> Tuple[bool, str, str]:
    """
    Evaluate a borderline run using the LLM.
    
    Args:
        new_run_id: ID of the new run
        new_metrics: Metrics from the new run
        best_run_id: ID of the best run
        best_metrics: Metrics from the best run
        history: Training history from the new run
        changes: Config changes that were applied
        
    Returns:
        Tuple of (accept, reason, feedback)
    """
    # Build history summary
    history_summary = build_history_summary(history)
    
    # Format changes
    changes_summary = "\n".join([f"- {k}: {v}" for k, v in changes.items()])
    if not changes_summary:
        changes_summary = "No changes (first run)"
    
    # Format prompt
    prompt = EVALUATOR_PROMPT.format(
        new_run_id=new_run_id,
        new_val_mse=new_metrics.get("val_mse", 0),
        new_test_mse=new_metrics.get("test_mse", 0),
        new_train_mse=new_metrics.get("train_mse", 0),
        new_best_epoch=new_metrics.get("best_epoch", 0),
        best_run_id=best_run_id,
        best_val_mse=best_metrics.get("val_mse", 0),
        best_test_mse=best_metrics.get("test_mse", 0),
        changes_summary=changes_summary,
        history_summary=history_summary,
    )
    
    if graph_config.VERBOSE:
        print("\nEvaluator LLM invoked for borderline case...")
    
    # Create LLM with structured output
    llm = create_evaluator_llm()
    structured_llm = llm.with_structured_output(EvaluatorDecision)
    
    try:
        messages = [HumanMessage(content=prompt)]
        decision = structured_llm.invoke(messages)
        
        return decision.accept, decision.reason, decision.feedback
        
    except Exception as e:
        print(f"Error in Evaluator LLM: {e}")
        # Conservative fallback: reject
        return False, f"LLM evaluation failed: {e}", "Try again with simpler changes"


def evaluate_run(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the training run and decide whether to accept it.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with evaluation decision
    """
    if not state["training_success"]:
        if graph_config.VERBOSE:
            print("\n" + "=" * 80)
            print("EVALUATOR - Training failed")
            print("=" * 80)
        
        state["accept_run"] = False
        state["evaluation_reason"] = "Training execution failed"
        state["feedback"] = "Fix training issues before proceeding"
        return state
    
    run_id = state["current_run_id"]
    
    if graph_config.VERBOSE:
        print("\n" + "=" * 80)
        print(f"EVALUATOR - Evaluating {run_id}")
        print("=" * 80)
    
    # Load run results
    try:
        metrics, history = load_run_results(run_id)
    except Exception as e:
        print(f"Error loading run results: {e}")
        state["accept_run"] = False
        state["evaluation_reason"] = f"Failed to load results: {e}"
        state["feedback"] = "Check run output files"
        return state
    
    # Load agent state to get best run
    agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
    best_run = get_best_run(agent_state)
    
    best_metrics = None
    if best_run:
        best_run_dir = graph_config.RUNS_ROOT / best_run.run_id
        try:
            from nas_system.nas_agent import load_metrics
            best_metrics = load_metrics(best_run_dir)
        except:
            pass
    
    # Deterministic evaluation
    decision, reason = evaluate_run_deterministic(metrics, best_metrics, history)
    
    if decision is not None:
        # Clear decision from rules
        if graph_config.VERBOSE:
            print(f"Deterministic decision: {'Accept' if decision else 'Reject'}")
            print(f"Reason: {reason}")
        
        state["accept_run"] = decision
        state["evaluation_reason"] = reason
        state["feedback"] = "Continue exploring" if decision else "Try different approach"
        
    else:
        # Borderline case - use LLM
        if graph_config.VERBOSE:
            print(f"Borderline case: {reason}")
            print("Consulting LLM evaluator...")
        
        best_run_id = best_run.run_id if best_run else "None"
        accept, llm_reason, feedback = evaluate_with_llm(
            run_id, metrics,
            best_run_id, best_metrics or {},
            history,
            state["config_changes"]
        )
        
        if graph_config.VERBOSE:
            print(f"LLM decision: {'Accept' if accept else 'Reject'}")
            print(f"Reason: {llm_reason}")
            print(f"Feedback: {feedback}")
        
        state["accept_run"] = accept
        state["evaluation_reason"] = f"[LLM] {llm_reason}"
        state["feedback"] = feedback
    
    return state
