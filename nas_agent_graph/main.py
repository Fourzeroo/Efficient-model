"""
Main Entry Point - LLM-Orchestrated NAS System

This module provides the main entry point for running the Neural Architecture
Search optimization loop using LangGraph and LLM agents.

Usage:
    from nas_agent_graph import run_nas_optimization
    
    run_nas_optimization(max_iterations=20)
    
Or from command line:
    python -m nas_agent_graph.main --max-iterations 20
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nas_system.nas_agent import load_agent_state, get_best_run
from .graph import build_nas_graph, NASGraphState
from . import config as graph_config
from .phoenix_tracing import setup_phoenix_tracing


def run_nas_optimization(max_iterations: int = 20) -> None:
    """
    Run the LLM-orchestrated Neural Architecture Search optimization.
    
    This function initializes the LangGraph workflow and runs it for up to
    max_iterations, using the Planner LLM to decide configuration changes,
    the Executor to apply them and run training, and the Evaluator to
    assess results.
    
    Args:
        max_iterations: Maximum number of NAS iterations to run
        
    Example:
        >>> from nas_agent_graph import run_nas_optimization
        >>> run_nas_optimization(max_iterations=20)
    """
    print("=" * 80)
    print("LLM-ORCHESTRATED NEURAL ARCHITECTURE SEARCH")
    print("=" * 80)
    print(f"Max iterations: {max_iterations}")
    print(f"Using model: {graph_config.PLANNER_MODEL}")
    print(f"NAS system root: {graph_config.NAS_SYSTEM_ROOT}")
    
    # Initialize Phoenix tracing
    phoenix_enabled = setup_phoenix_tracing()
    if phoenix_enabled:
        print(f"Phoenix tracing: http://{graph_config.PHOENIX_HOST}:{graph_config.PHOENIX_PORT}")
    
    print("=" * 80)
    
    # Load existing agent state to determine starting iteration
    agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
    start_iteration = len(agent_state.runs)
    
    if start_iteration > 0:
        print(f"\nResuming from iteration {start_iteration}")
        best_run = get_best_run(agent_state)
        if best_run:
            print(f"Current best: {best_run.run_id} (val_mse={best_run.val_mse:.6f})")
    else:
        print("\nStarting fresh NAS session")
    
    # Build the graph
    print("\nBuilding LangGraph workflow...")
    graph = build_nas_graph()
    
    # Initialize state
    initial_state: NASGraphState = {
        "iteration": start_iteration,
        "max_iterations": max_iterations,
        "step_type": "small",
        "plan": "",
        "reason": "",
        "config_changes": {},
        "current_run_id": "",
        "training_success": True,
        "accept_run": False,
        "evaluation_reason": "",
        "feedback": "",
    }
    
    print("\nStarting NAS optimization loop...")
    print("=" * 80)
    
    # Run the graph
    try:
        final_state = graph.invoke(initial_state)
        
        print("\n" + "=" * 80)
        print("NAS OPTIMIZATION COMPLETE")
        print("=" * 80)
        
        # Load final agent state and display results
        agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
        best_run = get_best_run(agent_state)
        
        if best_run:
            print(f"\nBest run: {best_run.run_id}")
            print(f"  Validation MSE: {best_run.val_mse:.6f}")
            print(f"  Test MSE: {best_run.test_mse:.6f}")
            print(f"\nHistory summary:")
            print(f"  {best_run.history_summary}")
        else:
            print("\nNo successful runs found.")
        
        print(f"\nTotal runs completed: {len(agent_state.runs)}")
        print(f"Accepted runs: {sum(1 for r in agent_state.runs if r.accepted)}")
        print(f"Rejected runs: {sum(1 for r in agent_state.runs if not r.accepted)}")
        
        print("\n" + "=" * 80)
        print("Results saved to:")
        print(f"  - Agent state: {graph_config.AGENT_STATE_PATH}")
        print(f"  - Run outputs: {graph_config.RUNS_ROOT}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        agent_state = load_agent_state(graph_config.AGENT_STATE_PATH)
        print(f"Completed {len(agent_state.runs)} runs before interruption")
        
    except Exception as e:
        print(f"\n\nError during NAS optimization: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM-orchestrated Neural Architecture Search"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=graph_config.DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of NAS iterations (default: {graph_config.DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=graph_config.VERBOSE,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=graph_config.DEBUG,
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Update config
    graph_config.VERBOSE = args.verbose
    graph_config.DEBUG = args.debug
    
    # Run optimization
    run_nas_optimization(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
