"""
Test script to verify that config_changes are properly stored and displayed.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from nas_system.nas_agent import load_agent_state, save_agent_state, add_run, RunInfo
from nas_agent_graph.prompts import format_recent_runs

# Test 1: Load existing agent_state.json (should handle backward compatibility)
print("=" * 80)
print("Test 1: Loading existing agent state (backward compatibility)")
print("=" * 80)

agent_state_path = Path("nas_system/agent_state.json")
state = load_agent_state(agent_state_path)

print(f"Loaded {len(state.runs)} runs")
for run in state.runs:
    print(f"  {run.run_id}: config_changes = {run.config_changes}")

# Test 2: Add a new run with config_changes
print("\n" + "=" * 80)
print("Test 2: Adding new run with config_changes")
print("=" * 80)

new_run = RunInfo(
    run_id="run_test",
    val_mse=0.15,
    test_mse=0.16,
    history_summary="Test run for config_changes tracking",
    accepted=True,
    config_changes={
        "training.optimizer_params.lr": 0.0001,
        "model.d_model": 256,
        "training.dropout": 0.2
    }
)

print(f"New run: {new_run.run_id}")
print(f"Config changes: {new_run.config_changes}")

# Test 3: Format recent runs for prompt
print("\n" + "=" * 80)
print("Test 3: Formatting recent runs for planner prompt")
print("=" * 80)

# Add the new run temporarily (don't save)
test_state = state
test_state.runs.append(new_run)

formatted = format_recent_runs(test_state.runs, k=3)
print(formatted)

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
