"""
Example LLM Agent Interaction Script

This script demonstrates how an external LLM agent would interact with
the NAS system to perform architecture search via filesystem and shell commands.

This is a SIMULATION of what an LLM agent would do. In practice, the LLM
would be running in a separate environment (Claude, Cursor, Windsurf, etc.)
and would execute these operations based on its reasoning.
"""

import json
import subprocess
import sys
from pathlib import Path

# Import the nas_agent library
sys.path.insert(0, str(Path(__file__).parent))
from nas_agent import (
    load_agent_state,
    save_agent_state,
    add_run,
    get_best_run,
    get_recent_runs,
    load_history,
    load_metrics,
    build_history_summary,
    RunInfo,
)


def read_config(config_path: Path) -> dict:
    """Read config.py file."""
    import importlib.util
    import sys
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_read"] = config_module
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


def write_config(config_path: Path, config: dict):
    """Write config.py file."""
    # Read original file to preserve imports and structure
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find where each section starts
    import_end = 0
    data_start = 0
    model_start = 0
    training_start = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
        elif line.strip().startswith('data ='):
            data_start = i
        elif line.strip().startswith('model ='):
            model_start = i
        elif line.strip().startswith('training ='):
            training_start = i
    
    # Reconstruct file with updated values
    with open(config_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.writelines(lines[:import_end])
        f.write('\n\n')
        
        # Write data section
        f.write('# Dataset configuration\n')
        f.write(f'data = {repr(config["data"])}\n\n\n')
        
        # Write model section
        f.write('# Model architecture (from best Optuna params: val_loss=0.1855)\n')
        f.write(f'model = {repr(config["model"])}\n\n\n')
        
        # Write training section (preserve classes)
        f.write('# Training configuration\n')
        f.write('training = {\n')
        for key, value in config["training"].items():
            if key.endswith('_class'):
                # Preserve class references
                if 'Adam' in str(value):
                    f.write(f'    "{key}": optim.Adam,\n')
                elif 'SGD' in str(value):
                    f.write(f'    "{key}": optim.SGD,\n')
                elif 'MSELoss' in str(value):
                    f.write(f'    "{key}": nn.MSELoss,\n')
                elif 'CosineAnnealingLR' in str(value):
                    f.write(f'    "{key}": optim.lr_scheduler.CosineAnnealingLR,\n')
                else:
                    f.write(f'    "{key}": {value},\n')
            else:
                f.write(f'    "{key}": {repr(value)},\n')
        f.write('}\n\n\n')
        
        # Write rest
        f.write('# Logging and reproducibility\n')
        f.write(f'logging = {repr(config["logging"])}\n\n\n')
        f.write('# Random seed for reproducibility\n')
        f.write(f'seed = {config["seed"]}\n\n\n')
        f.write('# Export config as dict for compatibility\n')
        f.write('def get_config():\n')
        f.write('    """Return config as a dictionary."""\n')
        f.write('    return {\n')
        f.write('        "data": data,\n')
        f.write('        "model": model,\n')
        f.write('        "training": training,\n')
        f.write('        "logging": logging,\n')
        f.write('        "seed": seed,\n')
        f.write('    }\n')


def run_training(config_path: Path, run_tag: str) -> dict:
    """
    Run training via subprocess and return metrics.
    
    This simulates an LLM agent executing a shell command.
    """
    print(f"\n{'='*80}")
    print(f"LLM Agent: Running training for {run_tag}...")
    print(f"{'='*80}")
    
    cmd = [sys.executable, "train.py", "--config", str(config_path), "--tag", run_tag]
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None
    
    # Read results
    run_dir = Path("runs") / run_tag
    metrics = load_metrics(run_dir)
    history = load_history(run_dir)
    summary = build_history_summary(history)
    
    return {
        "metrics": metrics,
        "history": history,
        "summary": summary
    }


def llm_agent_reasoning(state, recent_results=None):
    """
    Simulate LLM agent reasoning about next architecture to try.
    
    In practice, this would be the LLM's internal reasoning process.
    """
    print("\n" + "="*80)
    print("LLM Agent Reasoning:")
    print("="*80)
    
    if len(state.runs) == 0:
        print("No previous runs. Starting with baseline configuration.")
        return "baseline", {}
    
    best = get_best_run(state)
    recent = get_recent_runs(state, k=3)
    
    print(f"\nCurrent best: {best.run_id} with val_mse={best.val_mse:.6f}")
    print(f"\nRecent runs:")
    for run in recent:
        print(f"  - {run.run_id}: val_mse={run.val_mse:.6f}, accepted={run.accepted}")
    
    if recent_results:
        print(f"\nLatest run summary:")
        print(f"  {recent_results['summary']}")
    
    # Simple heuristic: alternate between different modifications
    run_number = len(state.runs)
    
    if run_number % 3 == 0:
        print("\nDecision: Try increasing model capacity (more layers)")
        return "increase_layers", {"n_encoder_layers": 3, "n_decoder_layers": 2}
    elif run_number % 3 == 1:
        print("\nDecision: Try reducing model size (smaller d_model)")
        return "reduce_size", {"d_model": 256, "d_ff": 1024}
    else:
        print("\nDecision: Try different attention configuration")
        return "adjust_attention", {"n_heads": 4}


def modify_config(config: dict, strategy: str, params: dict) -> dict:
    """
    Modify configuration based on strategy.
    
    This simulates an LLM agent editing the config file.
    """
    import copy
    new_config = copy.deepcopy(config)
    
    # Modify model parameters
    for key, value in params.items():
        new_config["model"][key] = value
    
    return new_config


def main():
    """
    Main simulation of LLM agent performing NAS.
    
    This demonstrates the complete workflow:
    1. Load global state
    2. Reason about next architecture
    3. Modify config
    4. Run training
    5. Analyze results
    6. Update global state
    7. Repeat
    """
    print("="*80)
    print("LLM-based NAS Agent Simulation")
    print("="*80)
    print("\nThis script simulates how an external LLM agent would interact")
    print("with the NAS system to perform architecture search.\n")
    
    config_path = Path("config.py")
    num_iterations = 3  # Run 3 iterations for demonstration
    
    # Load or initialize agent state
    state = load_agent_state(Path("agent_state.json"))
    print(f"Loaded agent state: {len(state.runs)} previous runs")
    
    # Load base config
    base_config = read_config(config_path)
    
    for iteration in range(num_iterations):
        print(f"\n{'#'*80}")
        print(f"# Iteration {iteration + 1}/{num_iterations}")
        print(f"{'#'*80}")
        
        # LLM reasoning: decide what to try next
        recent_results = None
        if len(state.runs) > 0:
            last_run_dir = Path("runs") / state.runs[-1].run_id
            if last_run_dir.exists():
                recent_results = {
                    "metrics": load_metrics(last_run_dir),
                    "history": load_history(last_run_dir),
                    "summary": state.runs[-1].history_summary
                }
        
        strategy, params = llm_agent_reasoning(state, recent_results)
        
        # Modify config based on strategy
        if strategy == "baseline":
            # Use base config as-is
            current_config = base_config
        else:
            current_config = modify_config(base_config, strategy, params)
            print(f"\nModified config: {params}")
        
        # Save modified config
        write_config(config_path, current_config)
        
        # Run training
        run_tag = f"run_{len(state.runs):04d}"
        results = run_training(config_path, run_tag)
        
        if results is None:
            print("Training failed, skipping this run")
            continue
        
        # Analyze results
        metrics = results["metrics"]
        summary = results["summary"]
        
        print(f"\n{'='*80}")
        print("LLM Agent: Analyzing results...")
        print(f"{'='*80}")
        print(f"Val MSE: {metrics['val_mse']:.6f}")
        print(f"Test MSE: {metrics['test_mse']:.6f}")
        print(f"Training time: {metrics['train_time_sec']:.2f}s")
        print(f"\nSummary: {summary}")
        
        # Decide whether to accept this run
        best = get_best_run(state)
        if best is None or metrics["val_mse"] < best.val_mse * 1.1:  # Accept if within 10% of best
            accepted = True
            print("\n✓ Run ACCEPTED")
        else:
            accepted = False
            print("\n✗ Run REJECTED (val_mse too high)")
        
        # Update global state
        run_info = RunInfo(
            run_id=run_tag,
            val_mse=metrics["val_mse"],
            test_mse=metrics["test_mse"],
            history_summary=summary,
            accepted=accepted
        )
        state = add_run(state, run_info)
        save_agent_state(state, Path("agent_state.json"))
        
        print(f"\nUpdated agent_state.json")
    
    # Final summary
    print(f"\n{'#'*80}")
    print("# Final Summary")
    print(f"{'#'*80}")
    
    best = get_best_run(state)
    print(f"\nCompleted {len(state.runs)} runs")
    print(f"Best run: {best.run_id}")
    print(f"  Val MSE: {best.val_mse:.6f}")
    print(f"  Test MSE: {best.test_mse:.6f}")
    print(f"\nAll runs:")
    for run in state.runs:
        status = "✓" if run.accepted else "✗"
        best_marker = " ← BEST" if run.run_id == best.run_id else ""
        print(f"  {status} {run.run_id}: val_mse={run.val_mse:.6f}{best_marker}")
    
    print(f"\n{'='*80}")
    print("Simulation complete!")
    print(f"{'='*80}")
    print("\nResults saved in:")
    print(f"  - runs/run_XXXX/  (individual run outputs)")
    print(f"  - agent_state.json (global NAS state)")


if __name__ == "__main__":
    main()
