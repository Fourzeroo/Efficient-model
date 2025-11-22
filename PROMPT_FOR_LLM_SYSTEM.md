# Project: NAS Multi-Agent System with LangGraph and OpenRouter

## Context: Review Existing Implementation First

**IMPORTANT:** Before starting, please explore the existing `nas_system` directory to understand the current implementation:

1. **Read the documentation:**
   - `nas_system/README.md` - Overview of the system
   - `nas_system/QUICKSTART.md` - Quick start guide
   - `nas_system/PYTHON_CONFIG_MIGRATION.md` - Recent migration to Python config

2. **Examine the existing codebase:**
   - `nas_system/config.py` - **Python configuration file** (NOT YAML)
   - `nas_system/model.py` - Model definition (Informer from Informer2020)
   - `nas_system/train.py` - Training script
   - `nas_system/nas_agent/` - Existing Python package with utilities

3. **Understand the key differences from typical setups:**
   - Configuration is in **Python format**, not YAML
   - Config uses **direct class references** (e.g., `optim.Adam`, `nn.MSELoss`)
   - Training script loads config dynamically via `importlib`
   - Model uses real Informer from external Informer2020 repository

---

## What We Already Have

We have a fully functional training backend in the `nas_system` directory:

### Core Scripts

- **`train.py`** — A CLI script that trains a time-series model (Informer) based on `config.py` and `model.py`.
  - Usage: `python train.py --config config.py --tag run_xxxx`
  - Writes `metrics.json`, `history.json`, `learning_curve.png`, and `best_model.pth` into `runs/run_xxxx/`
  - Creates a snapshot of the config as `config_used.py` in the run directory
  - **This file should NOT be modified by agents**

- **`config.py`** — **Python configuration module** (not YAML) with:
  - Direct PyTorch class references: `optim.Adam`, `nn.MSELoss`, `optim.lr_scheduler.CosineAnnealingLR`
  - Model architecture parameters: `d_model`, `n_heads`, `e_layers`, `d_ff`, `dropout`, etc.
  - Training hyperparameters: optimizer, scheduler, loss function, learning rate, etc.
  - Data configuration: dataset path, batch size, sequence lengths
  - `get_config()` function that returns the config as a dictionary
  - **This file CAN be modified by agents**

- **`model.py`** — Model definition that:
  - Imports real Informer from `Informer2020` repository (expected in parent directory)
  - Provides `build_model(config)` function
  - **This file CAN be modified by agents to explore different architectures (optional, advanced feature)**

### Python Package: `nas_agent`

Located at `nas_system/nas_agent/`, provides utilities:

- **`run_manager.py`** — Run directory management:
  - `get_run_dir(tag: str) -> Path` - Create/get run directory
  - `snapshot_config(config_path: Path, run_dir: Path)` - Copy config to run directory
  - `ensure_runs_root()` - Ensure runs directory exists

- **`logging_utils.py`** — Metrics and history I/O:
  - `save_metrics(run_dir: Path, metrics: dict)`
  - `load_metrics(run_dir: Path) -> dict`
  - `save_history(run_dir: Path, history: dict)`
  - `load_history(run_dir: Path) -> dict`

- **`history_summary.py`** — Learning curve analysis:
  - `build_history_summary(history: dict) -> str` - Generates natural language summary of training dynamics for LLM consumption

- **`agent_state.py`** — Global NAS state management:
  - `RunInfo` dataclass: Contains `run_id`, `val_mse`, `test_mse`, `history_summary`, `accepted`
  - `AgentState` dataclass: Contains `max_runs`, `best_run_id`, `runs: List[RunInfo]`
  - `load_agent_state(path: Path = Path("agent_state.json")) -> AgentState`
  - `save_agent_state(state: AgentState, path: Path)`
  - `add_run(state: AgentState, run: RunInfo) -> AgentState` - Adds run and updates best_run_id
  - `get_best_run(state: AgentState) -> Optional[RunInfo]`
  - `get_recent_runs(state: AgentState, k: int = 5) -> List[RunInfo]`

### Key Implementation Details

1. **Config Format**: The config is a **Python module** (`config.py`), not YAML:
   ```python
   import torch.optim as optim
   import torch.nn as nn
   
   training = {
       "optimizer_class": optim.Adam,  # Direct class reference
       "optimizer_params": {"lr": 1.603e-05},
       "criterion_class": nn.MSELoss,
       "scheduler_class": optim.lr_scheduler.CosineAnnealingLR,
   }
   ```

2. **Training Loop**: `train.py` does NOT need to be modified by agents:
   - Only `config.py` (and optionally `model.py`) should be edited by agents
   - The training loop (`train_epoch`, `validate` functions) is fixed

3. **Model**: Uses real Informer from Informer2020 repository:
   - Requires `Informer2020` directory at `../Informer2020/`
   - Agents can modify hyperparameters via config or change architecture in `model.py`

4. **Outputs per run** (`runs/run_xxxx/`):
   - `metrics.json` - Final metrics: `{"val_mse": 0.185, "test_mse": 0.192, "best_epoch": 12, ...}`
   - `history.json` - Per-epoch history: `{"epochs": [{"epoch": 0, "train_mse": 1.02, "val_mse": 1.05}, ...]}`
   - `learning_curve.png` - Visualization
   - `best_model.pth` - Best model checkpoint
   - `config_used.py` - Snapshot of config used for this run

---

## What You Need to Build

Create a **simplified 2-component LLM orchestration system** in a new directory `nas_agent_graph/`:

### Architecture Overview

The system consists of two main components orchestrated by LangGraph:

1. **Planner Agent (LLM)** - Analyzes history and decides what to try next
2. **Executor (Python)** - Applies changes and runs training
3. **Evaluator (Hybrid)** - Uses deterministic rules + optional LLM for edge cases

**Key simplifications compared to complex multi-agent systems:**
- No MCP servers needed - use simple Python functions
- Structured output with Pydantic instead of JSON string parsing
- Deterministic evaluation with rules, LLM only for edge cases
- Single LLM call per iteration (Planner only)

### Component 1: Planner Agent (LLM)

**Purpose:** Analyze experiment history and decide the next step.

**Input:**
- Current iteration number
- Global state from `agent_state.json` (best run, recent runs)
- History summaries from `history_summary.py`
- Current `config.py` content

**Output (Pydantic model):**
```python
class PlannerDecision(BaseModel):
    step_type: Literal["small", "medium", "radical", "stop"]
    plan: str = Field(description="What to try next (human-readable)")
    reason: str = Field(description="Why this approach makes sense")
    config_changes: Dict[str, Any] = Field(
        description="Specific changes to apply to config.py"
    )
```

**Step Types:**
- `small`: Minor hyperparameter tuning (lr, batch_size, dropout, grad_clip)
- `medium`: Architecture changes (n_heads, d_model, e_layers, d_ff, activation)
- `radical`: Major changes (new optimizer, scheduler, data augmentation)
- `stop`: Goal achieved or max iterations reached

**Implementation:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="anthropic/claude-3.5-sonnet"
)

planner = llm.with_structured_output(PlannerDecision)
```

**Prompt Template:**
```python
PLANNER_PROMPT = """You are a Neural Architecture Search expert optimizing Informer for time series forecasting.

Current State:
- Iteration: {iteration}/{max_iterations}
- Best validation MSE: {best_val_mse}
- Best run ID: {best_run_id}

Recent Runs:
{recent_runs_summary}

Current Config:
{current_config}

Task:
Analyze the history and decide what to try next. Consider:
- Learning dynamics (overfitting, underfitting, instability)
- Architecture capacity (too small/large)
- Training hyperparameters (lr too high/low, batch size)

Choose step_type based on:
- small: Fine-tune hyperparameters (1.1-2x changes)
- medium: Architectural changes (2-4x changes)
- radical: Major strategy shift
- stop: If val_mse < 0.17 or iteration >= max_iterations

Provide specific config_changes as a dict, e.g.:
{{"training.optimizer_params.lr": 0.0001, "model.d_model": 256}}
"""
```

### Component 2: Executor (Python)

**Purpose:** Apply Planner's decisions and run training.

**Key Functions:**

```python
def apply_config_changes(config_path: Path, changes: Dict[str, Any]) -> None:
    """
    Modify config.py based on nested key paths.
    
    Example:
        changes = {"training.optimizer_params.lr": 0.0001}
        Updates: training["optimizer_params"]["lr"] = 0.0001
    """
    # Read current config
    with open(config_path) as f:
        config_code = f.read()
    
    # Parse as AST and modify
    # (Use ast.parse, ast.NodeTransformer, or simple string replacement)
    
    # Write back
    with open(config_path, "w") as f:
        f.write(modified_code)

def run_training(config_path: Path, tag: str) -> str:
    """
    Run training script and return run_id.
    
    Returns:
        run_id: e.g., "run_0005"
    """
    cmd = f"python train.py --config {config_path} --tag {tag}"
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd="nas_system",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    return tag

def load_run_results(run_id: str) -> Tuple[dict, dict]:
    """
    Load metrics and history for a completed run.
    
    Returns:
        metrics: {"val_mse": 0.185, "test_mse": 0.192, ...}
        history: {"epochs": [{"epoch": 0, "train_mse": 1.02, ...}, ...]}
    """
    from nas_agent.logging_utils import load_metrics, load_history
    
    run_dir = Path(f"nas_system/runs/{run_id}")
    metrics = load_metrics(run_dir)
    history = load_history(run_dir)
    
    return metrics, history
```

### Component 3: Evaluator (Hybrid)

**Purpose:** Decide whether to accept or reject a run using deterministic rules, with optional LLM consultation for edge cases.

**Deterministic Rules:**

```python
def evaluate_run_deterministic(
    new_run: dict, 
    best_run: dict
) -> Tuple[Optional[bool], str]:
    """
    Returns:
        (decision, reason) where decision is:
        - True: Accept
        - False: Reject
        - None: Unclear, need LLM
    """
    new_val = new_run["val_mse"]
    best_val = best_run["val_mse"]
    
    # Calculate improvement
    improvement = (best_val - new_val) / best_val * 100
    
    # Rule 1: Significant improvement
    if improvement > 2.0:
        return True, f"Significant improvement: {improvement:.1f}%"
    
    # Rule 2: Clear regression
    if improvement < -0.5:
        return False, f"Performance degraded: {improvement:.1f}%"
    
    # Rule 3: Check overfitting
    train_val_gap = abs(new_run["train_mse"] - new_val) / new_val * 100
    if train_val_gap > 50:
        return False, f"Severe overfitting: train/val gap {train_val_gap:.1f}%"
    
    # Rule 4: Training failed
    if new_run.get("best_epoch", 0) < 3:
        return False, "Training failed or converged too quickly"
    
    # Edge case: small improvement (0% - 2%)
    if 0 < improvement <= 2.0:
        return None, f"Small improvement: {improvement:.1f}%, needs LLM review"
    
    # Edge case: minor regression (-0.5% to 0%)
    if -0.5 <= improvement <= 0:
        return None, f"Minor regression: {improvement:.1f}%, needs LLM review"
    
    return False, "No clear improvement"
```

**LLM Consultation (for edge cases):**

```python
class EvaluatorDecision(BaseModel):
    accept: bool
    reason: str = Field(description="Justification for accept/reject")
    feedback: str = Field(description="Suggestions for next iteration")

evaluator_llm = llm.with_structured_output(EvaluatorDecision)

EVALUATOR_PROMPT = """You are evaluating a borderline training run.

New Run:
- val_mse: {new_val_mse}
- test_mse: {new_test_mse}
- train_mse: {new_train_mse}
- best_epoch: {new_best_epoch}

Best Run So Far:
- val_mse: {best_val_mse}
- test_mse: {best_test_mse}

Changes Made:
{changes_summary}

History Summary:
{history_summary}

The automatic rules are uncertain. Decide whether to accept this run considering:
- Is the improvement meaningful even if small?
- Are the training dynamics healthy?
- Does the test_mse align with val_mse?

Provide feedback for the next iteration.
"""
```

### LangGraph Workflow

**State Definition:**

```python
from typing import TypedDict, Literal, Optional
from langchain_core.pydantic_v1 import BaseModel

class NASGraphState(TypedDict):
    # Iteration tracking
    iteration: int
    max_iterations: int
    
    # Planner outputs
    step_type: Literal["small", "medium", "radical", "stop"]
    plan: str
    config_changes: dict
    
    # Executor outputs
    current_run_id: str
    training_success: bool
    
    # Evaluator outputs
    accept_run: bool
    evaluation_reason: str
    feedback: str
    
    # Global state
    agent_state: dict  # From agent_state.json
```

**Graph Structure:**

```python
from langgraph.graph import StateGraph, END

def build_nas_graph():
    workflow = StateGraph(NASGraphState)
    
    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("update_state", update_state_node)
    
    # Edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    workflow.add_edge("evaluator", "update_state")
    
    # Conditional routing: continue or stop
    workflow.add_conditional_edges(
        "update_state",
        should_continue,
        {
            "continue": "planner",
            "stop": END
        }
    )
    
    return workflow.compile()

def should_continue(state: NASGraphState) -> Literal["continue", "stop"]:
    if state["step_type"] == "stop":
        return "stop"
    if state["iteration"] >= state["max_iterations"]:
        return "stop"
    return "continue"
```

**Node Implementations:**

```python
def planner_node(state: NASGraphState) -> NASGraphState:
    """Call Planner LLM to decide next step."""
    agent_state = load_agent_state()
    config_content = Path("nas_system/config.py").read_text()
    
    best_run = get_best_run(agent_state)
    recent_runs = get_recent_runs(agent_state, k=5)
    
    prompt = PLANNER_PROMPT.format(
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
        best_val_mse=best_run.val_mse if best_run else "N/A",
        best_run_id=best_run.run_id if best_run else "N/A",
        recent_runs_summary=format_recent_runs(recent_runs),
        current_config=config_content
    )
    
    decision = planner.invoke(prompt)
    
    return {
        **state,
        "step_type": decision.step_type,
        "plan": decision.plan,
        "config_changes": decision.config_changes
    }

def executor_node(state: NASGraphState) -> NASGraphState:
    """Apply config changes and run training."""
    if state["step_type"] == "stop":
        return state
    
    try:
        # Apply changes
        apply_config_changes(
            Path("nas_system/config.py"),
            state["config_changes"]
        )
        
        # Run training
        run_id = f"run_{state['iteration']:04d}"
        run_training(Path("nas_system/config.py"), run_id)
        
        return {
            **state,
            "current_run_id": run_id,
            "training_success": True
        }
    except Exception as e:
        print(f"Training failed: {e}")
        return {
            **state,
            "current_run_id": "",
            "training_success": False
        }

def evaluator_node(state: NASGraphState) -> NASGraphState:
    """Evaluate run with rules + optional LLM."""
    if not state["training_success"]:
        return {
            **state,
            "accept_run": False,
            "evaluation_reason": "Training failed",
            "feedback": "Fix configuration errors"
        }
    
    # Load results
    metrics, history = load_run_results(state["current_run_id"])
    agent_state = load_agent_state()
    best_run = get_best_run(agent_state)
    
    # Try deterministic evaluation
    decision, reason = evaluate_run_deterministic(
        metrics,
        best_run.__dict__ if best_run else {"val_mse": float("inf")}
    )
    
    if decision is not None:
        # Clear decision from rules
        return {
            **state,
            "accept_run": decision,
            "evaluation_reason": reason,
            "feedback": ""
        }
    
    # Edge case: consult LLM
    history_summary = build_history_summary(history)
    prompt = EVALUATOR_PROMPT.format(
        new_val_mse=metrics["val_mse"],
        new_test_mse=metrics["test_mse"],
        new_train_mse=metrics["train_mse"],
        new_best_epoch=metrics["best_epoch"],
        best_val_mse=best_run.val_mse if best_run else "N/A",
        best_test_mse=best_run.test_mse if best_run else "N/A",
        changes_summary=str(state["config_changes"]),
        history_summary=history_summary
    )
    
    llm_decision = evaluator_llm.invoke(prompt)
    
    return {
        **state,
        "accept_run": llm_decision.accept,
        "evaluation_reason": llm_decision.reason,
        "feedback": llm_decision.feedback
    }

def update_state_node(state: NASGraphState) -> NASGraphState:
    """Update agent_state.json with results."""
    agent_state = load_agent_state()
    
    if state["training_success"]:
        metrics, history = load_run_results(state["current_run_id"])
        history_summary = build_history_summary(history)
        
        run_info = RunInfo(
            run_id=state["current_run_id"],
            val_mse=metrics["val_mse"],
            test_mse=metrics["test_mse"],
            history_summary=history_summary,
            accepted=state["accept_run"]
        )
        
        agent_state = add_run(agent_state, run_info)
        save_agent_state(agent_state)
    
    return {
        **state,
        "iteration": state["iteration"] + 1,
        "agent_state": agent_state.__dict__
    }
```

### Main Entry Point

```python
def run_nas_optimization(max_iterations: int = 20):
    """Run the NAS optimization loop."""
    # Initialize
    agent_state = load_agent_state()
    graph = build_nas_graph()
    
    initial_state = NASGraphState(
        iteration=0,
        max_iterations=max_iterations,
        step_type="small",
        plan="",
        config_changes={},
        current_run_id="",
        training_success=True,
        accept_run=False,
        evaluation_reason="",
        feedback="",
        agent_state=agent_state.__dict__
    )
    
    # Run graph
    final_state = graph.invoke(initial_state)
    
    # Print summary
    best_run = get_best_run(load_agent_state())
    print(f"\n=== Optimization Complete ===")
    print(f"Best run: {best_run.run_id}")
    print(f"Best val_mse: {best_run.val_mse:.4f}")
    print(f"Test mse: {best_run.test_mse:.4f}")

if __name__ == "__main__":
    run_nas_optimization(max_iterations=20)
```

---

## Directory Structure

After implementation, you should have:

```
nas_system/
    config.py               # Modified by agents
    model.py                # Optionally modified by agents
    train.py                # Fixed, not modified
    nas_agent/              # Existing utilities
        __init__.py
        agent_state.py
        run_manager.py
        logging_utils.py
        history_summary.py
    runs/                   # Training outputs
        run_0001/
        run_0002/
        ...
    agent_state.json        # Global state

nas_agent_graph/            # NEW: LLM orchestration
    __init__.py
    main.py                 # Entry point (run_nas_optimization)
    planner.py              # Planner agent
    executor.py             # Executor functions
    evaluator.py            # Evaluator (rules + LLM)
    graph.py                # LangGraph workflow
    prompts.py              # Prompt templates
    config.py               # Graph configuration (API keys, etc.)
    requirements.txt        # langgraph, langchain-openai
```

---

## Dependencies

Add to `nas_agent_graph/requirements.txt`:

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
pydantic>=2.0.0
```

**Environment Variables:**

Use `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```

---

## Testing Strategy

1. **Test Planner in isolation:**
   ```python
   from nas_agent_graph.planner import planner, PLANNER_PROMPT
   
   # Mock input
   state = {...}
   decision = planner.invoke(PLANNER_PROMPT.format(**state))
   print(decision)
   ```

2. **Test Executor functions:**
   ```python
   from nas_agent_graph.executor import apply_config_changes, run_training
   
   changes = {"training.optimizer_params.lr": 0.0001}
   apply_config_changes(Path("nas_system/config.py"), changes)
   
   # Verify config was modified
   ```

3. **Test Evaluator rules:**
   ```python
   from nas_agent_graph.evaluator import evaluate_run_deterministic
   
   new_run = {"val_mse": 0.180, "train_mse": 0.175}
   best_run = {"val_mse": 0.185}
   decision, reason = evaluate_run_deterministic(new_run, best_run)
   assert decision == True
   ```

4. **Run full workflow with dry-run:**
   ```python
   # Set max_iterations=2 for quick test
   run_nas_optimization(max_iterations=2)
   ```

---

## Key Design Decisions

1. **Python-native tools instead of MCP:**
   - Simpler implementation
   - No external server dependencies
   - Easier to debug

2. **Structured output with Pydantic:**
   - No JSON parsing errors
   - Type safety
   - Automatic validation

3. **Hybrid evaluation (rules + LLM):**
   - Deterministic for clear cases (90%)
   - LLM only for edge cases (10%)
   - Faster and cheaper

4. **Single LLM agent (Planner):**
   - One LLM call per iteration
   - Simpler state management
   - Lower API costs

5. **Config-only modifications:**
   - Start with config.py changes only
   - Add model.py modifications later (advanced)
   - Safer and more predictable

---

## Success Criteria

After implementation, you should be able to:

1. Run: `python nas_agent_graph/main.py`
2. Watch the system:
   - Analyze previous runs
   - Decide what to try next
   - Modify config.py
   - Run training
   - Evaluate results
   - Loop until goal achieved or max iterations
3. Check `agent_state.json` for history
4. Inspect `nas_system/runs/` for all experiments
5. Find the best configuration in `runs/{best_run_id}/config_used.py`

The system should iteratively improve val_mse from the baseline (~0.185) toward better performance (target: < 0.17).

---

## Additional Notes

- **Error Handling:** Wrap training in try-except to handle crashes gracefully
- **Logging:** Log all decisions and actions for debugging
- **Validation:** Validate config changes before applying (check types, ranges)
- **Checkpointing:** Save graph state periodically in case of interruption
- **Manual Override:** Allow manual intervention to force accept/reject runs

Good luck! Let me know if you have questions about any part of the implementation.
