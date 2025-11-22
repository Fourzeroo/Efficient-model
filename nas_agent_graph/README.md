# LangGraph NAS Agent System

LLM-orchestrated Neural Architecture Search using LangGraph and OpenRouter.

## Overview

This system provides an automated, LLM-driven approach to Neural Architecture Search (NAS) for the Informer time series forecasting model. It uses:

- **LangGraph** for workflow orchestration
- **OpenRouter** for LLM API access (Claude 3.5 Sonnet by default)
- **Structured Output (Pydantic)** for reliable LLM responses
- **Hybrid Evaluation** (deterministic rules + LLM for borderline cases)
- **Phoenix Observability** for comprehensive tracing and monitoring

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐      ┌──────────┐      ┌───────────┐         │
│  │ Planner  │ ───> │ Executor │ ───> │ Evaluator │         │
│  │  (LLM)   │      │ (Python) │      │  (Hybrid) │         │
│  └──────────┘      └──────────┘      └───────────┘         │
│       │                                     │                │
│       │                                     │                │
│       └──────────────── Loop ──────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Planner Agent (LLM)**
   - Analyzes training history and current config
   - Decides what hyperparameters/architecture to change next
   - Returns structured output with concrete config changes
   - Step types: "small", "medium", "radical", "stop"

2. **Executor (Python)**
   - Applies config changes to `nas_system/config.py`
   - Runs the training script (`train.py`)
   - Loads results (metrics and history)

3. **Evaluator (Hybrid)**
   - Uses deterministic rules for clear cases:
     - Accept: improvement > 2%
     - Reject: degradation > 0.5% or severe overfitting
     - Borderline: small improvements or minor regressions
   - Falls back to LLM for borderline cases
   - Updates global agent state

4. **State Manager**
   - Tracks all runs in `agent_state.json`
   - Maintains best run across iterations
   - Provides history for Planner context

## Installation

### 1. Install Dependencies

```bash
cd nas_agent_graph
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional (defaults shown)
PLANNER_MODEL=anthropic/claude-3.5-sonnet
EVALUATOR_MODEL=anthropic/claude-3.5-sonnet
VERBOSE=true
DEBUG=false
```

Get your OpenRouter API key from: https://openrouter.ai/

### 3. Verify nas_system Setup

Ensure the `nas_system` directory is properly configured:

```bash
cd ../nas_system
python train.py --config config.py --tag test_run
```

This should complete successfully before running the NAS agent.

## Usage

### Python API

```python
from nas_agent_graph import run_nas_optimization

# Run 20 iterations of NAS
run_nas_optimization(max_iterations=20)
```

### Command Line

```bash
# Run with default settings (20 iterations)
python -m nas_agent_graph.main

# Customize iterations
python -m nas_agent_graph.main --max-iterations 30

# Enable verbose output
python -m nas_agent_graph.main --max-iterations 20 --verbose
```

## How It Works

### Iteration Flow

1. **Planner analyzes context:**
   - Current iteration / max iterations
   - Best run so far (val_mse, run_id)
   - Recent runs summary (last 5)
   - Current configuration

2. **Planner decides:**
   - Step type: small/medium/radical/stop
   - Plan: what to try
   - Reason: why this makes sense
   - Config changes: concrete updates

3. **Executor applies changes:**
   - Modifies `nas_system/config.py` using Python AST manipulation
   - Runs training: `python train.py --config config.py --tag run_XXXX`
   - Loads results from `runs/run_XXXX/`

4. **Evaluator assesses results:**
   - Deterministic rules first (fast, reliable)
   - LLM evaluation for borderline cases
   - Decision: accept or reject

5. **State update:**
   - Add RunInfo to agent_state.json
   - Update best_run_id if improved
   - Increment iteration

6. **Loop or stop:**
   - Continue if iteration < max_iterations and step_type != "stop"
   - Otherwise, finish and report results

### Example Iteration

```
================================================================================
PLANNER - Iteration 0/20
================================================================================
Best so far: None (val_mse=inf)
Recent runs: 0

Plan: Start with baseline configuration to establish performance
Step type: small
Reason: No previous runs; establish baseline
Changes: {}

================================================================================
EXECUTOR - Running iteration 0
================================================================================
Running training: python train.py --config config.py --tag run_0000
Training completed successfully for run_0000

================================================================================
EVALUATOR - Evaluating run_0000
================================================================================
Deterministic decision: Accept
Reason: First run with reasonable validation MSE

================================================================================
UPDATE STATE NODE
================================================================================
Added run run_0000 to agent state
  Val MSE: 0.1855
  Test MSE: 0.1923
  Accepted: True
Iteration incremented to 1
```

## Configuration Changes

The Planner can propose changes using dot-notation keys:

### Hyperparameter Changes

```python
{
    "training.optimizer_params.lr": 0.0001,        # Learning rate
    "training.optimizer_params.weight_decay": 0.01, # Weight decay
    "model.dropout": 0.3,                          # Dropout rate
    "data.batch_size": 32,                         # Batch size
}
```

### Architecture Changes

```python
{
    "model.d_model": 256,      # Model dimension
    "model.n_heads": 8,        # Attention heads
    "model.e_layers": 3,       # Encoder layers
    "model.d_ff": 1024,        # Feedforward dimension
}
```

### Optimizer/Scheduler Changes

```python
{
    "training.optimizer_class": "AdamW",           # Adam, AdamW, SGD
    "training.scheduler_class": "StepLR",          # StepLR, CosineAnnealingLR, etc.
    "training.criterion_class": "SmoothL1Loss",    # MSELoss, L1Loss, SmoothL1Loss
}
```

## File Structure

```
nas_agent_graph/
├── __init__.py              # Package exports
├── main.py                  # Entry point (run_nas_optimization)
├── config.py                # Configuration (API keys, paths)
├── prompts.py               # Prompt templates for LLM agents
├── planner.py               # Planner LLM agent
├── executor.py              # Config modification + training runner
├── evaluator.py             # Hybrid evaluator (rules + LLM)
├── graph.py                 # LangGraph workflow definition
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Outputs

### Agent State (`nas_system/agent_state.json`)

Tracks all runs and best performance:

```json
{
  "max_runs": 100,
  "best_run_id": "run_0003",
  "runs": [
    {
      "run_id": "run_0000",
      "val_mse": 0.1855,
      "test_mse": 0.1923,
      "history_summary": "20 epochs, best_epoch=17...",
      "accepted": true
    },
    ...
  ]
}
```

### Run Outputs (`nas_system/runs/run_XXXX/`)

Each run produces:
- `metrics.json` - Final metrics
- `history.json` - Per-epoch training history
- `learning_curve.png` - Visualization
- `best_model.pth` - Model checkpoint
- `config_used.py` - Config snapshot

## Advanced Usage

### Resume from Previous Session

The system automatically resumes from the last completed iteration:

```python
# First session: runs iterations 0-9
run_nas_optimization(max_iterations=10)

# Second session: continues from iteration 10
run_nas_optimization(max_iterations=20)
```

### Custom Models

To use different LLM models, set environment variables:

```bash
PLANNER_MODEL=anthropic/claude-3-opus
EVALUATOR_MODEL=openai/gpt-4-turbo
```

Supported models (via OpenRouter):
- `anthropic/claude-3.5-sonnet` (default, best balance)
- `anthropic/claude-3-opus` (most capable, expensive)
- `openai/gpt-4-turbo` (good alternative)
- `openai/gpt-3.5-turbo` (faster, cheaper, less capable)

### Debug Mode

Enable detailed logging:

```python
import nas_agent_graph.config as cfg
cfg.VERBOSE = True
cfg.DEBUG = True

run_nas_optimization(max_iterations=5)
```

## Design Principles

1. **Simplicity over complexity**
   - No MCP servers or tool-calling
   - Direct Python execution
   - Clear, linear workflow

2. **Reliability through structure**
   - Pydantic models for LLM outputs
   - Deterministic rules for clear cases
   - LLM only for uncertainty

3. **Transparency**
   - Verbose logging of decisions
   - All state saved to disk
   - Reproducible runs via config snapshots

4. **Modularity**
   - Each component has one responsibility
   - Easy to extend or replace components
   - Clean separation of LLM and Python logic

## Troubleshooting

### Import Errors

If you see import errors for `langgraph`, `langchain_openai`, etc.:

```bash
pip install -r requirements.txt
```

### OpenRouter API Key Error

```
ValueError: OPENROUTER_API_KEY not found in environment
```

Create a `.env` file with your API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

### Training Failures

If training consistently fails:

1. Test training manually:
   ```bash
   cd nas_system
   python train.py --config config.py --tag manual_test
   ```

2. Check GPU availability and CUDA setup

3. Verify dataset can be downloaded

### Config Modification Issues

If config changes aren't applied correctly:

1. Check the config.py format matches expectations
2. Enable debug mode to see regex patterns
3. Use simple changes first (e.g., single numeric values)

## Comparison with Existing `nas_system`

| Feature | `nas_system` | `nas_agent_graph` |
|---------|-------------|-------------------|
| **LLM Integration** | External (manual) | Built-in (automated) |
| **Workflow** | Manual loop | LangGraph state machine |
| **Decision Making** | Human | LLM Planner |
| **Evaluation** | Human | Hybrid (rules + LLM) |
| **Orchestration** | None | Automatic |
| **State Management** | `agent_state.py` | Same (reused) |
| **Training Script** | `train.py` | Same (reused) |

The `nas_agent_graph` builds on top of `nas_system` to provide full automation.

## Future Enhancements

Potential improvements:

1. **Parallel Runs**: Train multiple configs simultaneously
2. **Bayesian Optimization**: Combine LLM with BO for hyperparameters
3. **Early Stopping**: Stop unpromising runs early
4. **Architecture Search**: Modify `model.py` structure (advanced)
5. **Multi-Objective**: Optimize for speed + accuracy
6. **Transfer Learning**: Learn from past NAS sessions

## License

Same as parent project.

## Credits

Built on top of the `nas_system` framework, using:
- LangGraph for orchestration
- OpenRouter for LLM access
- Informer2020 for time series modeling
