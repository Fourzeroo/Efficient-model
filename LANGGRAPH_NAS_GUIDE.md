# LangGraph NAS System - Setup and Usage Guide

Complete guide for the LLM-orchestrated Neural Architecture Search system.

## Quick Start

### 1. Install Dependencies

```bash
# Install nas_system dependencies
pip install -r nas_system/requirements.txt

# Install nas_agent_graph dependencies  
pip install -r nas_agent_graph/requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your OpenRouter API key:

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Get your API key from: https://openrouter.ai/

### 3. Test the System

```bash
# Test the base training system first
cd nas_system
python train.py --config config.py --tag test_run
cd ..

# Run a quick NAS test (5 iterations)
python quickstart_nas_graph.py
```

### 4. Run Full NAS Optimization

```bash
# From Python
python -c "from nas_agent_graph import run_nas_optimization; run_nas_optimization(20)"

# Or from command line
python -m nas_agent_graph.main --max-iterations 20
```

## Project Structure

```
Efficient-model/
├── .env                          # Your API keys (create from .env.example)
├── .env.example                  # Template for environment variables
├── quickstart_nas_graph.py       # Quick test script
│
├── nas_system/                   # Base NAS system (existing)
│   ├── config.py                 # Training configuration (MODIFIED BY AGENT)
│   ├── train.py                  # Training script (DO NOT MODIFY)
│   ├── model.py                  # Informer model wrapper
│   ├── nas_agent/                # Utility library
│   │   ├── agent_state.py        # Global NAS state management
│   │   ├── logging_utils.py      # Metrics/history I/O
│   │   ├── history_summary.py    # Learning curve analysis
│   │   └── run_manager.py        # Run directory management
│   ├── runs/                     # Training outputs (auto-created)
│   │   ├── run_0000/
│   │   ├── run_0001/
│   │   └── ...
│   └── agent_state.json          # Global NAS state (auto-created)
│
└── nas_agent_graph/              # LangGraph orchestration (NEW)
    ├── __init__.py               # Package exports
    ├── main.py                   # Entry point
    ├── config.py                 # Graph configuration
    ├── prompts.py                # LLM prompt templates
    ├── planner.py                # Planner LLM agent
    ├── executor.py               # Config editor + training runner
    ├── evaluator.py              # Hybrid evaluator (rules + LLM)
    ├── graph.py                  # LangGraph workflow
    ├── requirements.txt          # Dependencies
    └── README.md                 # Detailed documentation
```

## System Architecture

### Two-Layer Design

1. **Base Layer (`nas_system`)**
   - Manages training runs
   - Provides utilities for state management
   - Can be used manually or by automation

2. **Orchestration Layer (`nas_agent_graph`)**
   - Automates the NAS process using LLMs
   - Builds on top of `nas_system`
   - Provides end-to-end workflow

### LangGraph Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    NAS Iteration Loop                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. PLANNER (LLM)                                        │
│     ├─ Analyze: history, best run, current config       │
│     ├─ Decide: what to change next                      │
│     └─ Output: step_type, plan, config_changes          │
│                        │                                 │
│                        ▼                                 │
│  2. EXECUTOR (Python)                                    │
│     ├─ Modify: nas_system/config.py                     │
│     ├─ Run: python train.py --config config.py          │
│     └─ Load: metrics.json, history.json                 │
│                        │                                 │
│                        ▼                                 │
│  3. EVALUATOR (Hybrid)                                   │
│     ├─ Rules: deterministic accept/reject                │
│     ├─ LLM: for borderline cases                        │
│     └─ Output: accept_run, reason, feedback             │
│                        │                                 │
│                        ▼                                 │
│  4. UPDATE STATE                                         │
│     ├─ Save: RunInfo to agent_state.json                │
│     ├─ Update: best_run_id if improved                  │
│     └─ Increment: iteration counter                     │
│                        │                                 │
│                        ▼                                 │
│  5. DECISION: Continue or Stop?                          │
│     ├─ Continue: if iteration < max and step_type != stop│
│     └─ Stop: otherwise, show final results              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Intelligent Planning
- LLM analyzes training history and learning dynamics
- Proposes specific, actionable config changes
- Step types: small (hyperparameters), medium (architecture), radical (major changes)

### 2. Reliable Execution
- Direct Python config file manipulation (no fragile string parsing)
- Subprocess training in isolated environment
- Automatic result loading and validation

### 3. Hybrid Evaluation
- Fast deterministic rules for clear cases (>2% improvement → accept)
- LLM consultation only for borderline cases (-0.5% to +2%)
- Conservative, reliable decision-making

### 4. State Management
- All runs tracked in `agent_state.json`
- Best model automatically identified
- Easy to resume interrupted sessions

### 5. Full Observability
- Verbose logging of all decisions
- Config snapshots for each run
- Learning curves and metrics preserved

## Configuration Changes Supported

The Planner can propose changes to any field in `nas_system/config.py`:

### Hyperparameters
```python
"training.optimizer_params.lr": 0.0001          # Learning rate
"training.optimizer_params.weight_decay": 0.01  # L2 regularization
"model.dropout": 0.3                            # Dropout rate
"data.batch_size": 32                           # Batch size
```

### Architecture
```python
"model.d_model": 256        # Model dimension
"model.n_heads": 8          # Number of attention heads
"model.e_layers": 3         # Encoder layers
"model.d_layers": 1         # Decoder layers
"model.d_ff": 1024          # Feedforward dimension
"model.factor": 4           # ProbSparse attention factor
```

### Optimizer/Scheduler/Loss
```python
"training.optimizer_class": "AdamW"              # Adam, AdamW, SGD
"training.scheduler_class": "StepLR"             # CosineAnnealingLR, StepLR, etc.
"training.criterion_class": "SmoothL1Loss"       # MSELoss, L1Loss, SmoothL1Loss
```

## Usage Examples

### Basic Usage

```python
from nas_agent_graph import run_nas_optimization

# Run 20 iterations
run_nas_optimization(max_iterations=20)
```

### Resume from Checkpoint

The system automatically resumes from the last completed iteration:

```python
# First session (runs 0-9)
run_nas_optimization(max_iterations=10)

# Later session (continues from 10)
run_nas_optimization(max_iterations=20)
```

### Command Line

```bash
# Basic usage
python -m nas_agent_graph.main

# Custom iterations
python -m nas_agent_graph.main --max-iterations 30

# Verbose output
python -m nas_agent_graph.main --verbose

# Debug mode
python -m nas_agent_graph.main --debug
```

### Analyze Results

```python
from pathlib import Path
from nas_system.nas_agent import load_agent_state, get_best_run, get_recent_runs

# Load state
state = load_agent_state(Path("nas_system/agent_state.json"))

# Get best run
best = get_best_run(state)
print(f"Best: {best.run_id}, val_mse={best.val_mse:.6f}")

# Get recent runs
recent = get_recent_runs(state, k=10)
for run in recent:
    status = "✓" if run.accepted else "✗"
    print(f"{status} {run.run_id}: {run.val_mse:.6f}")
```

## Advanced Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-...

# Optional (with defaults)
PLANNER_MODEL=anthropic/claude-3.5-sonnet
EVALUATOR_MODEL=anthropic/claude-3.5-sonnet
VERBOSE=true
DEBUG=false
```

### Supported Models

Via OpenRouter:
- `anthropic/claude-3.5-sonnet` (default, recommended)
- `anthropic/claude-3-opus` (most capable, expensive)
- `openai/gpt-4-turbo` (good alternative)
- `openai/gpt-3.5-turbo` (faster, cheaper)

### Custom Evaluation Rules

Edit `nas_agent_graph/evaluator.py`:

```python
def evaluate_run_deterministic(new_metrics, best_metrics, history):
    # Add your custom rules
    if custom_condition:
        return True, "Custom accept reason"
    # ...
```

## Troubleshooting

### Training Fails

```bash
# Test training manually
cd nas_system
python train.py --config config.py --tag debug_test
```

Check:
- GPU availability (`nvidia-smi`)
- Dataset download (ETTh1 from GitHub)
- Dependencies installed

### Config Not Modified

- Check config.py format matches expectations
- Enable debug mode: `DEBUG=true`
- Use simple changes first (e.g., single numeric values)

### LLM Errors

```
Error: OPENROUTER_API_KEY not found
```

Solution: Create `.env` file with your API key

```
Error: Rate limit exceeded
```

Solution: Add delay or use cheaper model for Evaluator

### Import Errors

```bash
pip install -r nas_system/requirements.txt
pip install -r nas_agent_graph/requirements.txt
```

## Performance Tips

1. **Start Small**: Use 5-10 iterations first to test
2. **Monitor Costs**: Claude 3.5 Sonnet ~$3 per 1M tokens
3. **Use Cheaper Evaluator**: Set `EVALUATOR_MODEL=openai/gpt-3.5-turbo`
4. **Parallel Runs** (future): Train multiple configs simultaneously
5. **Early Stopping**: Already built into training script

## Expected Results

Based on the current baseline (val_mse=0.1855):

- **Small improvements**: 0.18 - 0.19 (hyperparameter tuning)
- **Medium improvements**: 0.16 - 0.18 (architecture changes)
- **Radical improvements**: 0.14 - 0.16 (major redesign)

The system typically finds improvements within 5-10 iterations.

## Comparison: Manual vs Automated

| Aspect | Manual (`nas_system`) | Automated (`nas_agent_graph`) |
|--------|---------------------|-------------------------------|
| **Setup** | Simple | Requires API key |
| **Speed** | Human-limited | 24/7 automated |
| **Decisions** | Human expertise | LLM reasoning |
| **Exploration** | Biased | Systematic |
| **Cost** | Free (time) | API costs |
| **Learning** | Manual analysis | Automatic history analysis |

## Next Steps

1. ✅ Review existing `nas_system` implementation
2. ✅ Set up `.env` with OpenRouter API key
3. ✅ Test base training: `python nas_system/train.py --config config.py --tag test`
4. ✅ Run quick NAS test: `python quickstart_nas_graph.py`
5. ✅ Analyze results in `nas_system/agent_state.json`
6. ✅ Run full optimization: `python -m nas_agent_graph.main --max-iterations 20`

## Contributing

This is a research project. Key areas for improvement:

1. **Parallel Training**: Run multiple configs simultaneously
2. **Architecture Search**: Modify model.py structure (currently config-only)
3. **Multi-Objective**: Optimize speed + accuracy
4. **Transfer Learning**: Learn from past NAS sessions
5. **Bayesian Optimization**: Hybrid LLM + BO

## References

- **LangGraph**: https://github.com/langchain-ai/langgraph
- **OpenRouter**: https://openrouter.ai/
- **Informer**: https://github.com/zhouhaoyi/Informer2020
- **ETT Dataset**: https://github.com/zhouhaoyi/ETDataset

## License

Same as parent project.
