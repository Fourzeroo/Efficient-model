# Phoenix Observability Integration

This document describes the Phoenix observability integration for the LangGraph NAS system.

## Overview

Phoenix is an observability platform that provides comprehensive tracing and monitoring for LLM applications. The NAS system integrates Phoenix to track:

- **LLM Calls**: All Planner and Evaluator LLM invocations
- **Agent Operations**: Planner, Executor, and Evaluator workflows
- **Training Runs**: Config modifications and training execution
- **Decisions**: Accept/reject decisions with reasoning
- **Performance**: Latency, token usage, and costs

## Prerequisites

### 1. Phoenix Server Running

Phoenix should be running at `localhost:6006`. If you don't have it running:

```bash
# Install Phoenix
pip install arize-phoenix

# Start Phoenix server
python -m phoenix.server.main serve
```

Or if already installed, just access it at: http://localhost:6006

### 2. Install Dependencies

```bash
pip install -r nas_agent_graph/requirements.txt
```

This includes:
- `arize-phoenix>=4.0.0`
- `opentelemetry-api>=1.20.0`
- `opentelemetry-sdk>=1.20.0`
- `openinference-instrumentation-langchain`

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Phoenix Observability
PHOENIX_ENABLED=true          # Enable/disable Phoenix tracing
PHOENIX_HOST=localhost        # Phoenix server host
PHOENIX_PORT=6006            # Phoenix server port
```

### Defaults

If not specified, the system uses these defaults:
- `PHOENIX_ENABLED`: `true`
- `PHOENIX_HOST`: `localhost`
- `PHOENIX_PORT`: `6006`

## Usage

### Automatic Tracing

Phoenix tracing is automatically initialized when you run the NAS system:

```python
from nas_agent_graph import run_nas_optimization

# Phoenix tracing is automatically set up
run_nas_optimization(max_iterations=20)
```

You'll see output like:
```
================================================================================
LLM-ORCHESTRATED NEURAL ARCHITECTURE SEARCH
================================================================================
Max iterations: 20
Using model: anthropic/claude-3.5-sonnet
NAS system root: D:\notebooks\Efficient-model\nas_system
✓ Phoenix tracing enabled at http://localhost:6006/v1/traces
  View traces at: http://localhost:6006
================================================================================
```

### Disable Tracing

To disable Phoenix tracing:

```bash
# In .env file
PHOENIX_ENABLED=false
```

Or programmatically:

```python
import nas_agent_graph.config as cfg
cfg.PHOENIX_ENABLED = False

from nas_agent_graph import run_nas_optimization
run_nas_optimization(max_iterations=20)
```

## What Gets Traced

### 1. Planner Agent Operations

Each Planner invocation creates a span with:

**Attributes:**
- `iteration`: Current iteration number
- `max_iterations`: Total iterations planned
- `best_run_id`: ID of current best run
- `best_val_mse`: Current best validation MSE
- `step_type`: Decision type (small/medium/radical/stop)
- `plan`: Human-readable plan
- `num_config_changes`: Number of config changes proposed

**Events:**
- `agent_state_loaded`: When agent state is loaded
- `llm_invocation_start`: Before LLM call
- `decision_made`: When Planner makes a decision
- `planner_error`: If an error occurs

**Example Trace:**
```
planner_agent [5.2s]
├─ agent_state_loaded
│  └─ total_runs: 3, recent_runs_count: 3
├─ llm_invocation_start
└─ decision_made
   └─ step_type: small, changes: {"training.optimizer_params.lr": 0.0001}
```

### 2. Executor Operations

Each execution creates spans for:

**Main Execution Span:**
- `iteration`: Current iteration
- `step_type`: Type of change being made
- `run_id`: Generated run identifier
- `num_changes`: Number of config modifications

**Sub-operations:**
- `apply_config_changes`: Config file modification
- `training_run`: Actual training execution

**Events:**
- `execution_skipped`: If planner decided to stop
- `config_modified`: After config changes applied
- `training_completed`: After training finishes
- `execution_error`: If errors occur

**Example Trace:**
```
executor [8m 32s]
├─ apply_config_changes [0.2s]
│  └─ changes: {"training.optimizer_params.lr": 0.0001}
├─ config_modified
├─ training_run [8m 31s]
│  └─ run_id: run_0001
└─ training_completed (success: true)
```

### 3. Evaluator Operations

Each evaluation creates spans for:

**Main Evaluation Span:**
- `run_id`: Run being evaluated
- `training_success`: Whether training succeeded
- `val_mse`: Validation MSE achieved
- `test_mse`: Test MSE achieved
- `best_val_mse`: Current best validation MSE (for comparison)
- `evaluation_method`: "deterministic" or "llm"
- `decision`: "accept" or "reject"

**Sub-operations:**
- `deterministic_evaluation`: Rule-based evaluation
- `llm_evaluation`: LLM-based evaluation (if borderline)

**Events:**
- `evaluation_skipped`: If training failed
- `metrics_loaded`: When metrics are loaded
- `borderline_case`: When deterministic rules are uncertain
- `decision_made`: Final accept/reject decision
- `evaluation_error`: If errors occur

**Example Trace (Deterministic):**
```
evaluator [0.3s]
├─ metrics_loaded
│  └─ val_mse: 0.175000, test_mse: 0.182000
├─ deterministic_evaluation [0.1s]
└─ decision_made
   └─ method: deterministic, decision: accept, reason: "Significant improvement: 5.7%"
```

**Example Trace (LLM Borderline):**
```
evaluator [6.8s]
├─ metrics_loaded
│  └─ val_mse: 0.186000, test_mse: 0.193000
├─ deterministic_evaluation [0.1s]
├─ borderline_case
│  └─ reason: "Small improvement: 1.2% (borderline)"
├─ llm_evaluation [6.5s]
│  └─ (LLM invocation traces)
└─ decision_made
   └─ method: llm, decision: accept, reason: "Improvement is meaningful..."
```

### 4. LangChain/LLM Traces

All LLM calls (Planner and Evaluator) are automatically traced by Phoenix's LangChain instrumentation:

**Captured Data:**
- Model name (e.g., `anthropic/claude-3.5-sonnet`)
- Input tokens
- Output tokens
- Latency
- Prompts (full context)
- Responses (structured output)
- Cost estimation

## Viewing Traces in Phoenix

### Access the UI

Open your browser to: http://localhost:6006

### Navigate Traces

1. **Projects**: Select "nas-agent-graph" project
2. **Traces**: View all traces chronologically
3. **Spans**: Click on any trace to see detailed span hierarchy

### Filter and Search

- **By Operation**: Filter by "planner_agent", "executor", "evaluator"
- **By Status**: Filter by success/error
- **By Time**: Filter by time range
- **By Attributes**: Search by run_id, iteration, decision, etc.

### Analyze Performance

Phoenix provides:
- **Latency Distribution**: See which operations are slowest
- **Error Rate**: Track failures over time
- **Token Usage**: Monitor LLM costs
- **Trace Timeline**: Visualize operation sequences

## Example Trace Hierarchy

A complete NAS iteration creates this trace structure:

```
NAS Iteration 0 [8m 45s]
│
├─ planner_agent [5.2s]
│  ├─ agent_state_loaded
│  ├─ llm_invocation_start
│  ├─ LLM Call: anthropic/claude-3.5-sonnet [5.0s]
│  │  ├─ Input: 2,450 tokens
│  │  └─ Output: 125 tokens
│  └─ decision_made
│
├─ executor [8m 35s]
│  ├─ apply_config_changes [0.2s]
│  │  └─ changes: 3 modifications
│  ├─ config_modified
│  ├─ training_run [8m 34s]
│  │  └─ run_id: run_0000
│  └─ training_completed
│
└─ evaluator [5.1s]
   ├─ metrics_loaded
   │  └─ val_mse: 0.185500
   ├─ deterministic_evaluation [0.1s]
   └─ decision_made
      └─ accept (deterministic)
```

## Advanced Usage

### Custom Spans in Extensions

If you extend the system, you can add custom tracing:

```python
from nas_agent_graph.phoenix_tracing import trace_operation, add_span_attribute, add_span_event

def my_custom_operation():
    with trace_operation("custom_op", {"param": "value"}):
        # Your code here
        add_span_attribute("result", "success")
        add_span_event("milestone_reached", {"data": 123})
```

### Programmatic Access

Check if Phoenix is enabled:

```python
from nas_agent_graph.phoenix_tracing import is_phoenix_enabled

if is_phoenix_enabled():
    print("Phoenix tracing is active")
```

## Troubleshooting

### Issue: "Phoenix tracing unavailable"

**Solution:**
```bash
pip install arize-phoenix openinference-instrumentation-langchain
```

### Issue: Traces not appearing in Phoenix UI

**Checks:**
1. Is Phoenix server running? `curl http://localhost:6006`
2. Check port in `.env`: `PHOENIX_PORT=6006`
3. Check firewall settings
4. Verify traces endpoint: `curl http://localhost:6006/v1/traces`

### Issue: "Connection refused"

**Solution:**
```bash
# Start Phoenix server
python -m phoenix.server.main serve

# Or run in background
python -m phoenix.server.main serve &
```

### Issue: Want to use different Phoenix instance

**Solution:**
```bash
# In .env file
PHOENIX_HOST=my-phoenix-server.com
PHOENIX_PORT=6006
```

## Performance Impact

Phoenix tracing has minimal overhead:

- **Latency**: ~1-5ms per span
- **Memory**: ~1-2MB per 1000 spans
- **Network**: ~1-5KB per span (async)

The tracing is asynchronous and won't block your NAS workflow.

## Data Privacy

Phoenix is **self-hosted** and runs locally:

- ✅ No data sent to external services
- ✅ All traces stored locally
- ✅ Full control over retention
- ✅ Can run completely offline

## Monitoring Best Practices

### 1. Track Key Metrics

Monitor these across iterations:
- Planner decision latency
- Training success rate
- Evaluation method distribution (deterministic vs LLM)
- LLM token usage and cost

### 2. Identify Bottlenecks

Look for:
- Slow LLM calls (> 10s)
- Training failures (error spans)
- Frequent LLM evaluations (borderline cases)

### 3. Optimize Based on Traces

Use traces to:
- Identify which config changes lead to success
- Find patterns in accepted vs rejected runs
- Optimize prompts based on LLM performance
- Adjust deterministic rules to reduce LLM calls

## Integration with Phoenix Features

### Experiments

Tag different NAS sessions:

```python
# Future enhancement
from phoenix.trace import using_project

with using_project("nas-experiment-1"):
    run_nas_optimization(max_iterations=20)
```

### Datasets

Export successful runs for analysis:

```python
# Future enhancement
from phoenix.trace import TraceDataset

dataset = TraceDataset.from_project("nas-agent-graph")
dataset.filter(lambda t: t.get_attribute("decision") == "accept")
dataset.export("successful_runs.jsonl")
```

## Summary

Phoenix integration provides:

✅ **Full Observability**: See every decision and operation
✅ **Performance Monitoring**: Track latency and costs
✅ **Error Tracking**: Quickly identify and debug issues
✅ **Cost Analysis**: Monitor LLM token usage
✅ **Pattern Discovery**: Find what works in NAS
✅ **Self-Hosted**: Complete data privacy and control

Access your traces at: **http://localhost:6006**
