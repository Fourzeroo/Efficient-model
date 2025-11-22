# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running a Single Training

```bash
python train.py --config config.py --tag run_0000
```

This will create `runs/run_0000/` with:
- `config_used.py` - Snapshot of configuration
- `metrics.json` - Final metrics
- `history.json` - Per-epoch training history  
- `learning_curve.png` - Visualization
- `best_model.pth` - Model checkpoint

## Using the NAS Agent Library

### Load and Analyze Results

```python
from pathlib import Path
from nas_agent import load_metrics, load_history, build_history_summary

# Load results
metrics = load_metrics(Path("runs/run_0000"))
history = load_history(Path("runs/run_0000"))

# Generate textual summary
summary = build_history_summary(history)
print(summary)

print(f"Val MSE: {metrics['val_mse']}")
print(f"Test MSE: {metrics['test_mse']}")
```

### Manage Global NAS State

```python
from nas_agent import (
    load_agent_state, 
    save_agent_state, 
    add_run,
    get_best_run,
    RunInfo
)

# Load or create state
state = load_agent_state()

# Add a run
run_info = RunInfo(
    run_id="run_0000",
    val_mse=0.324,
    test_mse=0.341,
    history_summary=summary,
    accepted=True
)
state = add_run(state, run_info)
save_agent_state(state)

# Query best run
best = get_best_run(state)
print(f"Best: {best.run_id} with val_mse={best.val_mse}")
```

### Modify Configuration

Конфигурация теперь в формате Python для максимальной гибкости:

```python
# config.py

import torch.optim as optim
import torch.nn as nn

# Model architecture - можно менять любые параметры
model = {
    "d_model": 256,      # Изменили
    "n_heads": 8,        # Изменили
    "e_layers": 2,       # Изменили
    # ... остальные параметры
}

# Training - можно менять классы оптимизаторов, schedulers, loss
training = {
    # Можно использовать любой оптимизатор из torch.optim
    "optimizer_class": optim.AdamW,  # Изменили на AdamW
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 0.01,  # Добавили регуляризацию
    },
    
    # Можно использовать любую loss функцию
    "criterion_class": nn.SmoothL1Loss,  # Изменили loss
    "criterion_params": {},
    
    # Можно использовать любой scheduler
    "scheduler_class": optim.lr_scheduler.StepLR,  # Изменили scheduler
    "scheduler_params": {"step_size": 5, "gamma": 0.5},
}

# После изменения просто запустите:
# python train.py --config config.py --tag run_0001
```

## Running the Example Agent

To see a complete NAS workflow simulation:

```bash
python example_agent.py
```

This will:
1. Run multiple training iterations
2. Modify config between runs
3. Analyze results and update global state
4. Show LLM-style reasoning about architecture choices

## Typical LLM Agent Workflow

1. **Read current state:**
   ```python
   state = load_agent_state()
   best = get_best_run(state)
   recent = get_recent_runs(state, k=5)
   ```

2. **Analyze trends** (LLM reasoning):
   - "Recent runs show val_mse plateauing around 0.35"
   - "Best run has small model (d_model=256)"
   - "Larger models show overfitting"

3. **Decide on next architecture:**
   - "Try deeper encoder with regularization"

4. **Modify config:**
   ```python
   config["model"]["n_encoder_layers"] = 4
   config["model"]["dropout"] = 0.2
   ```

5. **Run training:**
   ```bash
   python train.py --config config.yaml --tag run_0005
   ```

6. **Analyze results:**
   ```python
   metrics = load_metrics(Path("runs/run_0005"))
   summary = build_history_summary(load_history(Path("runs/run_0005")))
   ```

7. **Update state and repeat:**
   ```python
   run_info = RunInfo(...)
   state = add_run(state, run_info)
   save_agent_state(state)
   ```

## Key Files

- `config.py` - **Modify this** to change architecture, optimizer, scheduler, loss
- `model.py` - **Modify this** to change model implementation
- `train.py` - Training pipeline (usually no need to modify)
- `nas_agent/` - Library functions (usually no need to modify)
- `runs/` - Output directory (auto-created)
- `agent_state.json` - Global state (auto-created/updated)

## Advantages of Python Config

**Старый формат (YAML):**
```yaml
training:
  optimizer: "adam"  # Только строка
  learning_rate: 1e-4
```

**Новый формат (Python):**
```python
training = {
    "optimizer_class": optim.Adam,  # Прямая ссылка на класс
    "optimizer_params": {"lr": 1e-4},
    
    # Можно использовать ЛЮБОЙ оптимизатор из PyTorch или кастомный
    # "optimizer_class": optim.AdamW,
    # "optimizer_class": optim.SGD,
    # "optimizer_class": MyCustomOptimizer,
}
```

**Преимущества:**
- ✅ Не ограничены предопределенным списком оптимизаторов
- ✅ Можно использовать любые классы из PyTorch
- ✅ Можно добавлять кастомные классы
- ✅ LLM агент не ограничен в выборе

## Tips for LLM Agents

1. Always read `agent_state.json` before deciding next architecture
2. Use `build_history_summary()` to understand learning dynamics
3. Compare recent runs to identify trends
4. Accept/reject runs based on validation performance
5. Track best configuration in `agent_state.json`
