# LLM-based NAS Agent for Time Series Forecasting

Система автоматического улучшения архитектуры модели на основе LLM для задачи прогнозирования временных рядов (Informer на датасете ETTh1).

## Обзор

Эта система предоставляет инструментарий для LLM-агента, который может:
- Запускать тренировочные скрипты
- Изменять конфигурацию модели (гиперпараметры)
- Изменять архитектуру модели
- Итеративно улучшать метрики (MSE)

**Важно:** Система НЕ включает саму LLM. LLM работает извне (Claude, Cursor, Windsurf) и взаимодействует через файлы и командную строку.

## Структура проекта

```
nas_system/
├── model.py              # Реальная модель Informer (может изменяться LLM)
├── config.yaml           # Конфигурация (гиперпараметры, изменяется LLM)
├── train.py              # Тренировочный скрипт (НЕ изменяется LLM)
├── nas_agent/            # Библиотека для управления NAS
│   ├── run_manager.py      # Управление директориями запусков
│   ├── logging_utils.py    # Сохранение/загрузка метрик
│   ├── history_summary.py  # Текстовые резюме кривых обучения
│   └── agent_state.py      # Глобальное состояние NAS
├── runs/                 # Результаты запусков (создается автоматически)
│   ├── run_0000/
│   │   ├── config_used.yaml
│   │   ├── metrics.json
│   │   ├── history.json
│   │   ├── learning_curve.png
│   │   └── best_model.pth
│   └── run_0001/
│       └── ...
└── agent_state.json      # Глобальное состояние NAS (создается автоматически)
```

## Основные компоненты

### 1. Тренировочный пайплайн (`train.py`)

Полный цикл обучения на PyTorch:
- Загрузка реального датасета ETTh1
- Использование модели Informer из `Informer2020`
- Early stopping (терпение настраивается)
- Оптимизаторы: Adam, AdamW, SGD
- Schedulers: Step, Cosine, ReduceLROnPlateau
- Сохранение метрик и чекпоинтов
- Визуализация кривых обучения

**Использование:**
```bash
python train.py --config config.yaml --tag run_0000
```

**Важно:** LLM агент НЕ должен изменять логику обучения в `train.py` (функции `train_epoch`, `validate`).

### 2. NAS Agent Library (`nas_agent/`)

A Python library providing utilities for:

#### Run Management (`run_manager.py`)
- `get_run_dir(tag)`: Create/get run directory
- `snapshot_config(config_path, run_dir)`: Snapshot configuration
- `ensure_runs_root()`: Ensure runs directory exists

#### Logging (`logging_utils.py`)
- `save_metrics(run_dir, metrics)`: Save metrics to JSON
- `save_history(run_dir, history)`: Save training history to JSON
- `load_metrics(run_dir)`: Load metrics from JSON
- `load_history(run_dir)`: Load training history from JSON

#### History Summarization (`history_summary.py`)
- `build_history_summary(history)`: Generate textual summaries of learning curves for LLM consumption

Example output:
```
30 epochs, best_epoch=17. train_mse decreased from 1.02 to 0.24. 
val_mse started at 1.05, reached minimum of 0.32 at epoch 17, 
ended at 0.40. This suggests overfitting: validation loss starts 
rising after epoch 17, increasing from 0.32 to ~0.40 by epoch 29. 
The train/val gap widened from 0.03 to 0.16, indicating potential 
overfitting.
```

#### Agent State Management (`agent_state.py`)
- `load_agent_state()`: Load global NAS state
- `save_agent_state(state)`: Save global NAS state
- `add_run(state, run_info)`: Add new run and update best
- `get_best_run(state)`: Get best performing run
- `get_recent_runs(state, k)`: Get k most recent runs

### 3. Модель (`model.py`)

Реальная модель Informer из репозитория Informer2020:
- ProbSparse attention механизм
- Transformer encoder-decoder структура
- Функция `build_model(config)` для создания модели из конфига
- **LLM может изменять параметры модели через `config.yaml`**
- **LLM может изменять сам код модели в `model.py` для исследования новых архитектур**

### 4. Конфигурация (`config.py`)

Python файл с настройками (максимальная гибкость для LLM):
- Пути к датасету и preprocessing
- **Архитектура модели** (d_model, n_heads, n_layers, etc.) - **изменяется LLM**
- **Классы оптимизаторов** (optim.Adam, optim.AdamW, optim.SGD) - **изменяется LLM**
- **Классы schedulers** (CosineAnnealingLR, StepLR, ReduceLROnPlateau) - **изменяется LLM**
- **Классы loss функций** (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss) - **изменяется LLM**
- **Параметры оптимизации** (lr, weight_decay, betas, momentum) - **изменяется LLM**
- Early stopping параметры
- Настройки логирования

**Преимущества Python конфига:**
- Можно напрямую указывать классы PyTorch (не строки)
- Полная гибкость для LLM агента
- Можно добавлять кастомные оптимизаторы/schedulers

**Текущая конфигурация содержит лучшие параметры из Optuna оптимизации (val_mse=0.1855).**

## Установка

### Требования

- Python 3.10+
- PyTorch
- Зависимости: PyYAML, matplotlib, numpy, pandas, scikit-learn

```bash
pip install -r requirements.txt
```

### Настройка Informer2020

Система использует реальную модель Informer. Склонируйте репозиторий в родительской директории:

```bash
cd ..  # Перейти в Efficient-model/
git clone https://github.com/zhouhaoyi/Informer2020.git
cd nas_system
```

Структура должна быть:
```
Efficient-model/
├── Informer2020/          # Репозиторий с моделью
│   └── models/model.py
└── nas_system/            # Эта система NAS
    ├── train.py
    └── ...
```

## Быстрый старт

### 1. Запуск одного обучения

```bash
python train.py --config config.py --tag run_0000
```

Это:
1. Создаст директорию `runs/run_0000/`
2. Загрузит датасет ETTh1 с GitHub
3. Обучит модель Informer с early stopping
4. Сохранит результаты:
   - `metrics.json`: Финальные метрики (val_mse, test_mse, параметры модели)
   - `history.json`: История обучения по эпохам
   - `learning_curve.png`: Визуализация
   - `best_model.pth`: Чекпоинт лучшей модели
   - `config_used.py`: Копия конфигурации

**Ожидаемый результат:** val_mse ≈ 0.18-0.20 (базовая конфигурация из Optuna)

### 2. Изменение конфигурации

Конфиг теперь в формате Python для максимальной гибкости:

```python
# config.py

import torch.optim as optim
import torch.nn as nn

# Можно менять оптимизатор
training = {
    "optimizer_class": optim.AdamW,  # Или optim.SGD, optim.RMSprop
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),  # Для Adam/AdamW
    },
    
    # Можно менять loss функцию
    "criterion_class": nn.SmoothL1Loss,  # Или nn.MSELoss, nn.L1Loss
    "criterion_params": {"beta": 0.1},
    
    # Можно менять scheduler
    "scheduler_class": optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 5, "gamma": 0.5},
}
```

### 2. Using the NAS Agent Library

```python
from pathlib import Path
from nas_agent import (
    load_agent_state, save_agent_state, add_run,
    load_history, build_history_summary,
    RunInfo
)

# Load global state
state = load_agent_state(Path("agent_state.json"))

# Load run results
history = load_history(Path("runs/run_0000"))
summary = build_history_summary(history)

# Create run info
run_info = RunInfo(
    run_id="run_0000",
    val_mse=0.324,
    test_mse=0.341,
    history_summary=summary,
    accepted=True
)

# Update global state
state = add_run(state, run_info)
save_agent_state(state)

# Query state
best = get_best_run(state)
print(f"Best run: {best.run_id} with val_mse={best.val_mse}")
```

## Рабочий процесс LLM-агента

Внешние LLM-агенты взаимодействуют с системой через файлы и командную строку:

### 1. Изменить архитектуру
Отредактировать `config.py`:
```python
# Model architecture
model = {
    "d_model": 256,     # Уменьшили с 512
    "n_heads": 8,       # Уменьшили с 16
    "e_layers": 2,      # Уменьшили с 3
    "dropout": 0.2,     # Изменили регуляризацию
    # ... остальные параметры
}

# Training configuration - можно менять оптимизатор!
training = {
    "optimizer_class": optim.AdamW,  # Изменили с Adam на AdamW
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 0.01,  # Добавили weight decay
    },
    "criterion_class": nn.SmoothL1Loss,  # Изменили loss
    # ... остальные параметры
}
```

### 2. Запустить обучение
```bash
python train.py --config config.py --tag run_0001
```

### 3. Прочитать результаты
```python
from nas_agent import load_metrics, load_history, build_history_summary
from pathlib import Path

# Загрузить метрики
metrics = load_metrics(Path("runs/run_0001"))
print(f"Val MSE: {metrics['val_mse']:.6f}")
print(f"Test MSE: {metrics['test_mse']:.6f}")

# Получить текстовое резюме для LLM
history = load_history(Path("runs/run_0001"))
summary = build_history_summary(history)
print(summary)
```

### 4. Обновить глобальное состояние
```python
from nas_agent import load_agent_state, add_run, save_agent_state, RunInfo

state = load_agent_state()
run_info = RunInfo(
    run_id="run_0001",
    val_mse=metrics["val_mse"],
    test_mse=metrics["test_mse"],
    history_summary=summary,
    accepted=True  # Принять, если метрика улучшилась
)
state = add_run(state, run_info)
save_agent_state(state)
```

### 5. Принять решение на основе анализа
```python
from nas_agent import get_best_run, get_recent_runs

# Лучший запуск
best = get_best_run(state)
print(f"Лучший: {best.run_id} с val_mse={best.val_mse:.6f}")

# Анализ последних запусков
recent = get_recent_runs(state, k=5)
for run in recent:
    print(f"{run.run_id}: val_mse={run.val_mse:.4f}")

# LLM решает: попробовать другую архитектуру на основе трендов
```

## File Formats

### `metrics.json`
```json
{
  "best_epoch": 17,
  "val_mse": 0.324,
  "test_mse": 0.341,
  "train_mse": 0.245,
  "train_time_sec": 127.5,
  "n_params": 2458624
}
```

### `history.json`
```json
{
  "best_epoch": 17,
  "epochs": [
    {"epoch": 0, "train_mse": 1.02, "val_mse": 1.05, "lr": 0.0001},
    {"epoch": 1, "train_mse": 0.85, "val_mse": 0.89, "lr": 0.0001},
    ...
  ]
}
```

### `agent_state.json`
```json
{
  "max_runs": 100,
  "best_run_id": "run_0003",
  "runs": [
    {
      "run_id": "run_0000",
      "val_mse": 0.356,
      "test_mse": 0.372,
      "history_summary": "30 epochs, best_epoch=22...",
      "accepted": true
    },
    ...
  ]
}
```

## Принципы дизайна

### Для LLM-агентов
- **Файловая система**: Все состояние в JSON/YAML файлах
- **Изолированные запуски**: Каждый запуск в отдельной директории
- **Текстовые резюме**: Кривые обучения преобразованы в естественный язык
- **Четкие интерфейсы**: Документированные функции с type hints

### Для людей
- **Модульность**: Разделение обучения, логирования и управления состоянием
- **Расширяемость**: Легко добавить новые модели, датасеты, метрики
- **Воспроизводимость**: Снапшоты конфигураций и random seeds
- **Наблюдаемость**: Кривые обучения, метрики, чекпоинты

## Что НЕ включено

Эта система **НЕ реализует**:
- Вызовы LLM API (OpenAI, Anthropic, etc.)
- MCP клиенты или серверы
- Сам LLM multi-agent механизм

LLM работает извне и взаимодействует через:
- Чтение/запись файлов (YAML, JSON)
- Выполнение команд (`python train.py`)
- Использование библиотеки `nas_agent`

## Базовые метрики

С лучшими параметрами из Optuna:
- **Validation MSE**: 0.1855
- **Test MSE**: ~0.19
- **Параметры модели**: 4,932,103
- **Время обучения**: ~5-10 минут на GPU
- **Датасет**: ETTh1 (8640 train + 2880 val + 2880 test часов)

## Примеры использования

### Тестирование системы
```bash
python test_system.py
```

### Пример LLM-агента
Симуляция работы LLM-агента (3 итерации NAS):
```bash
python example_agent.py
```

## Расширение системы

### Добавить новую модель
Изменить `model.py` и `build_model()`:
```python
def build_model(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get("model", {}).get("type", "informer")
    if model_type == "informer":
        return build_informer(config)
    elif model_type == "transformer":
        return build_transformer(config)
```

### Добавить новый датасет
Изменить `config.yaml`:
```yaml
data:
  data_path: "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
```

### Добавить метрики
Расширить `validate()` в `train.py`:
```python
mae = torch.mean(torch.abs(outputs - batch_y))
metrics = {"mse": mse, "mae": mae}
```

## Устранение неполадок

### Ошибка импорта Informer
```
ImportError: No module named 'models.model'
```
**Решение**: Склонировать Informer2020 в родительской директории (см. раздел "Установка").

### CUDA Out of Memory
**Решение**: Уменьшить batch_size в `config.yaml`:
```yaml
data:
  batch_size: 32  # Было 64
```

### Разные результаты
Нормальное отклонение ±0.01-0.02 MSE. Для воспроизводимости установите `seed: 42` в config.

## Благодарности

- Архитектура Informer: [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- Датасет ETT: [Electricity Transformer Temperature dataset](https://github.com/zhouhaoyi/ETDataset)
- Репозиторий Informer2020: [https://github.com/zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020)

## Подробная документация

- **QUICKSTART.md** - Быстрый старт и примеры использования
- **SETUP_REAL.md** - Детальная инструкция по настройке системы
