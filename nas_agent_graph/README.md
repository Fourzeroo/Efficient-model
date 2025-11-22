# LangGraph NAS Agent System

LLM-оркестрация Neural Architecture Search для модели Informer.

## Обзор

Автоматическая система оптимизации архитектуры на основе LLM-агентов. Использует:

- **LangGraph** для оркестрации воркфлоу
- **OpenRouter** для доступа к LLM API (Claude 3.5 Sonnet)
- **Pydantic** для структурированных ответов LLM
- **Гибридная оценка** (детерминированные правила + LLM)

## Архитектура

```
┌────────────────────────────────────────────────┐
│              LangGraph Workflow                 │
├────────────────────────────────────────────────┤
│                                                 │
│  Planner → Executor → Evaluator → Update       │
│   (LLM)     (Python)    (Hybrid)      ↓        │
│     ↑                                  ↓        │
│     └──────────────── Loop ────────────┘        │
│                                                 │
└────────────────────────────────────────────────┘
```

### Компоненты

1. **Planner (LLM)** - Анализирует историю, решает, что изменить
2. **Executor (Python)** - Применяет изменения, запускает обучение
3. **Evaluator (Hybrid)** - Оценивает результаты (правила + LLM)
4. **State Manager** - Отслеживает все эксперименты

## Установка

```bash
# Установить зависимости
pip install -r requirements.txt

# Создать .env файл в корне проекта
echo "OPENROUTER_API_KEY=your_key_here" > ../.env
```

Получить API ключ: https://openrouter.ai/

## Использование

### Python API

```python
from nas_agent_graph import run_nas_optimization

run_nas_optimization(max_iterations=20)
```

### Командная строка

```bash
python -m nas_agent_graph.main --max-iterations 20 --verbose
```

## Процесс работы

### Одна итерация NAS

1. **Planner**: Анализирует контекст (лучший запуск, последние 5, конфиг)
2. **Planner**: Решает что изменить и возвращает изменения
3. **Executor**: Модифицирует `config.py` и запускает обучение
4. **Evaluator**: Оценивает результаты (правила → LLM при неясности)
5. **Update**: Обновляет `agent_state.json`, best_run_id
6. **Loop/Stop**: Продолжает или останавливается

### Типы изменений

```python
# Гиперпараметры
{
    "training.optimizer_params.lr": 0.0001,
    "model.dropout": 0.3,
    "data.batch_size": 32,
}

# Архитектура
{
    "model.d_model": 256,
    "model.n_heads": 8,
    "model.e_layers": 3,
}

# Оптимизатор/Loss
{
    "training.optimizer_class": "AdamW",
    "training.criterion_class": "SmoothL1Loss",
}
```

## Структура файлов

```
nas_agent_graph/
├── __init__.py              # Экспорты
├── main.py                  # Точка входа
├── config.py                # Конфигурация (API ключи)
├── graph.py                 # LangGraph воркфлоу
├── planner.py               # LLM-агент планирования
├── executor.py              # Применение изменений
├── evaluator.py             # Гибридная оценка
├── prompts.py               # Шаблоны промптов
├── phoenix_tracing.py       # Трейсинг (опционально)
└── requirements.txt         # Зависимости
```

## Результаты

### Agent State (`nas_system/agent_state.json`)

```json
{
  "best_run_id": "run_0003",
  "runs": [
    {
      "run_id": "run_0000",
      "val_mse": 0.1855,
      "test_mse": 0.1923,
      "accepted": true
    }
  ]
}
```

### Результаты запусков (`nas_system/runs/run_XXXX/`)

- `metrics.json` - Финальные метрики
- `history.json` - История по эпохам
- `learning_curve.png` - Визуализация
- `best_model.pth` - Чекпоинт
- `config_used.py` - Снапшот конфига

## Дополнительно

### Возобновление работы

Система автоматически продолжает с последней итерации:

```python
# Первая сессия: 0-9
run_nas_optimization(max_iterations=10)

# Вторая сессия: продолжает с 10
run_nas_optimization(max_iterations=20)
```

### Другие модели LLM

```bash
# В .env
PLANNER_MODEL=anthropic/claude-3-opus
EVALUATOR_MODEL=openai/gpt-4-turbo
```

Поддерживаются (через OpenRouter):
- `anthropic/claude-3.5-sonnet` (по умолчанию)
- `anthropic/claude-3-opus` (дороже, мощнее)
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo` (дешевле)

### Отладка

```python
import nas_agent_graph.config as cfg
cfg.VERBOSE = True
cfg.DEBUG = True
```

## Устранение неполадок

### Ошибки импорта

```bash
pip install -r requirements.txt
```

### API ключ не найден

Создать `.env` в корне проекта:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key
```

### Ошибки обучения

```bash
cd nas_system
python train.py --config config.py --tag test
```

### Конфиг не изменяется

1. Проверить формат `config.py`
2. Включить DEBUG режим
3. Начать с простых изменений

## Принципы дизайна

1. **Простота** - прямое выполнение Python, без MCP
2. **Надежность** - Pydantic для LLM, правила для ясных случаев
3. **Прозрачность** - подробное логирование, сохранение состояния
4. **Модульность** - каждый компонент с одной ответственностью

## Сравнение с `nas_system`

| Функция | `nas_system` | `nas_agent_graph` |
|---------|--------------|-------------------|
| LLM | Внешний | Встроенный |
| Воркфлоу | Ручной | Автоматический |
| Решения | Человек | LLM |
| Оценка | Человек | Гибридная |

`nas_agent_graph` строится поверх `nas_system` для полной автоматизации.

## Лицензия

Та же, что и у родительского проекта.
