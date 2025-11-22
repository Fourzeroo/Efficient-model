"""
Демонстрация работы обновленной системы с отслеживанием config_changes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nas_system.nas_agent import load_agent_state
from nas_agent_graph.prompts import format_recent_runs

print("=" * 80)
print("ДЕМОНСТРАЦИЯ: Отслеживание изменений конфигурации в NAS Agent")
print("=" * 80)

# Загружаем состояние агента
agent_state_path = Path("nas_system/agent_state.json")
state = load_agent_state(agent_state_path)

print(f"\nВсего запусков: {len(state.runs)}")
print(f"Лучший запуск: {state.best_run_id}")

# Показываем форматированный вывод для планировщика
print("\n" + "=" * 80)
print("ПРОМПТ ДЛЯ ПЛАНИРОВЩИКА (последние 3 запуска)")
print("=" * 80)
print()

formatted_output = format_recent_runs(state.runs, k=3)
print(formatted_output)

print("\n" + "=" * 80)
print("КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
print("=" * 80)
print("""
1. ✓ Планировщик теперь видит конкретные изменения конфигурации для каждого запуска
2. ✓ Можно отследить, какие параметры уже пробовались
3. ✓ Легче избежать повторения неудачных экспериментов
4. ✓ Лучшее понимание влияния параметров на метрики
5. ✓ Обратная совместимость со старыми данными
""")

print("\nПримечание: Старые запуски (run_0000-run_0004) показывают '(not recorded)',")
print("так как были созданы до добавления этой функциональности.")
print("Новые запуски будут автоматически сохранять config_changes.")
