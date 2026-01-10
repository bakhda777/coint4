---
type: "agent_requested"
description: "Best practices for managing YAML configuration files in the `configs/` directory. Rules for defining a single source of truth (`main_2024.yaml`), avoiding duplication, and ensuring all parameters are clearly documented with comments."
---
### ⚙️ Правила Конфигурации (для `configs/`)

1.  **`main_2024.yaml` — источник правды.** Все остальные `.yaml` файлы (для тестов, отладки) должны только переопределять необходимые параметры, а не дублировать всю структуру.
2.  **Комментируй всё неочевидное.** Каждый параметр в `.yaml` должен иметь комментарий, объясняющий его назначение и влияние на стратегию.