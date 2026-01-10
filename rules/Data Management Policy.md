---
type: "agent_requested"
description: "Rules for handling data processing scripts and data storage. Emphasizes data immutability, using efficient file formats like Parquet, and preferring Polars for I/O operations. Applies to `scripts/` manipulating data and `data_*/` directories."
---
### 💾 Правила Работы с Данными

1.  **Данные неизменяемы (Immutable).** Скрипты никогда не изменяют исходные файлы. Результаты очистки или трансформации всегда сохраняются в новый каталог (например, из `data_raw` в `data_clean`).
2.  **Parquet и Polars — наш стандарт.** Для хранения используй Parquet. Для I/O операций и обработки данных в скриптах предпочитай Polars, так как он значительно быстрее `pandas`.