# BTCUSDT Adaptive Oracle (Bybit Spot) — **прогнозирование без торговли**

Система **прогнозирования направления и "различимости движения"** для `BTCUSDT` (Bybit **Spot**), построенная как **адаптивный онлайн‑ансамбль** с **паттерн‑памятью** и "самовоспитанием" через **награды/штрафы**.

- ✅ **Только прогнозы** (без ордеров, без торговли)
- ✅ **Постоянная адаптивность** (online learning + drift control)
- ✅ **Запоминание решений** (Decision Ledger + Pattern Memory + Replay)
- ✅ **Скорость** (потоковый inference, кэш, асинхронный I/O, < 10с на цикл)
- ✅ **Масштабируемость** (монолитное ядро → легко нарезать на сервисы)

## Быстрый старт

```bash
# 1) установка
pip install -e .

# 2) скачать историю (например 30 дней)
python scripts/download_history.py --days 30 --symbol BTCUSDT

# 3) (опционально) pretrain на 30 днях (можно пропустить)
python scripts/pretrain_30d.py --config config/default.yaml

# 4) live режим
python scripts/run_live.py --config config/default.yaml

# 5) API/дашборд
# http://localhost:8000
```

## Структура проекта

См. README в корне проекта для полной документации.

