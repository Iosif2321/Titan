# Titan — системная спецификация (полный итог обсуждения)

Документ фиксирует все согласованные решения и требования, включая архитектуру,
метрики, протоколы оценки, логику сигналов, систему наград/штрафов, хранение
и поиск паттернов, а также правила масштабирования ансамбля.

---

## 0) Контекст и исходные требования

**Проект**: модернизация Titan.  
**Задача**: прогноз направления движения BTCUSDT в режиме Bybit Spot.  
**Система**: ансамбль взаимосвязанных моделей с двумя головами (UP/DOWN).  
**Горизонты**: 1m, 5m, 10m, 15m, 30m, 60m, прогноз на тот же горизонт.  
**Цель**: >= 75% directional accuracy (учитываются только UP/DOWN).  
**Режим обучения**: предобучение + online learning с постоянной адаптацией.  
**Частота online обновления**: каждую минуту, версионирование раз в час.  
**Ограничения**: GPU GTX 1060.  
**Базовый TF**: не обязан быть 1m, но по умолчанию 1m (настраивается).  
**Связность ансамбля**: модели видят предсказания друг друга (внутри и между TF).

В качестве референсов использовались локальные проекты:
- **Granit**: paired ensemble, MC Dropout, inter-model learning, reward/penalty.
- **Chronos**: multi-horizon, режимы рынка, калибровка, конфликты/uncertain.

---

## 1) Классы и метрики

### Классы
- **UP** — ожидаемое движение вверх на горизонте TF.
- **DOWN** — ожидаемое движение вниз на горизонте TF.
- **FLAT** — движение ниже порога (неразличимо).
- **UNCERTAIN** — модель/ансамбль не уверены или есть конфликт.

### Метрики
- **Directional accuracy** (UP/DOWN только) — основная метрика.
- **Coverage** — доля UP/DOWN прогнозов (не FLAT, не UNCERTAIN).
- **Consensus** — согласие моделей внутри TF и между TF.
- **Uncertainty** — неопределенность (MC Dropout std/entropy).
- **Streaks** — серии ошибок/успехов (по модели, по TF, глобально).
- **Latency** — время инференса.

---

## 2) Протоколы оценки (утверждено)

Для каждого TF используются:
- **Walk-forward expanding**: обучаемся на 1..t, валидируем t+1..t+k.
- **Walk-forward rolling**: обучаемся на последних N, валидируем на k.
- **Prequential**: predict -> сравнить -> обновить, метрика по последним N.

Отчеты по точности строятся только по направлению (UP/DOWN),
доля UNCERTAIN/FLAT анализируется отдельно (coverage).

---

## 3) Данные и горизонты

### Горизонт
Горизонт прогноза равен TF:
- 5m => прогноз на 5 минут вперед.
- 60m => прогноз на 60 минут вперед.

### Данные на старте (утверждено)
- **Confirmed candles** — для меток и базовых признаков.
- **Partial candles** — для раннего сигнала (обновление каждые 5–10с).
- **Trades/Orderbook** — не используем в первой итерации.

### Дальше (опционально)
- Trade-фичи в агрегатах 1s/5s.
- Top-of-book признаки (spread, imbalance).

Обоснование: trades/orderbook повышают сложность, нагрузку и риск лагов,
поэтому вводятся после стабилизации основного контура.

---

## 4) Ансамбль и масштабирование

### Стартовый состав (пары UP/DOWN на TF)
- 1m: 3 пары
- 5m: 3 пары
- 10m: 2 пары
- 15m: 2 пары
- 30m: 2 пары
- 60m: 2 пары

### Fallback (если latency высокое)
- 1m: 2 пары
- 5m: 2 пары
- 10m: 2 пары
- 15m: 1 пара
- 30m: 1 пара
- 60m: 1 пара

### Требования к конфигу
- Одним параметром меняется число моделей.
- Можно настраивать число моделей отдельно на каждом TF.
- Допускаются разные типы моделей (LSTM/Transformer/MLP).

---

## 5) Архитектура модели и коммуникация

### Две независимые головы
- Общий backbone, две головы (UP/DOWN).
- Головы должны предсказывать только “свой” класс.

### Связность
Модели видят друг друга следующим образом:
- UP и DOWN головы одной модели видят друг друга.
- Все UP головы внутри TF видят друг друга.
- Все DOWN головы внутри TF видят друг друга.
- Все UP головы между TF видят друг друга.
- Все DOWN головы между TF видят друг друга.

### Контекст
Общий context vector включает:
- вероятности и уверенности других голов;
- метрики их качества и streaks;
- TF-уровневый консенсус и uncertainty.

---

## 6) Логика сигналов: FLAT / UNCERTAIN / UP / DOWN

### FLAT (утверждено)
Гибридный порог:
```
flat_thr_tf = max(min_bp, quantile_q(|r_h|))
```
Где:
- **quantile_q** подстраивается под целевую долю FLAT (например 15–30%).
- **min_bp** не дает порогу “упасть” в низкой волатильности.

### UNCERTAIN (утверждено)
UNCERTAIN если хотя бы одно:
- max(p_up, p_down) < p_min
- |p_up - p_down| < margin
- consensus < consensus_min
- uncertainty > uncertainty_max

### Порядок решения
1) FLAT  
2) UNCERTAIN  
3) UP/DOWN по взвешенному голосу

---

## 7) Награды, штрафы и контроль направлений

### Принципы (утверждено)
- Уверенная ошибка штрафуется сильнее.
- Ошибка “не в своем направлении” (misdirection) штрафуется сильнее, чем FLAT.
- Серии ошибок усиливают штраф.
- Серии успехов снижают штраф.
- Если все модели TF ошиблись, а одна угадала — она получает бонус и приоритет.

### Пример формулы штрафа
```
base_loss = cross_entropy
conf_penalty = 1 + a * wrong * confidence
misdir_penalty = 1 + b * misdirection
streak_penalty = 1 + k * error_streak
tf_penalty = 1 + g * tf_error_streak
total_loss = base_loss * conf_penalty * misdir_penalty * streak_penalty * tf_penalty
```

### Бонусы
- “Неуверенный, но верный” прогноз — бонус.
- “Единственный верный в TF” — дополнительный бонус и рост веса.

### Долгоживущие веса (чтобы не “забывать”)
- Два EMA масштаба:
  - **w_short** — часы/дни.
  - **w_long** — недели.
- Итоговый вес: `w = 0.7*w_long + 0.3*w_short`.
- EWC/L2-якорь на часовые чекпоинты.
- Replay buffer с балансом старых и новых данных.

---

## 8) Online learning цикл

Каждая минута:
1) Получаем новую свечу/partial-обновления.
2) Делаем прогнозы по всем TF и всем моделям.
3) Проверяем результаты прошлых прогнозов.
4) Обновляем rewards/penalties и веса.
5) Обновляем паттерны.
6) Выполняем online шаги обучения.
7) Сохраняем состояние при необходимости.

### Диаграмма online-цикла

```text
[New candle / partial updates]
        |
        v
[Predict all TF + all models]
        |
        v
[Resolve previous outcomes]
        |
        v
[Rewards/Penalties + weight update]
        |
        v
[Pattern memory update]
        |
        v
[Online training steps]
        |
        v
[Checkpoint? (hourly)]
```

Каждый час:
- Версионирование и checkpoint моделей/весов.

---

## 9) Память паттернов (Pattern Memory)

### Определение
Паттерн — это память решений и рыночных условий, общая для всех моделей и TF.
Паттерн хранит:
- условия рынка;
- контекст моделей (их прогнозы/уверенность);
- решения и результаты;
- статистику успешности по моделям и по TF.

### Объем на один паттерн (утверждено)
- 10,000 последних решений (ring buffer).
- до 1,000,000 агрегированных решений.
- 1,000 экстремумов (лучшие/худшие) в полной форме.

### Экстремумы
Хранятся полностью, не агрегируются:
- DecisionRecord
- полный feature-вектор
- полный набор свечей/контекст

### DecisionRecord (бинарный формат)
- ts:uint64
- tf_id:uint16
- model_id:uint16
- head_id:uint8 (UP/DOWN)
- pred_class:uint8
- actual_class:uint8
- flags:uint8 (misdirection/conflict/uncertain/flat)
- p_up:float16
- p_down:float16
- confidence:float16
- reward:float16
- outcome_margin:float16

### Агрегация старых решений
Если ring buffer переполнен:
- старые решения переходят в агрегаты:
  - agg_count
  - sum_reward
  - sum_conf
  - sum_conf2
  - sum_margin
  - sum_correct
  - sum_wrong
  - sum_misdirection
  - sum_confident_wrong

Если agg_count достигает 1,000,000:
- используется leaky average (EMA) для долговременной памяти.

---

## 10) Хэширование и поиск паттернов

### Двойной 256-бит хэш (утверждено)
- **market_hash** — только рыночные фичи.
- **context_hash** — рыночные фичи + предсказания/уверенность моделей.

Оба хеша — SimHash 256 бит (фиксированные проекции).

### Порог совпадения
- Базовый: 90% совпадений (<= 25 бит расхождения).
- Порог адаптивный:
  - слишком много кандидатов — повышаем;
  - слишком мало — понижаем.

### Двухэтапный поиск (утверждено)
1) LSH по market_hash и context_hash.
2) Пересечение кандидатов.
3) Hamming-фильтр (>= 0.90).
4) Чтение полного вектора и точное сравнение (cosine/L2).
5) Фильтр по режимам (vol/trend/session).
6) Проверка статистики паттерна + contrarian.

### Диаграмма поиска паттернов

```text
[Features + model context]
        |
        v
[market_hash + context_hash]
        |
        v
[LSH lookup (market)]----\
                          +-->[Intersect candidates]
[LSH lookup (context)]---/
                          |
                          v
                 [Hamming filter >= 0.90]
                          |
                          v
                 [Load full vectors (top-K)]
                          |
                          v
          [Exact similarity + regime filter]
                          |
                          v
          [Pattern score + contrarian check]
                          |
                          v
                   [Best pattern(s)]
```

### Почему так
SimHash — подпись сходства, а не “сжатие всех данных”.
Противоположные паттерны обычно не проходят 90% порог.
Второй этап по полному вектору защищает от ложных совпадений.

---

## 11) Хранилище (LMDB + binary log)

### LMDB таблицы
- patterns_meta: pattern_id -> метаданные, хеши, статистика, offsets.
- lsh_market: band_key -> список pattern_id.
- lsh_context: band_key -> список pattern_id.

### Binary log (append-only)
- ring_buffer.bin
- extremes.bin
- feature_blobs.bin

### LSH banding
- 8 bands x 32 бита (256 бит).
- band_key = (band_index, band_bits).

---

## 12) Отсев слабых паттернов (culling)

### Байесовская оценка
```
p ~ Beta(alpha0 + correct, beta0 + wrong)
lb = 5% quantile(p)
```

Параметры (утверждено):
- N_min = 500
- p0 = базовая точность по TF + режим
- delta = 0.02–0.03

### Статусы
- active — участвует в поиске.
- probation — ослаблен, но не архив.
- archived — убран из индекса, оставлен в агрегатах.
- purged — удален при нехватке места.

### Contrarian
Если паттерн стабильно дает противоположные результаты:
- помечаем contrarian,
- используем с инверсией,
- не удаляем.

---

## 13) Лимит 5GB и очистка

Если общий размер > 5GB:
1) Архивируем слабые/редкие паттерны (убираем из индекса).
2) Сокращаем архив до агрегатов + экстремумов.
3) Если все еще > 5GB — удаляем худшие archived.

---

## 14) Защита от ложных совпадений

Чтобы не попадать на противоположные паттерны:
- требуем совпадения **market_hash** и **context_hash**;
- делаем вторую проверку по полному вектору;
- фильтруем по режиму рынка;
- учитываем contrarian-статус.

---

## 15) Параметры по умолчанию (утверждено)

- N_min = 500
- p0 = базовая точность по TF + режим
- delta = 0.02–0.03
- similarity threshold = 0.90 (<= 25 бит), адаптивно
- target candidates = 20–50
- disk cap = 5GB
- hash = dual 256-bit SimHash (market + context)
- storage = LMDB + binary log
- base TF = 1m (настраивается)
- update cadence = 1 минута
- checkpoint cadence = 1 час

### Примеры конфигурации (YAML)

```yaml
# config/base.yaml
project:
  name: Titan
  version: 1.0.0

data:
  symbol: BTCUSDT
  source: bybit_spot
  base_timeframe: "1m"
  horizons: ["1m", "5m", "10m", "15m", "30m", "60m"]
  confirmed_candles: true
  partial_candles:
    enabled: true
    update_seconds: 5

evaluation:
  walk_forward:
    mode: expanding  # or rolling
    window_days: 30
  prequential:
    window: 500

prediction:
  min_confidence: 0.55
  margin: 0.10
  min_consensus: 0.70
  uncertainty_max: 0.08

flat:
  min_bp: 3
  target_share: [0.15, 0.30]
  quantile_window: 2000

online:
  interval_minutes: 1
  steps_per_candle: 3
  checkpoint_hours: 1
```

```yaml
# config/ensemble.yaml
ensemble:
  pairs_per_tf:
    "1m": 3
    "5m": 3
    "10m": 2
    "15m": 2
    "30m": 2
    "60m": 2
  model_types: ["lstm", "transformer"]
  mc_dropout:
    passes_default: 7
    passes_min: 3
    passes_max: 10
    latency_budget_ms: 8000

communication:
  intra_model: true
  intra_tf: true
  inter_tf: true

weights:
  short_half_life_hours: 12
  long_half_life_days: 21
  mix_long: 0.7
  mix_short: 0.3
  min_weight: 0.05
  max_weight: 5.0

rewards:
  misdirection_weight: 2.0
  confident_error_weight: 1.0
  streak_weight: 0.1
  tf_streak_weight: 0.1
  lone_winner_bonus: 0.2
```

```yaml
# config/patterns.yaml
patterns:
  storage:
    lmdb_path: artifacts/patterns/patterns.lmdb
    log_dir: artifacts/patterns/logs
    disk_cap_gb: 5

  hash:
    type: simhash
    bits: 256
    bands: 8
    band_bits: 32
    dual_hash: true

  search:
    similarity_min: 0.90
    adaptive_step: 0.02
    target_candidates_min: 20
    target_candidates_max: 50
    full_vector_metric: cosine
    regime_filter: true

  limits:
    ring_buffer: 10000
    aggregates_max: 1000000
    extremes_max: 1000

  culling:
    n_min: 500
    p0_scope: tf+regime
    delta: 0.02
    contrarian: true
```

---

## 16) Резюме ключевых договоренностей

- Точность измеряется только по UP/DOWN, coverage учитывается отдельно.
- FLAT определяется гибридным порогом (min_bp + quantile).
- UNCERTAIN определяется через уверенность, маржу, консенсус и uncertainty.
- Ансамбль масштабируется по TF и настраивается через конфиг.
- Головы UP/DOWN строго контролируются штрафами за misdirection.
- Память паттернов хранит полный контекст и экстремумы.
- Поиск паттернов — dual hash + Hamming + full-vector check.
- Хранилище — LMDB + binary log, лимит 5GB.
- Слабые паттерны архивируются, contrarian паттерны используются инвертировано.
