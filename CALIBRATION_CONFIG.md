# Система конфигурации калибровки

## Обзор

Адаптивная система калибровки теперь поддерживает гибкую настройку через JSON конфигурационные файлы с:
- **Глобальными параметрами** для всех моделей
- **Индивидуальными настройками** для каждой модели (TRENDVIC, OSCILLATOR, VOLUMEMETRIX)

## Быстрый старт

### 1. Использование готового конфига

```bash
# Базовый конфиг (дефолтные параметры)
.venv/bin/python -m src.offline_replay \
  --symbol BTCUSDT --tf 1 --minutes 240 \
  --mode train \
  --fact-flat-mode adaptive \
  --pred-flat-mode adaptive \
  --reward-mode shaped \
  --run-dir runs/test_adaptive \
  --calib-config config/calibration_default.json

# Агрессивный конфиг для OSCILLATOR (решение коллапса параметров)
.venv/bin/python -m src.offline_replay \
  --symbol BTCUSDT --tf 1 --minutes 240 \
  --mode train \
  --fact-flat-mode adaptive \
  --pred-flat-mode adaptive \
  --reward-mode shaped \
  --run-dir runs/oscillator_fix \
  --calib-config config/oscillator_aggressive.json
```

### 2. Production использование

```bash
python -m src.main \
  --symbol BTCUSDT \
  --tfs 1,5,15 \
  --calib-config config/oscillator_aggressive.json
```

## Доступные конфиги

| Файл | Назначение |
|------|------------|
| `config/calibration_default.json` | Базовый конфиг с дефолтами |
| `config/calibration_example.json` | Пример с per-model настройками |
| `config/oscillator_aggressive.json` | Решение проблемы коллапса OSCILLATOR |

Подробности в `config/README.md`

## Структура конфига

```json
{
  "global": {
    "calib_lr": 0.005,
    "calib_lr_min": 0.0001,
    "calib_lr_max": 0.02,
    "calib_ece_target": 0.05,
    "calib_a_min": 0.30,
    "calib_a_max": 2.0
  },
  "per_model": {
    "OSCILLATOR": {
      "lr": 0.010,
      "a_min": 0.15,
      "a_max": 3.5,
      "lr_max": 0.05,
      "ece_target": 0.10
    }
  }
}
```

## Интеграция с системой

### Где работает адаптивная калибровка

Адаптивная система **уже интегрирована в ядро** проекта:

1. **ModelRunner** (src/engine.py:162-168):
   ```python
   self.calib_controller = AdaptiveCalibController.create(
       model_type=model_type,
       training_config=training,
       per_model_config=per_model_cfg
   )
   ```

2. **Предсказания** (src/engine.py:281):
   ```python
   p_up, p_down = calibrate_from_logits(
       logit_up, logit_down,
       self.calib_controller.state,
       flip=self.direction_flip
   )
   ```

3. **Обновление при фактах** (src/engine.py:511-519):
   ```python
   ece = self.calibration.get_recent_ece(window_size=50)
   self.calib_controller.update(
       logit_up=pending.logits_up,
       logit_down=pending.logits_down,
       y_up=y_up,
       weight=calib_weight,
       flip=self.direction_flip,
       current_ece=ece  # ECE обратная связь!
   )
   ```

4. **Сохранение состояния** (src/engine.py:661):
   ```python
   "calibration_adaptive": self.calib_controller.to_dict()
   ```

### Что НЕ нужно делать

- ❌ Не нужно изменять код для использования адаптивной системы
- ❌ Не нужно ничего включать/выключать
- ❌ Не нужно вызывать специальные методы

### Что нужно делать

- ✅ Просто передать `--calib-config` при запуске
- ✅ Система автоматически применит настройки
- ✅ Состояние сохраняется и восстанавливается

## Решение проблем

### Проблема: OSCILLATOR коллапс параметров

**Симптомы**:
- `calib_a = 0.300` (застрял на минимуме)
- `ECE > 0.50` (серьёзная переуверенность)
- `confident_wrong > 40%`

**Решение**:
```bash
--calib-config config/oscillator_aggressive.json
```

**Что делает**:
- Расширяет границы: `a_min=0.15`, `a_max=3.5`
- Повышает lr: `lr=0.01`, `lr_max=0.05`
- Начинает с `init_a=0.50` вместо 1.0

### Проблема: Слишком быстрая адаптация

**Симптомы**: lr прыгает, параметры нестабильны

**Решение**: Увеличить интервал адаптации
```json
{
  "global": {
    "calib_adaptation_interval": 50,
    "calib_ece_window_size": 100
  }
}
```

### Проблема: Недостаточная адаптация

**Симптомы**: ECE остаётся высоким, lr не увеличивается

**Решение**: Более агрессивные факторы
```json
{
  "global": {
    "calib_lr_increase_factor": 2.0,
    "calib_lr_max": 0.05
  }
}
```

## Мониторинг

### При запуске проверьте логи:
```
INFO Loaded calibration config from: config/oscillator_aggressive.json
INFO Per-model configs: ['OSCILLATOR', 'VOLUMEMETRIX', 'TRENDVIC']
```

### В отчёте смотрите:

**Calibration Evolution**:
```
- a: initial=1.000 final=0.660 min=0.653 max=1.002
- b: initial=0.000 final=0.032 min=-0.022 max=0.056
```
- `final`: текущее значение параметра
- `min/max`: диапазон за сессию
- Если `final = min` → возможен коллапс

**Calibration Metrics**:
```
- calibration: ece=0.301 mce=0.566 brier=0.336
- confident_wrong_rate: 0.067
```
- `ece < 0.10`: хорошая калибровка
- `ece 0.10-0.25`: приемлемо
- `ece > 0.25`: плохая калибровка
- `confident_wrong < 10%`: хорошо

## Создание своего конфига

1. Скопируйте `config/calibration_example.json`
2. Измените параметры под свои нужды
3. Протестируйте на offline_replay
4. Используйте в production

**Пример**: Создать конфиг для более консервативной адаптации

```json
{
  "global": {
    "calib_lr_increase_factor": 1.2,
    "calib_lr_decrease_factor": 0.95,
    "calib_adaptation_interval": 30,
    "calib_ece_window_size": 75
  },
  "per_model": {}
}
```

## Технические детали

### Алгоритм адаптации

Каждые `calib_adaptation_interval` обновлений:

```python
if ece > calib_ece_bad_threshold:
    lr *= calib_lr_increase_factor  # Ускориться
elif ece < calib_ece_good_threshold:
    lr *= calib_lr_decrease_factor  # Замедлиться
else:
    # Проверить тренд, мягкая корректировка
    pass

lr = clamp(lr, calib_lr_min, calib_lr_max)
```

### Обновление параметров (SGD)

```python
# Градиент
grad_a = -weight * (y_up - 0.5) * m_raw + l2_a * (a - 1.0)
grad_b = -weight * (y_up - 0.5) + l2_b * b

# Обновление
a_new = clamp(a - lr * grad_a, a_min, a_max)
b_new = clamp(b - lr * grad_b, b_min, b_max)
```

## См. также

- `config/README.md` - подробная документация конфигов
- `src/adaptive_calibration.py` - реализация адаптивного контроллера
- `tests/test_adaptive_calibration.py` - unit тесты
