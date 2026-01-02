# Анализ системы тестирования Titan

## 1. Архитектура текущей системы

### 1.1 Компоненты

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                            │
│  Commands: backtest, history, live                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    history.py / backtest.py                     │
│  - Загрузка данных (Bybit REST API)                             │
│  - Валидация gaps                                               │
│  - Prefill warmup                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FeatureStream                             │
│  Фичи: close, return_1, ma_fast/slow, volatility_z,            │
│        volume_z, rsi, ma_delta                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Models                                  │
│  TrendVIC │ Oscillator │ VolumeMetrix                          │
│     ▼           ▼            ▼                                  │
│  ModelOutput(prob_up, prob_down, state, metrics)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ensemble + RegimeDetector                    │
│  - Взвешивание моделей                                          │
│  - Адаптация по режиму                                          │
│  - Decision(direction, confidence, prob_up, prob_down)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BacktestStats + Evaluation                   │
│  - Сравнение prediction vs actual                               │
│  - Сбор метрик                                                  │
│  - Генерация отчётов                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Поток данных

1. **Входные данные**: OHLCV свечи (Bybit API или CSV)
2. **Feature Engineering**: FeatureStream вычисляет 11 фичей
3. **Prediction**: 3 модели дают prob_up/prob_down
4. **Ensemble**: Взвешенное усреднение → Decision
5. **Evaluation**: Сравнение с реальным движением
6. **Output**: summary.json, report.md, predictions.jsonl

---

## 2. Текущие метрики

### 2.1 Метрики прогнозирования (CORE - важные для проекта)

| Метрика | Описание | Полезность |
|---------|----------|------------|
| **accuracy** | % правильных UP/DOWN | ★★★★★ Главная |
| **precision** | Точность по направлению | ★★★★☆ |
| **recall** | Полнота по направлению | ★★★★☆ |
| **f1_score** | Гармоническое среднее | ★★★★☆ |
| **confusion_matrix** | UP→UP, UP→DOWN и т.д. | ★★★★★ Для анализа ошибок |
| **ECE** | Expected Calibration Error | ★★★★★ Качество уверенности |
| **confident_wrong_rate** | % ошибок при conf≥70% | ★★★★★ Критичная |
| **direction_balance** | Баланс UP/DOWN | ★★★★☆ Проверка на collapse |

### 2.2 Метрики по моделям

| Метрика | Описание | Полезность |
|---------|----------|------------|
| **per_model.accuracy** | Accuracy каждой модели | ★★★★★ |
| **per_model.confusion** | Confusion для модели | ★★★★☆ |
| **model_agreement** | Согласие моделей | ★★★☆☆ |
| **regime_analysis** | Accuracy по режимам | ★★★★★ Новая! |

### 2.3 Метрики торговли (НЕ ПРИОРИТЕТ)

| Метрика | Описание | Полезность |
|---------|----------|------------|
| sharpe_ratio | Risk-adjusted return | ★★☆☆☆ |
| win_rate | % прибыльных сделок | ★★☆☆☆ |
| profit_factor | Profit/Loss ratio | ★★☆☆☆ |
| max_drawdown | Максимальная просадка | ★★☆☆☆ |

---

## 3. СЛАБЫЕ МЕСТА И GAPS

### 3.1 Критические проблемы

#### ❌ Нет анализа ошибок по времени
**Проблема**: Мы не знаем, КОГДА ошибаемся чаще.
```
Текущее: Видим общую accuracy 50%
Нужно:   Видеть accuracy по часам, дням недели, сессиям
```

#### ❌ Нет анализа последовательных ошибок
**Проблема**: Не отслеживаем серии ошибок.
```
Текущее: max_loss_streak = 7
Нужно:   Анализ ЧТО вызывает серии, их характеристики
```

#### ❌ Нет анализа по величине движения
**Проблема**: Не знаем, на каких движениях ошибаемся.
```
Текущее: Считаем все движения одинаково
Нужно:   Accuracy на малых vs больших движениях
```

#### ❌ Нет feature importance analysis
**Проблема**: Не знаем, какие фичи реально влияют.
```
Текущее: Простая корреляция
Нужно:   Feature importance, SHAP values
```

### 3.2 Структурные проблемы

#### ❌ Однопроходное тестирование
**Проблема**: Один прогон не даёт статистической значимости.
```
Текущее: 1 прогон 24h
Нужно:   Walk-forward, cross-validation
```

#### ❌ Нет A/B тестирования
**Проблема**: Сложно сравнивать версии моделей.
```
Текущее: Ручное сравнение runs/
Нужно:   Автоматическое сравнение конфигов
```

#### ❌ Нет статистических тестов
**Проблема**: Нет p-values, confidence intervals.
```
Текущее: accuracy = 50%
Нужно:   accuracy = 50% ± 2% (95% CI), p < 0.05 vs random
```

### 3.3 Информационные gaps

#### ❌ Нет детального лога предсказаний
**Проблема**: predictions.jsonl есть, но анализ сложен.
```
Текущее: Сырые JSON записи
Нужно:   Удобные фильтры, SQL-like запросы
```

#### ❌ Нет визуализации
**Проблема**: Только текстовые отчёты.
```
Текущее: report.md с таблицами
Нужно:   Графики accuracy over time, confusion heatmaps
```

---

## 4. ПРЕДЛОЖЕНИЯ ПО УЛУЧШЕНИЮ

### 4.1 Приоритет 1: Новые метрики для анализа ошибок

#### 4.1.1 Временной анализ
```python
class TemporalAnalyzer:
    """Анализ accuracy по временным сегментам."""

    def analyze_by_hour(self) -> Dict[int, float]:
        """Accuracy по часам (0-23 UTC)."""

    def analyze_by_session(self) -> Dict[str, float]:
        """Accuracy по сессиям (Asia/Europe/US)."""

    def analyze_by_day_of_week(self) -> Dict[int, float]:
        """Accuracy по дням недели."""
```

#### 4.1.2 Анализ по magnitude движения
```python
class MagnitudeAnalyzer:
    """Анализ по величине движения."""

    def accuracy_by_return_bucket(self) -> Dict[str, float]:
        """
        Buckets:
        - tiny: |ret| < 0.01%
        - small: 0.01% <= |ret| < 0.05%
        - medium: 0.05% <= |ret| < 0.1%
        - large: |ret| >= 0.1%
        """
```

#### 4.1.3 Анализ серий
```python
class StreakAnalyzer:
    """Анализ последовательностей правильных/неправильных."""

    def get_error_streaks(self) -> List[StreakInfo]:
        """Информация о сериях ошибок."""

    def analyze_streak_causes(self) -> Dict[str, float]:
        """Что общего у предсказаний в серии ошибок."""
```

### 4.2 Приоритет 2: Улучшение тестовой инфраструктуры

#### 4.2.1 Walk-Forward Validation
```python
def walk_forward_test(
    data: List[Candle],
    train_window: int = 500,
    test_window: int = 100,
    step: int = 50,
) -> List[WalkForwardResult]:
    """
    Скользящее окно для честной валидации:

    |---train---|--test--|
         |---train---|--test--|
              |---train---|--test--|
    """
```

#### 4.2.2 Statistical Significance
```python
class StatisticalValidator:
    """Статистическая проверка результатов."""

    def binomial_test(self, correct: int, total: int) -> float:
        """P-value для accuracy vs random (50%)."""

    def confidence_interval(self, accuracy: float, n: int) -> Tuple[float, float]:
        """95% доверительный интервал."""

    def compare_configs(self, results_a: Stats, results_b: Stats) -> ComparisonResult:
        """A/B тест двух конфигураций."""
```

#### 4.2.3 Multi-run aggregation
```python
def run_multiple_tests(
    config: Dict,
    periods: List[Tuple[int, int]],  # [(start1, end1), (start2, end2), ...]
) -> AggregatedResults:
    """Запуск на нескольких периодах для robustness."""
```

### 4.3 Приоритет 3: Инструменты для быстрого дебага

#### 4.3.1 Error Explorer
```python
class ErrorExplorer:
    """Интерактивный анализ ошибок."""

    def get_worst_predictions(self, n: int = 20) -> List[PredictionRecord]:
        """Самые уверенные неправильные предсказания."""

    def filter_errors(
        self,
        regime: Optional[str] = None,
        confidence_min: Optional[float] = None,
        model: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """Фильтрация ошибок по критериям."""

    def compare_correct_vs_wrong(self) -> FeatureComparison:
        """Средние фичи для правильных vs неправильных."""
```

#### 4.3.2 Quick Diagnosis
```python
def diagnose_model(model_name: str, predictions_path: str) -> DiagnosisReport:
    """
    Быстрая диагностика модели:
    - Bias detection (тенденция к UP или DOWN)
    - Overconfidence detection
    - Regime weakness
    - Feature sensitivity
    """
```

#### 4.3.3 Regression Detection
```python
class RegressionDetector:
    """Обнаружение деградации между версиями."""

    def compare_runs(self, run_a: str, run_b: str) -> RegressionReport:
        """Сравнение двух runs/ с выявлением регрессий."""

    def detect_degradation(self, history: List[str]) -> List[DegradationAlert]:
        """Анализ тренда accuracy по runs."""
```

### 4.4 Приоритет 4: Улучшенные отчёты

#### 4.4.1 Structured Error Report
```markdown
## Топ ошибок

### 1. Confident Wrong (conf >= 70%, wrong)
- Количество: 21
- Характеристики:
  - Средний regime: volatile (60%)
  - Средний RSI: 42.3
  - Модели не согласны: 80%

### 2. Серии ошибок > 5
- Найдено: 3 серии
- Характеристики: все в volatile regime
```

#### 4.4.2 Feature Contribution Report
```markdown
## Вклад фичей в ошибки

| Фича | При ошибке | При успехе | Разница |
|------|------------|------------|---------|
| volatility_z | 1.8 | 0.3 | +1.5 |
| volume_z | 0.2 | 1.1 | -0.9 |
```

---

## 5. Приоритетный план реализации

### Фаза 1: Quick Wins (1-2 дня)
- [ ] TemporalAnalyzer (accuracy по часам)
- [ ] MagnitudeAnalyzer (accuracy по величине)
- [ ] Confidence Interval в summary

### Фаза 2: Статистика (2-3 дня)
- [ ] Binomial test (p-value vs random)
- [ ] Walk-forward validation
- [ ] Multi-run aggregation

### Фаза 3: Инструменты дебага (2-3 дня)
- [ ] ErrorExplorer
- [ ] Quick Diagnosis
- [ ] Regression Detection

### Фаза 4: Улучшенные отчёты (1-2 дня)
- [ ] Structured Error Report
- [ ] Feature Contribution Report
- [ ] HTML отчёт с графиками

---

## 6. Предлагаемые изменения в BacktestStats

### Новые поля:
```python
@dataclass
class BacktestStats:
    # Существующие...

    # НОВЫЕ: Временной анализ
    hourly_accuracy: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # НОВЫЕ: Анализ по magnitude
    magnitude_buckets: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # НОВЫЕ: Серии
    error_streaks: List[StreakInfo] = field(default_factory=list)

    # НОВЫЕ: Feature comparison
    feature_when_correct: Dict[str, float] = field(default_factory=dict)
    feature_when_wrong: Dict[str, float] = field(default_factory=dict)
```

### Новые методы:
```python
def temporal_analysis(self) -> Dict[str, object]:
    """Accuracy по временным сегментам."""

def magnitude_analysis(self) -> Dict[str, float]:
    """Accuracy по величине движения."""

def streak_analysis(self) -> Dict[str, object]:
    """Анализ серий правильных/неправильных."""

def statistical_significance(self) -> Dict[str, float]:
    """P-value, confidence interval."""
```

---

## 7. Резюме

### Сильные стороны текущей системы:
- Чистая архитектура
- Хороший базовый набор метрик
- Подробные JSON отчёты
- Режимный анализ (добавлен)

### Критические улучшения для ускорения разработки:
1. **Временной анализ** - понять КОГДА ошибаемся
2. **Анализ по magnitude** - понять НА ЧЁМ ошибаемся
3. **Статистическая значимость** - уверенность в результатах
4. **Error Explorer** - быстрый дебаг ошибок
5. **Regression Detection** - автоматическое обнаружение деградации
