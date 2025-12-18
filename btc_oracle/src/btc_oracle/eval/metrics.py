"""Метрики для оценки качества системы."""

from collections import defaultdict
from typing import Optional

import numpy as np

from btc_oracle.core.types import Decision, Label


class MetricsCollector:
    """Сборщик метрик."""
    
    def __init__(self):
        self.predictions: list[tuple[Decision, Optional[Label], Optional[float]]] = []
        self.latencies: list[float] = []
    
    def record(
        self,
        decision: Decision,
        truth: Optional[Label] = None,
        magnitude: Optional[float] = None,
    ) -> None:
        """Записать прогноз и truth."""
        self.predictions.append((decision, truth, magnitude))
        self.latencies.append(decision.latency_ms)
    
    def compute_logloss(self) -> float:
        """Вычислить средний logloss."""
        losses = []
        for decision, truth, _ in self.predictions:
            if truth is None:
                continue
            
            if truth == Label.UP:
                p_true = decision.p_up
            elif truth == Label.DOWN:
                p_true = decision.p_down
            elif truth == Label.FLAT:
                p_true = decision.p_flat
            else:
                continue
            
            losses.append(-np.log(p_true + 1e-10))
        
        return float(np.mean(losses)) if losses else 0.0
    
    def compute_brier_score(self) -> float:
        """Вычислить средний Brier score."""
        scores = []
        for decision, truth, _ in self.predictions:
            if truth is None:
                continue
            
            y_up = 1.0 if truth == Label.UP else 0.0
            y_down = 1.0 if truth == Label.DOWN else 0.0
            y_flat = 1.0 if truth == Label.FLAT else 0.0
            
            score = (
                (decision.p_up - y_up) ** 2 +
                (decision.p_down - y_down) ** 2 +
                (decision.p_flat - y_flat) ** 2
            ) / 3.0
            scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_coverage(self) -> float:
        """Вычислить coverage (доля не-UNCERTAIN)."""
        if not self.predictions:
            return 0.0
        
        non_uncertain = sum(
            1 for decision, _, _ in self.predictions
            if decision.label != Label.UNCERTAIN
        )
        return non_uncertain / len(self.predictions)
    
    def compute_conditional_accuracy(self) -> float:
        """Вычислить условную accuracy (только UP/DOWN, когда не UNCERTAIN)."""
        correct = 0
        total = 0
        
        for decision, truth, _ in self.predictions:
            if truth is None or decision.label == Label.UNCERTAIN or truth == Label.FLAT:
                continue
            
            if decision.label == truth:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_latency_stats(self) -> dict[str, float]:
        """Вычислить статистику латентности."""
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
        
        return {
            "p50": float(np.percentile(self.latencies, 50)),
            "p95": float(np.percentile(self.latencies, 95)),
            "mean": float(np.mean(self.latencies)),
        }
    
    def get_confusion_matrix(self) -> dict[str, int]:
        """Получить confusion matrix для FLAT vs UNCERTAIN."""
        confusion = defaultdict(int)
        
        for decision, truth, _ in self.predictions:
            if truth is None:
                continue
            
            key = f"{decision.label.value}_{truth.value}"
            confusion[key] += 1
        
        return dict(confusion)
    
    def reset(self) -> None:
        """Сбросить метрики."""
        self.predictions.clear()
        self.latencies.clear()

