"""Тесты для Pattern Memory."""

import pytest
from datetime import datetime, timedelta

from btc_oracle.core.types import Label
from btc_oracle.patterns.stats import PatternStats


def test_pattern_stats_update():
    """Тест обновления статистики паттерна."""
    stats = PatternStats(
        pattern_id=123,
        timeframe="1m",
        horizon=5,
    )
    
    # Обновляем UP
    stats.update(Label.UP, magnitude=0.1)
    assert stats.alpha_up == 2.0  # 1.0 + 1
    assert stats.beta_down == 1.0
    assert stats.n == 1
    
    # Обновляем DOWN
    stats.update(Label.DOWN, magnitude=0.2)
    assert stats.alpha_up == 2.0
    assert stats.beta_down == 2.0  # 1.0 + 1
    assert stats.n == 2
    
    # Обновляем FLAT
    stats.update(Label.FLAT, magnitude=0.01)
    assert stats.alpha_flat == 2.0  # 1.0 + 1
    assert stats.beta_not_flat == 3.0  # 1.0 + 2 (UP и DOWN не FLAT)
    assert stats.n == 3


def test_pattern_stats_decay():
    """Тест decay статистики."""
    stats = PatternStats(
        pattern_id=123,
        timeframe="1m",
        horizon=5,
        alpha_up=10.0,
        beta_down=5.0,
        n=15,
        last_seen=datetime.now() - timedelta(days=10),
    )
    
    # Применяем decay
    stats._apply_decay(half_life_hours=168)  # 7 дней
    
    # Проверяем, что параметры уменьшились (но не ниже prior)
    assert stats.alpha_up >= 1.0
    assert stats.beta_down >= 1.0
    assert stats.decay_factor < 1.0


def test_pattern_stats_cooldown():
    """Тест cooldown механизма."""
    stats = PatternStats(
        pattern_id=123,
        timeframe="1m",
        horizon=5,
    )
    
    assert not stats.is_in_cooldown
    
    # Активируем cooldown
    stats.record_error(cooldown_duration_hours=24)
    
    assert stats.is_in_cooldown
    assert stats.cooldown_until is not None

