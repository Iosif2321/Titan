"""Тесты для Policy (критично: FLAT != UNCERTAIN)."""

import pytest
from datetime import datetime

from btc_oracle.core.config import Config
from btc_oracle.core.types import FusedForecast, Label, ReasonCode
from btc_oracle.fusion.policy import DecisionPolicy


def test_flat_not_uncertain():
    """Тест: FLAT определяется по flat_score, не по uncertainty."""
    config = Config.default()
    policy = DecisionPolicy(config)
    
    # Случай: высокий p_flat, низкий u_mag, но высокий u_dir
    forecast = FusedForecast(
        p_up=0.3,
        p_down=0.2,
        p_flat=0.7,  # высокий p_flat
        u_dir=0.8,  # высокий uncertainty для direction
        u_mag=0.1,  # низкий uncertainty для magnitude
        flat_score=0.7,
        uncertainty_score=0.8,  # высокий общий uncertainty
        consensus=0.3,  # низкий consensus
    )
    
    decision = policy.decide(
        forecast=forecast,
        symbol="BTCUSDT",
        horizon_min=5,
        ts=datetime.now(),
    )
    
    # Должен быть FLAT, не UNCERTAIN
    assert decision.label == Label.FLAT
    assert decision.reason_code == ReasonCode.TINY_MOVE_EXPECTED


def test_uncertain_not_flat():
    """Тест: UNCERTAIN определяется по uncertainty, не по p_flat."""
    config = Config.default()
    policy = DecisionPolicy(config)
    
    # Случай: низкий p_flat, но высокий uncertainty
    forecast = FusedForecast(
        p_up=0.5,
        p_down=0.3,
        p_flat=0.2,  # низкий p_flat
        u_dir=0.5,
        u_mag=0.4,  # высокий u_mag (но не критично)
        flat_score=0.2,
        uncertainty_score=0.6,  # высокий uncertainty
        consensus=0.4,  # низкий consensus
    )
    
    decision = policy.decide(
        forecast=forecast,
        symbol="BTCUSDT",
        horizon_min=5,
        ts=datetime.now(),
    )
    
    # Должен быть UNCERTAIN, не FLAT
    assert decision.label == Label.UNCERTAIN
    assert decision.reason_code in [
        ReasonCode.HIGH_EPISTEMIC_UNCERTAINTY,
        ReasonCode.LOW_ENSEMBLE_CONSENSUS,
    ]


def test_up_down_decisions():
    """Тест: UP/DOWN решения."""
    config = Config.default()
    policy = DecisionPolicy(config)
    
    # UP
    forecast_up = FusedForecast(
        p_up=0.7,
        p_down=0.2,
        p_flat=0.1,
        u_dir=0.2,
        u_mag=0.1,
        flat_score=0.1,
        uncertainty_score=0.2,
        consensus=0.8,
    )
    
    decision = policy.decide(
        forecast=forecast_up,
        symbol="BTCUSDT",
        horizon_min=5,
        ts=datetime.now(),
    )
    
    assert decision.label == Label.UP
    
    # DOWN
    forecast_down = FusedForecast(
        p_up=0.2,
        p_down=0.7,
        p_flat=0.1,
        u_dir=0.2,
        u_mag=0.1,
        flat_score=0.1,
        uncertainty_score=0.2,
        consensus=0.8,
    )
    
    decision = policy.decide(
        forecast=forecast_down,
        symbol="BTCUSDT",
        horizon_min=5,
        ts=datetime.now(),
    )
    
    assert decision.label == Label.DOWN

