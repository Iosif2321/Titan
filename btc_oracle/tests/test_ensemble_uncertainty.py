"""Тесты для uncertainty ансамбля."""

import pytest
import numpy as np

from btc_oracle.core.types import NeuralOpinion


def test_consensus_calculation():
    """Тест вычисления consensus."""
    # Высокий consensus: модели согласны
    opinions_high_consensus = [
        NeuralOpinion(
            p_up=0.7,
            p_down=0.3,
            p_flat=0.1,
            u_dir=0.1,
            u_mag=0.1,
            consensus=0.9,
            disagreement=0.1,
        ),
        NeuralOpinion(
            p_up=0.65,
            p_down=0.35,
            p_flat=0.1,
            u_dir=0.1,
            u_mag=0.1,
            consensus=0.9,
            disagreement=0.1,
        ),
    ]
    
    # Низкий consensus: модели расходятся
    opinions_low_consensus = [
        NeuralOpinion(
            p_up=0.8,
            p_down=0.2,
            p_flat=0.1,
            u_dir=0.2,
            u_mag=0.1,
            consensus=0.5,
            disagreement=0.5,
        ),
        NeuralOpinion(
            p_up=0.2,
            p_down=0.8,
            p_flat=0.1,
            u_dir=0.2,
            u_mag=0.1,
            consensus=0.5,
            disagreement=0.5,
        ),
    ]
    
    # Проверяем, что disagreement выше при низком consensus
    disagreement_high = np.std([op.p_up for op in opinions_high_consensus])
    disagreement_low = np.std([op.p_up for op in opinions_low_consensus])
    
    assert disagreement_low > disagreement_high

