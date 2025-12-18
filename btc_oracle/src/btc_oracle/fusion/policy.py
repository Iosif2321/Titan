"""Policy для принятия решений (4 состояния: UP, DOWN, FLAT, UNCERTAIN)."""

from datetime import datetime

from btc_oracle.core.config import Config
from btc_oracle.core.types import Decision, FusedForecast, Label, ReasonCode


class DecisionPolicy:
    """Политика принятия решений с 4 состояниями."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: конфигурация системы
        """
        self.config = config
    
    def decide(
        self,
        forecast: FusedForecast,
        symbol: str,
        horizon_min: int,
        ts: datetime,
        latency_ms: float = 0.0,
    ) -> Decision:
        """
        Принять решение на основе прогноза.
        
        Порядок проверки (строгий):
        1. FLAT (если p_flat >= threshold И u_mag <= max)
        2. UNCERTAIN (если uncertainty высокий ИЛИ consensus низкий)
        3. UP/DOWN (если вероятность направления >= threshold)
        
        Args:
            forecast: объединённый прогноз
            symbol: символ
            horizon_min: горизонт в минутах
            ts: timestamp прогноза
            latency_ms: латентность инференса
        
        Returns:
            Decision
        """
        # 1. Проверяем FLAT (сначала!)
        if (
            forecast.p_flat >= self.config.flat.p_thr
            and forecast.u_mag <= self.config.flat.u_mag_max
        ):
            return Decision(
                label=Label.FLAT,
                reason_code=ReasonCode.TINY_MOVE_EXPECTED,
                ts=ts,
                symbol=symbol,
                horizon_min=horizon_min,
                p_up=forecast.p_up,
                p_down=forecast.p_down,
                p_flat=forecast.p_flat,
                flat_score=forecast.flat_score,
                uncertainty_score=forecast.uncertainty_score,
                consensus=forecast.consensus,
                memory=forecast.memory_support,
                latency_ms=latency_ms,
            )
        
        # 2. Проверяем UNCERTAIN
        if (
            forecast.uncertainty_score >= self.config.uncertainty.u_thr
            or forecast.consensus < self.config.uncertainty.consensus_thr
        ):
            # Определяем причину
            if forecast.consensus < self.config.uncertainty.consensus_thr:
                reason = ReasonCode.LOW_ENSEMBLE_CONSENSUS
            elif forecast.uncertainty_score >= self.config.uncertainty.u_thr:
                reason = ReasonCode.HIGH_EPISTEMIC_UNCERTAINTY
            else:
                reason = ReasonCode.HIGH_EPISTEMIC_UNCERTAINTY
            
            # Проверяем конфликт с памятью
            if forecast.memory_support and forecast.memory_support.get("weight", 0) > 0.3:
                # Есть конфликт если память и нейросеть расходятся
                mem_p_up = forecast.memory_support.get("p_up_mem", 0.5)
                neural_p_up = forecast.p_up
                if abs(mem_p_up - neural_p_up) > 0.3:
                    reason = ReasonCode.MEMORY_VS_NEURAL_CONFLICT
            
            return Decision(
                label=Label.UNCERTAIN,
                reason_code=reason,
                ts=ts,
                symbol=symbol,
                horizon_min=horizon_min,
                p_up=forecast.p_up,
                p_down=forecast.p_down,
                p_flat=forecast.p_flat,
                flat_score=forecast.flat_score,
                uncertainty_score=forecast.uncertainty_score,
                consensus=forecast.consensus,
                memory=forecast.memory_support,
                latency_ms=latency_ms,
            )
        
        # 3. Проверяем направление (UP/DOWN)
        if forecast.p_up >= self.config.decision.dir_p_thr:
            reason = ReasonCode.STRONG_DIRECTIONAL_SIGNAL
            if forecast.memory_support and forecast.memory_support.get("credibility", 0) > 0.7:
                reason = ReasonCode.MEMORY_HIGH_CONFIDENCE
            
            return Decision(
                label=Label.UP,
                reason_code=reason,
                ts=ts,
                symbol=symbol,
                horizon_min=horizon_min,
                p_up=forecast.p_up,
                p_down=forecast.p_down,
                p_flat=forecast.p_flat,
                flat_score=forecast.flat_score,
                uncertainty_score=forecast.uncertainty_score,
                consensus=forecast.consensus,
                memory=forecast.memory_support,
                latency_ms=latency_ms,
            )
        
        if forecast.p_down >= self.config.decision.dir_p_thr:
            reason = ReasonCode.STRONG_DIRECTIONAL_SIGNAL
            if forecast.memory_support and forecast.memory_support.get("credibility", 0) > 0.7:
                reason = ReasonCode.MEMORY_HIGH_CONFIDENCE
            
            return Decision(
                label=Label.DOWN,
                reason_code=reason,
                ts=ts,
                symbol=symbol,
                horizon_min=horizon_min,
                p_up=forecast.p_up,
                p_down=forecast.p_down,
                p_flat=forecast.p_flat,
                flat_score=forecast.flat_score,
                uncertainty_score=forecast.uncertainty_score,
                consensus=forecast.consensus,
                memory=forecast.memory_support,
                latency_ms=latency_ms,
            )
        
        # Fallback: если ничего не подошло → UNCERTAIN
        return Decision(
            label=Label.UNCERTAIN,
            reason_code=ReasonCode.INSUFFICIENT_PATTERN_SUPPORT,
            ts=ts,
            symbol=symbol,
            horizon_min=horizon_min,
            p_up=forecast.p_up,
            p_down=forecast.p_down,
            p_flat=forecast.p_flat,
            flat_score=forecast.flat_score,
            uncertainty_score=forecast.uncertainty_score,
            consensus=forecast.consensus,
            memory=forecast.memory_support,
            latency_ms=latency_ms,
        )

