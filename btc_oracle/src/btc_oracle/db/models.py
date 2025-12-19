"""SQLAlchemy модели для TimescaleDB."""

from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, Index, LargeBinary, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Candle(Base):
    """Таблица свечей (hypertable в TimescaleDB)."""
    
    __tablename__ = "candles"
    
    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    timeframe = Column(String(10), primary_key=True, default="1m")
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    confirmed = Column(Integer, nullable=False, default=1)  # 1 = confirmed, 0 = not confirmed
    
    __table_args__ = (
        Index("ix_candles_time", "time"),
        Index("ix_candles_symbol_timeframe", "symbol", "timeframe", "time"),
    )


class Pattern(Base):
    """
    Таблица паттернов (память системы).
    
    stats_cold structure (multi-horizon):
    {
        "1m": {"UP": {"wins": N, "loss": M, "streak_fail": K}, "DOWN": {...}},
        "5m": {"UP": {...}, "DOWN": {...}},
        ...
    }
    
    weights structure:
    {
        "1m": 1.0,
        "5m": 1.0,
        ...
    }
    """
    
    __tablename__ = "patterns"
    
    signature = Column(BigInteger, primary_key=True)  # hash паттерна (uint64)
    history_hot = Column(LargeBinary)  # Bit-packed recent history (опционально)
    stats_cold = Column(JSONB)  # Агрегированная статистика по горизонтам/направлениям
    weights = Column(JSONB)  # Адаптивные веса по горизонтам
    last_updated = Column(DateTime(timezone=True))


class Prediction(Base):
    """
    Таблица прогнозов.
    
    Каждый прогноз уникально идентифицируется (time, symbol, horizon_minutes).
    Это позволяет хранить прогнозы для нескольких горизонтов на одном timestamp.
    """
    
    __tablename__ = "predictions"
    
    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    horizon_minutes = Column(Integer, primary_key=True)  # 1, 5, 10, 15, 30, 60
    
    # Прогноз
    label = Column(String(20), nullable=False)  # UP, DOWN, FLAT, UNCERTAIN
    reason_code = Column(String(50), nullable=False)
    p_up = Column(Float, nullable=False)
    p_down = Column(Float, nullable=False)
    p_flat = Column(Float, nullable=False)
    flat_score = Column(Float, nullable=False)
    uncertainty_score = Column(Float, nullable=False)
    consensus = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False, default=0.0)
    
    # Результат (заполняется после созревания)
    truth_label = Column(String(20), nullable=True)  # UP, DOWN, FLAT
    truth_magnitude = Column(Float, nullable=True)
    reward = Column(Float, nullable=True)
    matured_at = Column(DateTime(timezone=True), nullable=True)
    
    # Метаданные
    created_at = Column(DateTime(timezone=True), nullable=False)
    
    __table_args__ = (
        Index("ix_pred_symbol_horizon", "symbol", "horizon_minutes"),
        Index("ix_pred_pending", "matured_at", "horizon_minutes"),
        Index("ix_pred_latest", "symbol", "horizon_minutes", "time"),
    )


class ModelPrediction(Base):
    """
    Таблица прогнозов отдельных моделей (до фьюжна).

    Каждый прогноз уникален по (time, symbol, horizon_minutes, model_id).
    """

    __tablename__ = "model_predictions"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    horizon_minutes = Column(Integer, primary_key=True)
    model_id = Column(String(64), primary_key=True)

    label = Column(String(20), nullable=False)
    reason_code = Column(String(50), nullable=False)
    p_up = Column(Float, nullable=False)
    p_down = Column(Float, nullable=False)
    p_flat = Column(Float, nullable=False)
    u_dir = Column(Float, nullable=False)
    u_mag = Column(Float, nullable=False)
    consensus = Column(Float, nullable=False)
    disagreement = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_model_pred_symbol_horizon", "symbol", "horizon_minutes"),
        Index("ix_model_pred_latest", "symbol", "horizon_minutes", "time"),
        Index("ix_model_pred_model", "symbol", "model_id", "time"),
    )


class FeatureSnapshot(Base):
    """Снимок входных признаков, поданных в модель."""

    __tablename__ = "feature_snapshots"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    timeframe = Column(String(10), primary_key=True)

    window_size = Column(Integer, nullable=False)
    feature_dim = Column(Integer, nullable=False)
    feature_names = Column(JSONB, nullable=False)
    feature_vector = Column(JSONB, nullable=False)
    feature_meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_feature_snapshot_latest", "symbol", "timeframe", "time"),
    )
