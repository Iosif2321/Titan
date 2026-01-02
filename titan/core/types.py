from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelOutput:
    model_name: str
    prob_up: float
    prob_down: float
    state: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class Decision:
    direction: str
    confidence: float
    prob_up: float
    prob_down: float


@dataclass(frozen=True)
class PatternContext:
    pattern_id: int
    pattern_key: str
    model_name: str
    match_ratio: float
    accuracy: float
    up_accuracy: float
    down_accuracy: float
    bias: Optional[str]
    trust_confidence: float
    overconfident: bool
    confidence_cap: Optional[float]
    feature_insights: Dict[str, Dict[str, float]]
    temporal_insights: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class PredictionRecord:
    ts: int
    price: float
    pattern_id: int
    features: Dict[str, float]
    outputs: List[ModelOutput]
    decision: Decision


@dataclass(frozen=True)
class Outcome:
    actual_direction: str
    price_delta: float
    return_pct: float
