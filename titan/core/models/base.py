from typing import Dict, Optional

from titan.core.types import ModelOutput, PatternContext


class BaseModel:
    name: str

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional[PatternContext] = None,
    ) -> ModelOutput:
        raise NotImplementedError
