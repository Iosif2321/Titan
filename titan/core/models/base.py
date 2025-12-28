from typing import Dict

from titan.core.types import ModelOutput


class BaseModel:
    name: str

    def predict(self, features: Dict[str, float]) -> ModelOutput:
        raise NotImplementedError
