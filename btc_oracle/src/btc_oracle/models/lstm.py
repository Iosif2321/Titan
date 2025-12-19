"""LSTM backbone для модели."""

import torch
import torch.nn as nn

from btc_oracle.models.base import BaseModel


class LSTMModel(BaseModel):
    """Модель на основе LSTM с двумя головами."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: размерность входных признаков
            hidden_dim: размерность скрытого состояния LSTM
            num_layers: количество слоёв LSTM
            dropout: dropout rate
        """
        super().__init__(input_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Общий энкодер признаков (для случая без временной последовательности)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Direction head (UP/DOWN)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # UP, DOWN
        )
        
        # Flat head (FLAT / not FLAT)
        self.flat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # sigmoid для FLAT
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: входной тензор [batch, input_dim] или [batch, seq_len, input_dim]
        
        Returns:
            словарь с logits
        """
        # Если 2D, добавляем временную размерность
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Берём последний hidden state
        if lstm_out.size(1) > 0:
            encoded = lstm_out[:, -1, :]  # [batch, hidden_dim]
        else:
            # Fallback на feature encoder если нет временной размерности
            encoded = self.feature_encoder(x.squeeze(1))
        
        # Heads
        direction_logits = self.direction_head(encoded)
        flat_logits = self.flat_head(encoded)
        
        return {
            "direction_logits": direction_logits,
            "flat_logits": flat_logits,
        }

