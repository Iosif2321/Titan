"""Three-Head TFT Model for direction prediction.

Sprint 23: Replaces heuristic models (TrendVIC, Oscillator, VolumeMetrix) with
ML models using Temporal Fusion Transformer architecture.

Architecture:
    - Shared TFT Encoder: LSTM + Multi-Head Attention
    - Three specialized prediction heads: TrendML, OscillatorML, VolumeML
    - Pattern attention for historical pattern events
    - Designed for GTX 1060 (6GB): hidden=64, heads=4, seq=100

Memory estimation: ~3.5GB for batch_size=32
"""
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

from titan.core.config import ConfigStore
from titan.core.models.base import BaseModel
from titan.core.types import ModelOutput, PatternContext


# Feature groups for each head (from CLAUDE.md)
TREND_FEATURES = [
    "ma_fast", "ma_slow", "ma_delta", "ma_delta_pct",
    "ema_10_spread_pct", "ema_20_spread_pct", "adx",
    "price_momentum_3", "return_5", "return_10",
    "body_ratio", "candle_direction",
]

OSCILLATOR_FEATURES = [
    "rsi", "rsi_momentum", "rsi_oversold", "rsi_overbought",
    "bb_position", "stochastic_k", "stochastic_d", "mfi",
    "upper_wick_ratio", "lower_wick_ratio",
]

VOLUME_FEATURES = [
    "volume_z", "volume_trend", "volume_change_pct",
    "vol_imbalance_20", "vol_ratio", "atr_pct",
    "high_low_range_pct", "body_pct",
]

# Common features used by all heads
COMMON_FEATURES = ["close", "return_1", "volatility_z"]

# All features combined - MUST use sorted() for consistent ordering across Python sessions!
# Using list(set(...)) would cause random feature order on each Python restart, breaking model loading
ALL_FEATURES = sorted(set(TREND_FEATURES + OSCILLATOR_FEATURES + VOLUME_FEATURES + COMMON_FEATURES))


class VariableSelectionNetwork(nn.Module):
    """Selects and weights input features using learned attention."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # NOTE: Weight GRN must output input_dim (for variable weights), not hidden_dim!
        # Bug fix: was GRN(input_dim, hidden_dim) which caused dimension mismatch
        self.flattened_grn = GatedResidualNetwork(input_dim, input_dim, dropout)
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, dropout) for _ in range(input_dim)
        ])
        self.softmax = nn.Softmax(dim=-1)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            selected: [batch, seq_len, hidden_dim]
            weights: [batch, seq_len, input_dim] - variable importance weights
        """
        # Flatten for attention weights
        flat = x.view(x.shape[0] * x.shape[1], -1)
        weights = self.flattened_grn(flat)
        weights = self.softmax(weights[:, :self.input_dim])
        weights = weights.view(x.shape[0], x.shape[1], -1)

        # Process each variable
        var_outputs = []
        for i, grn in enumerate(self.variable_grns):
            var_input = x[:, :, i:i+1]
            var_out = grn(var_input.view(-1, 1))
            var_outputs.append(var_out.view(x.shape[0], x.shape[1], -1))

        var_outputs = torch.stack(var_outputs, dim=-1)  # [batch, seq, hidden, n_vars]
        weights_expanded = weights.unsqueeze(2)  # [batch, seq, 1, n_vars]
        selected = (var_outputs * weights_expanded).sum(dim=-1)  # [batch, seq, hidden]

        return selected, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for non-linear processing."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        h = gate * h + (1 - gate) * skip
        return self.layer_norm(h)


class TemporalFusionEncoder(nn.Module):
    """Shared TFT encoder with LSTM and Multi-Head Attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Variable Selection
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            encoded: [batch, hidden_dim] - final encoding
            var_weights: [batch, seq_len, input_dim] - variable importance
        """
        # Variable selection
        selected, var_weights = self.vsn(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)

        # Residual connection
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Post-attention processing
        processed = self.post_attn_grn(attn_out[:, -1, :])  # Use last timestep

        return processed, var_weights


class PatternAttention(nn.Module):
    """Attention mechanism for pattern event history."""

    def __init__(self, hidden_dim: int, event_dim: int = 6, num_events: int = 50, dropout: float = 0.2):
        super().__init__()
        self.event_embed = nn.Linear(event_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_events = num_events

    def forward(
        self,
        query: torch.Tensor,
        pattern_events: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, hidden_dim] - encoder output
            pattern_events: [batch, num_events, event_dim] - historical events

        Returns:
            enriched: [batch, hidden_dim] - query enriched with pattern info
        """
        if pattern_events is None:
            return query

        # Embed events
        events_embedded = self.event_embed(pattern_events)  # [batch, n_events, hidden]

        # Query expansion for attention
        query_expanded = query.unsqueeze(1)  # [batch, 1, hidden]

        # Cross-attention: query attends to events
        attn_out, _ = self.attention(query_expanded, events_embedded, events_embedded)

        # Project and combine
        attn_out = self.output_proj(attn_out.squeeze(1))

        return query + attn_out


class PredictionHead(nn.Module):
    """Specialized prediction head for direction probability."""

    def __init__(
        self,
        hidden_dim: int,
        head_features: List[str],
        head_name: str,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.head_name = head_name
        self.head_features = head_features
        feature_dim = len(head_features)

        # Feature embedding for head-specific features
        self.feature_embed = nn.Linear(feature_dim, hidden_dim // 2)

        # Combined processing
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 2)  # prob_up, prob_down

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)

    def forward(
        self,
        encoder_out: torch.Tensor,
        head_features: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: [batch, hidden_dim] - shared encoder output
            head_features: [batch, feature_dim] - head-specific features
            return_logits: If True, return (logits, probs), else just probs

        Returns:
            probs: [batch, 2] - prob_up, prob_down (softmax)
            or (logits, probs) if return_logits=True
        """
        # Embed head features
        feat_embed = F.relu(self.feature_embed(head_features))

        # Combine with encoder output
        combined = torch.cat([encoder_out, feat_embed], dim=-1)
        combined = self.layer_norm1(F.relu(self.fc1(combined)))
        combined = self.dropout(combined)
        combined = self.layer_norm2(F.relu(self.fc2(combined)))
        combined = self.dropout(combined)

        # Output logits and probabilities
        logits = self.output(combined)
        probs = F.softmax(logits, dim=-1)

        if return_logits:
            return logits, probs
        return probs


class ThreeHeadTFT(nn.Module):
    """Three-Head Temporal Fusion Transformer for direction prediction.

    Combines:
    - Shared TFT encoder for temporal patterns
    - Pattern attention for historical pattern events
    - Three specialized heads for trend, oscillator, and volume analysis
    """

    def __init__(
        self,
        config: ConfigStore,
        input_dim: int = None,  # Defaults to len(ALL_FEATURES) = 33
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        num_pattern_events: int = 50,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # Default input_dim to number of ALL_FEATURES
        if input_dim is None:
            input_dim = len(ALL_FEATURES)

        # Override from config if available
        hidden_dim = int(config.get("tft.hidden_dim", hidden_dim))
        num_heads = int(config.get("tft.num_heads", num_heads))
        num_lstm_layers = int(config.get("tft.num_lstm_layers", num_lstm_layers))
        dropout = float(config.get("tft.dropout", dropout))
        num_pattern_events = int(config.get("tft.num_pattern_events", num_pattern_events))

        # Shared encoder
        self.encoder = TemporalFusionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

        # Pattern attention
        self.pattern_attention = PatternAttention(
            hidden_dim=hidden_dim,
            event_dim=6,  # direction, confidence, hit, return_pct, recency, model_idx
            num_events=num_pattern_events,
            dropout=dropout,
        )

        # Prediction heads
        self.trend_head = PredictionHead(hidden_dim, TREND_FEATURES, "TRENDML", dropout)
        self.oscillator_head = PredictionHead(hidden_dim, OSCILLATOR_FEATURES, "OSCILLATORML", dropout)
        self.volume_head = PredictionHead(hidden_dim, VOLUME_FEATURES, "VOLUMEML", dropout)

        # Pattern aggregates embedding (10 features: accuracy, up_acc, down_acc, etc.)
        self.pattern_agg_embed = nn.Linear(10, hidden_dim)

        # Device tracking
        self._device = torch.device("cpu")

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(
        self,
        sequence: torch.Tensor,
        trend_features: torch.Tensor,
        oscillator_features: torch.Tensor,
        volume_features: torch.Tensor,
        pattern_aggregates: Optional[torch.Tensor] = None,
        pattern_events: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the three-head TFT model.

        Args:
            sequence: [batch, seq_len, input_dim] - historical features
            trend_features: [batch, n_trend] - trend-specific features
            oscillator_features: [batch, n_osc] - oscillator-specific features
            volume_features: [batch, n_vol] - volume-specific features
            pattern_aggregates: [batch, 10] - pattern aggregate stats
            pattern_events: [batch, n_events, 6] - historical pattern events
            mask: Optional attention mask
            return_logits: If True, aux contains logits for training

        Returns:
            trend_probs: [batch, 2]
            oscillator_probs: [batch, 2]
            volume_probs: [batch, 2]
            aux: Dict with variable weights, attention info, and optionally logits
        """
        # Shared encoding
        encoded, var_weights = self.encoder(sequence, mask)

        # Add pattern aggregates if available
        if pattern_aggregates is not None:
            agg_embed = F.relu(self.pattern_agg_embed(pattern_aggregates))
            encoded = encoded + agg_embed

        # Pattern attention
        encoded = self.pattern_attention(encoded, pattern_events)

        # Head predictions
        if return_logits:
            trend_logits, trend_probs = self.trend_head(encoded, trend_features, return_logits=True)
            osc_logits, osc_probs = self.oscillator_head(encoded, oscillator_features, return_logits=True)
            vol_logits, vol_probs = self.volume_head(encoded, volume_features, return_logits=True)
            aux = {
                "var_weights": var_weights,
                "encoded": encoded,
                "trend_logits": trend_logits,
                "osc_logits": osc_logits,
                "vol_logits": vol_logits,
            }
        else:
            trend_probs = self.trend_head(encoded, trend_features)
            osc_probs = self.oscillator_head(encoded, oscillator_features)
            vol_probs = self.volume_head(encoded, volume_features)
            aux = {
                "var_weights": var_weights,
                "encoded": encoded,
            }

        return trend_probs, osc_probs, vol_probs, aux


class TwoHeadMLP(nn.Module):
    """Simple but working MLP model for direction prediction.

    Replaces the broken TFT architecture with a proven working model.
    Uses LSTM for sequence encoding + MLP heads.

    Tested: Achieves 90%+ on overfit test (vs TFT which cannot learn).
    """

    def __init__(
        self,
        config: ConfigStore,
        input_dim: int,
        hidden_dim: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._config = config
        self.hidden_dim = hidden_dim

        # Simple LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Two prediction heads (trend + volume/oscillator combined)
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim + len(TREND_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # logits
        )

        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim + len(OSCILLATOR_FEATURES) + len(VOLUME_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # logits
        )

        self._device = torch.device("cpu")

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(
        self,
        sequence: torch.Tensor,
        trend_features: torch.Tensor,
        oscillator_features: torch.Tensor,
        volume_features: torch.Tensor,
        pattern_aggregates: Optional[torch.Tensor] = None,
        pattern_events: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass - compatible with ThreeHeadTFT interface.

        Returns same structure: trend_probs, osc_probs, vol_probs, aux
        (osc_probs and vol_probs are same since we use combined aux_head)
        """
        # LSTM encoding - use last hidden state
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        encoded = h_n[-1]  # [batch, hidden_dim] - last layer hidden state

        # Trend head
        trend_input = torch.cat([encoded, trend_features], dim=-1)
        trend_logits = self.trend_head(trend_input)
        trend_probs = F.softmax(trend_logits, dim=-1)

        # Aux head (combines oscillator + volume)
        aux_input = torch.cat([encoded, oscillator_features, volume_features], dim=-1)
        aux_logits = self.aux_head(aux_input)
        aux_probs = F.softmax(aux_logits, dim=-1)

        # Return same format as ThreeHeadTFT for compatibility
        aux_dict = {"encoded": encoded}

        if return_logits:
            aux_dict["trend_logits"] = trend_logits
            aux_dict["osc_logits"] = aux_logits  # same as vol_logits
            aux_dict["vol_logits"] = aux_logits

        # osc_probs and vol_probs are same (two-head model)
        return trend_probs, aux_probs, aux_probs, aux_dict


# Session indices for session-aware models (4 sessions aligned with patterns.py)
SESSION_TO_IDX = {"ASIA": 0, "EUROPE": 1, "OVERLAP": 2, "US": 3}
IDX_TO_SESSION = {0: "ASIA", 1: "EUROPE", 2: "OVERLAP", 3: "US"}


class SessionEmbeddingMLP(nn.Module):
    """Session-aware MLP with embedding (P2.5).

    Uses shared LSTM encoder + shared heads, but adds session embedding
    to provide session context while maintaining shared learning.

    Advantages over SessionGatedMLP:
    - Shared learning across all sessions (more data per head)
    - Fewer parameters (single set of heads + small embedding)
    - Session context without separate decision boundaries

    Goal: Overall >=54%, Worst >=53%, Gap <=2.5%
    """

    def __init__(
        self,
        config: ConfigStore,
        input_dim: int,
        hidden_dim: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
        session_embed_dim: int = 8,  # Small embedding for session context
    ):
        super().__init__()
        self._config = config
        self.hidden_dim = hidden_dim
        self.session_embed_dim = session_embed_dim

        # Session embedding: 4 sessions -> embed_dim (ASIA, EUROPE, OVERLAP, US)
        self.session_embedding = nn.Embedding(4, session_embed_dim)

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Shared prediction heads (with session embedding concatenated)
        # Input: hidden_dim + session_embed_dim + feature_dim
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim + session_embed_dim + len(TREND_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim + session_embed_dim + len(OSCILLATOR_FEATURES) + len(VOLUME_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self._device = torch.device("cpu")

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(
        self,
        sequence: torch.Tensor,
        trend_features: torch.Tensor,
        oscillator_features: torch.Tensor,
        volume_features: torch.Tensor,
        session_idx: torch.Tensor,  # [batch] - session indices (0=ASIA, 1=EUROPE, 2=US)
        pattern_aggregates: Optional[torch.Tensor] = None,
        pattern_events: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with session embedding.

        Args:
            sequence: [batch, seq_len, input_dim]
            trend_features: [batch, n_trend]
            oscillator_features: [batch, n_osc]
            volume_features: [batch, n_vol]
            session_idx: [batch] - session indices (0=ASIA, 1=EUROPE, 2=US)

        Returns same structure: trend_probs, osc_probs, vol_probs, aux
        """
        # Get session embedding
        session_emb = self.session_embedding(session_idx)  # [batch, session_embed_dim]

        # Shared LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        encoded = h_n[-1]  # [batch, hidden_dim]

        # Concatenate encoded + session_emb + features for each head
        trend_input = torch.cat([encoded, session_emb, trend_features], dim=-1)
        aux_input = torch.cat([encoded, session_emb, oscillator_features, volume_features], dim=-1)

        # Shared heads with session context
        trend_logits = self.trend_head(trend_input)
        aux_logits = self.aux_head(aux_input)

        # Convert to probabilities
        trend_probs = F.softmax(trend_logits, dim=-1)
        aux_probs = F.softmax(aux_logits, dim=-1)

        # Return same format as TwoHeadMLP
        aux_dict = {"encoded": encoded, "session_idx": session_idx, "session_emb": session_emb}

        if return_logits:
            aux_dict["trend_logits"] = trend_logits
            aux_dict["osc_logits"] = aux_logits
            aux_dict["vol_logits"] = aux_logits

        return trend_probs, aux_probs, aux_probs, aux_dict


class SessionGatedMLP(nn.Module):
    """Session-gated MLP model for direction prediction (P2).

    Uses shared LSTM encoder + 3 session-specific prediction heads.
    Each session (ASIA, EUROPE, US) has its own decision boundary.

    Goal: Reduce session gap to ≤2.5% while maintaining ≥54% overall accuracy.
    """

    def __init__(
        self,
        config: ConfigStore,
        input_dim: int,
        hidden_dim: int = 64,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._config = config
        self.hidden_dim = hidden_dim
        self.num_sessions = 4  # ASIA, EUROPE, OVERLAP, US

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Session-specific heads (4 sets: ASIA, EUROPE, OVERLAP, US)
        self.trend_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + len(TREND_FEATURES), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
            )
            for _ in range(self.num_sessions)
        ])

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + len(OSCILLATOR_FEATURES) + len(VOLUME_FEATURES), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
            )
            for _ in range(self.num_sessions)
        ])

        self._device = torch.device("cpu")

    def to(self, device):
        self._device = device
        return super().to(device)

    def forward(
        self,
        sequence: torch.Tensor,
        trend_features: torch.Tensor,
        oscillator_features: torch.Tensor,
        volume_features: torch.Tensor,
        session_idx: torch.Tensor,  # [batch] - session indices (0=ASIA, 1=EUROPE, 2=US)
        pattern_aggregates: Optional[torch.Tensor] = None,
        pattern_events: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with session gating.

        Args:
            sequence: [batch, seq_len, input_dim]
            trend_features: [batch, n_trend]
            oscillator_features: [batch, n_osc]
            volume_features: [batch, n_vol]
            session_idx: [batch] - session indices (0=ASIA, 1=EUROPE, 2=US)

        Returns same structure: trend_probs, osc_probs, vol_probs, aux
        """
        batch_size = sequence.shape[0]

        # Shared LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        encoded = h_n[-1]  # [batch, hidden_dim]

        # Prepare inputs for heads
        trend_input = torch.cat([encoded, trend_features], dim=-1)
        aux_input = torch.cat([encoded, oscillator_features, volume_features], dim=-1)

        # Initialize outputs
        trend_logits = torch.zeros(batch_size, 2, device=sequence.device)
        aux_logits = torch.zeros(batch_size, 2, device=sequence.device)

        # Route each sample to its session-specific head
        for sess_id in range(self.num_sessions):
            mask = (session_idx == sess_id)
            if mask.any():
                trend_logits[mask] = self.trend_heads[sess_id](trend_input[mask])
                aux_logits[mask] = self.aux_heads[sess_id](aux_input[mask])

        # Convert to probabilities
        trend_probs = F.softmax(trend_logits, dim=-1)
        aux_probs = F.softmax(aux_logits, dim=-1)

        # Return same format as TwoHeadMLP
        aux_dict = {"encoded": encoded, "session_idx": session_idx}

        if return_logits:
            aux_dict["trend_logits"] = trend_logits
            aux_dict["osc_logits"] = aux_logits
            aux_dict["vol_logits"] = aux_logits

        return trend_probs, aux_probs, aux_probs, aux_dict


class TFTModelWrapper(BaseModel):
    """Wrapper to use TFT model with existing Titan interface.

    Implements BaseModel interface for compatibility with Ensemble.
    """

    def __init__(
        self,
        config: ConfigStore,
        head_name: str,
        model: ThreeHeadTFT,
        feature_buffer: "FeatureBuffer",
    ):
        self.name = head_name
        self._config = config
        self._model = model
        self._feature_buffer = feature_buffer
        self._head_name = head_name

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional[PatternContext] = None,
    ) -> ModelOutput:
        """Make prediction using TFT model.

        Args:
            features: Current market features
            pattern_context: Pattern context (used for pattern aggregates)

        Returns:
            ModelOutput with probabilities
        """
        # Add features to buffer and get sequence
        self._feature_buffer.add(features)

        # Check if we have enough history
        if not self._feature_buffer.is_ready():
            # Return neutral prediction until buffer is filled
            return ModelOutput(
                model_name=self.name,
                prob_up=0.5,
                prob_down=0.5,
                state={"buffer_filling": True},
                metrics={},
            )

        # Get tensors from buffer
        with torch.no_grad():
            sequence, trend_feat, osc_feat, vol_feat = self._feature_buffer.get_tensors()

            # Get pattern data if available
            pattern_agg = None
            pattern_events = None
            if pattern_context:
                pattern_agg = self._build_pattern_aggregates(pattern_context)
                pattern_events = self._get_pattern_events(pattern_context.pattern_id)

            # Forward pass
            trend_probs, osc_probs, vol_probs, aux = self._model(
                sequence,
                trend_feat,
                osc_feat,
                vol_feat,
                pattern_agg,
                pattern_events,
            )

            # Select output based on head
            if self._head_name == "TRENDML":
                probs = trend_probs
            elif self._head_name == "OSCILLATORML":
                probs = osc_probs
            else:  # VOLUMEML
                probs = vol_probs

            prob_up = probs[0, 0].item()
            prob_down = probs[0, 1].item()

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={
                "encoded_norm": aux["encoded"].norm().item(),
            },
            metrics={
                "var_weights_mean": aux["var_weights"].mean().item(),
            },
        )

    def _build_pattern_aggregates(self, ctx: PatternContext) -> torch.Tensor:
        """Build pattern aggregates tensor from PatternContext."""
        agg = torch.tensor([[
            ctx.accuracy,
            ctx.up_accuracy,
            ctx.down_accuracy,
            ctx.trust_confidence,
            1.0 if ctx.bias == "UP" else (0.0 if ctx.bias == "DOWN" else 0.5),
            1.0 if ctx.overconfident else 0.0,
            ctx.confidence_cap or 0.65,
            ctx.match_ratio,
            0.0,  # placeholder for uses_count
            0.0,  # placeholder for recency
        ]], dtype=torch.float32, device=self._model._device)
        return agg

    def _get_pattern_events(self, pattern_id: int) -> Optional[torch.Tensor]:
        """Get pattern events tensor (placeholder - implement with PatternStore)."""
        # TODO: Integrate with PatternStore to get actual events
        return None


class FeatureBuffer:
    """Maintains sliding window of features for TFT model."""

    def __init__(self, seq_len: int = 100, device: torch.device = torch.device("cpu")):
        self.seq_len = seq_len
        self.device = device
        self.buffer: List[Dict[str, float]] = []

    def add(self, features: Dict[str, float]) -> None:
        """Add features to buffer."""
        self.buffer.append(features.copy())
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def is_ready(self) -> bool:
        """Check if buffer has enough history."""
        return len(self.buffer) >= self.seq_len

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert buffer to tensors for model input.

        Returns:
            sequence: [1, seq_len, n_features]
            trend_features: [1, n_trend]
            oscillator_features: [1, n_osc]
            volume_features: [1, n_vol]
        """
        if not self.is_ready():
            raise ValueError("Buffer not ready")

        # Build sequence tensor
        sequence_data = []
        for feat_dict in self.buffer:
            row = [feat_dict.get(f, 0.0) for f in ALL_FEATURES]
            sequence_data.append(row)

        sequence = torch.tensor([sequence_data], dtype=torch.float32, device=self.device)

        # Get latest features for heads
        latest = self.buffer[-1]
        trend_feat = torch.tensor(
            [[latest.get(f, 0.0) for f in TREND_FEATURES]],
            dtype=torch.float32,
            device=self.device,
        )
        osc_feat = torch.tensor(
            [[latest.get(f, 0.0) for f in OSCILLATOR_FEATURES]],
            dtype=torch.float32,
            device=self.device,
        )
        vol_feat = torch.tensor(
            [[latest.get(f, 0.0) for f in VOLUME_FEATURES]],
            dtype=torch.float32,
            device=self.device,
        )

        return sequence, trend_feat, osc_feat, vol_feat


def create_tft_models(
    config: ConfigStore,
    device: Optional[torch.device] = None,
) -> Tuple["TFTModelWrapper", "TFTModelWrapper", "TFTModelWrapper", ThreeHeadTFT]:
    """Create three TFT model wrappers sharing a single ThreeHeadTFT.

    Args:
        config: Configuration store
        device: Device to use (auto-detect if None)

    Returns:
        trend_model: TrendML model wrapper
        oscillator_model: OscillatorML model wrapper
        volume_model: VolumeML model wrapper
        tft_model: Underlying ThreeHeadTFT model (for training)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create shared model
    tft_model = ThreeHeadTFT(
        config=config,
        input_dim=len(ALL_FEATURES),
        hidden_dim=int(config.get("tft.hidden_dim", 64)),
        num_heads=int(config.get("tft.num_heads", 4)),
        num_lstm_layers=int(config.get("tft.num_lstm_layers", 2)),
        dropout=float(config.get("tft.dropout", 0.2)),
        num_pattern_events=int(config.get("tft.num_pattern_events", 50)),
    ).to(device)

    # Create shared feature buffer
    seq_len = int(config.get("tft.seq_len", 100))
    feature_buffer = FeatureBuffer(seq_len=seq_len, device=device)

    # Create wrappers for each head
    trend_model = TFTModelWrapper(config, "TRENDML", tft_model, feature_buffer)
    oscillator_model = TFTModelWrapper(config, "OSCILLATORML", tft_model, feature_buffer)
    volume_model = TFTModelWrapper(config, "VOLUMEML", tft_model, feature_buffer)

    return trend_model, oscillator_model, volume_model, tft_model


class TwoHeadMLPWrapper(BaseModel):
    """Wrapper to use TwoHeadMLP with existing Titan interface.

    Implements BaseModel interface for compatibility with Ensemble.
    TwoHeadMLP has only 2 heads (trend + aux), so we expose both
    as separate models for the ensemble.
    """

    def __init__(
        self,
        config: ConfigStore,
        head_name: str,
        model: TwoHeadMLP,
        feature_buffer: "FeatureBuffer",
        session_idx: int = 0,
    ):
        self.name = head_name
        self._config = config
        self._model = model
        self._feature_buffer = feature_buffer
        self._head_name = head_name
        self._session_idx = session_idx  # Default session (0=ASIA)

    def set_session(self, session: str) -> None:
        """Set current session for session-aware models."""
        self._session_idx = SESSION_TO_IDX.get(session, 0)

    def predict(
        self,
        features: Dict[str, float],
        pattern_context: Optional["PatternContext"] = None,
    ) -> "ModelOutput":
        """Make prediction using TwoHeadMLP model.

        Args:
            features: Current market features
            pattern_context: Pattern context (unused for TwoHeadMLP)

        Returns:
            ModelOutput with probabilities
        """
        from titan.core.types import ModelOutput

        # Add features to buffer and get sequence
        self._feature_buffer.add(features)

        # Check if we have enough history
        if not self._feature_buffer.is_ready():
            return ModelOutput(
                model_name=self.name,
                prob_up=0.5,
                prob_down=0.5,
                state={"buffer_filling": True},
                metrics={},
            )

        # Get tensors from buffer
        with torch.no_grad():
            sequence, trend_feat, osc_feat, vol_feat = self._feature_buffer.get_tensors()

            # Create session index tensor
            session_idx_tensor = torch.tensor(
                [self._session_idx], dtype=torch.long, device=sequence.device
            )

            # Forward pass
            if isinstance(self._model, (SessionGatedMLP, SessionEmbeddingMLP)):
                trend_probs, aux_probs, _, aux = self._model(
                    sequence, trend_feat, osc_feat, vol_feat, session_idx_tensor
                )
            else:
                trend_probs, aux_probs, _, aux = self._model(
                    sequence, trend_feat, osc_feat, vol_feat
                )

            # Get probabilities for this head
            if self._head_name == "TRENDML":
                probs = trend_probs[0]  # [batch=1, 2] -> [2]
            else:  # AUXML
                probs = aux_probs[0]

            prob_up = probs[1].item()
            prob_down = probs[0].item()

        return ModelOutput(
            model_name=self.name,
            prob_up=prob_up,
            prob_down=prob_down,
            state={"head": self._head_name, "session_idx": self._session_idx},
            metrics={"confidence": max(prob_up, prob_down)},
        )


def create_two_head_models(
    config: ConfigStore,
    device: Optional[torch.device] = None,
    model_class: str = "TwoHeadMLP",
    checkpoint_path: Optional[str] = None,
) -> Tuple["TwoHeadMLPWrapper", "TwoHeadMLPWrapper", "TwoHeadMLP"]:
    """Create two model wrappers for TwoHeadMLP.

    Args:
        config: Configuration store
        device: Device to use (auto-detect if None)
        model_class: Model class to use: "TwoHeadMLP", "SessionGatedMLP", or "SessionEmbeddingMLP"
        checkpoint_path: Optional path to load model checkpoint

    Returns:
        trend_model: TrendML model wrapper
        aux_model: AuxML model wrapper (combines oscillator + volume)
        raw_model: Underlying TwoHeadMLP model (for reference)
    """
    import os

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model based on class selection
    hidden_dim = int(config.get("tft.hidden_dim", 64))
    num_lstm_layers = int(config.get("tft.num_lstm_layers", 1))
    # Gemini recommendation: High dropout (0.4) to reduce overfitting on crypto data
    dropout = float(config.get("tft.dropout", 0.4))

    if model_class == "SessionGatedMLP":
        raw_model = SessionGatedMLP(
            config=config,
            input_dim=len(ALL_FEATURES),
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        ).to(device)
    elif model_class == "SessionEmbeddingMLP":
        raw_model = SessionEmbeddingMLP(
            config=config,
            input_dim=len(ALL_FEATURES),
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            session_embed_dim=int(config.get("tft.session_embed_dim", 8)),
        ).to(device)
    else:  # Default: TwoHeadMLP
        raw_model = TwoHeadMLP(
            config=config,
            input_dim=len(ALL_FEATURES),
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        ).to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[TwoHeadMLP] Loaded checkpoint: {checkpoint_path}")

    raw_model.eval()

    # Create shared feature buffer
    seq_len = int(config.get("tft.seq_len", 100))
    feature_buffer = FeatureBuffer(seq_len=seq_len, device=device)

    # Create wrappers for each head
    trend_model = TwoHeadMLPWrapper(config, "TRENDML", raw_model, feature_buffer)
    aux_model = TwoHeadMLPWrapper(config, "AUXML", raw_model, feature_buffer)

    return trend_model, aux_model, raw_model
