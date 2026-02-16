"""
Apex Predator — Perception Module (Layer 1: The Eyes)
======================================================
LSTM with optional Attention for pattern memory.
Encodes raw feature sequences into latent state embeddings.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config.settings import PerceptionConfig

logger = logging.getLogger("apex_predator.perception")


# ──────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Simple self-attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            Context vector: (batch, hidden_dim)
        """
        # Attention weights: (batch, seq_len, 1)
        weights = self.attention(lstm_output)
        weights = torch.softmax(weights, dim=1)

        # Weighted sum: (batch, hidden_dim)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


# ──────────────────────────────────────────────
# LSTM Perception Network
# ──────────────────────────────────────────────

class PerceptionNetwork(nn.Module):
    """
    LSTM-based sequence encoder with optional attention.

    Input:  (batch, sequence_length, input_dim)
    Output: (batch, embedding_dim) — latent state embedding
    """

    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.config = config

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Optional attention
        self.use_attention = config.use_attention
        if self.use_attention:
            self.attention = SelfAttention(config.hidden_dim)

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            embedding: (batch, embedding_dim)
        """
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.use_attention:
            # Attention over all timesteps
            context = self.attention(lstm_out)
        else:
            # Use last hidden state
            context = h_n[-1]  # (batch, hidden_dim)

        # Project to embedding
        embedding = self.projection(context)
        return embedding


# ──────────────────────────────────────────────
# Perception Module (High-level Interface)
# ──────────────────────────────────────────────

class PerceptionModule:
    """
    High-level interface for the perception subsystem.
    Manages the LSTM network, device placement, and inference.
    """

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.network = PerceptionNetwork(config).to(self.device)
        self.network.eval()
        logger.info(
            "PerceptionModule initialised — device=%s | params=%d",
            self.device,
            sum(p.numel() for p in self.network.parameters()),
        )

    # ── Inference ───────────────────────────────

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode a sequence of feature vectors into a latent embedding.

        Args:
            features: np.ndarray of shape (seq_len, input_dim) or
                      (batch, seq_len, input_dim)

        Returns:
            np.ndarray of shape (embedding_dim,) or (batch, embedding_dim)
        """
        single = features.ndim == 2
        if single:
            features = features[np.newaxis, ...]  # Add batch dim

        # Truncate/pad to expected sequence length
        seq_len = self.config.sequence_length
        if features.shape[1] > seq_len:
            features = features[:, -seq_len:, :]
        elif features.shape[1] < seq_len:
            pad_width = seq_len - features.shape[1]
            padding = np.zeros((features.shape[0], pad_width, features.shape[2]))
            features = np.concatenate([padding, features], axis=1)

        tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            embedding = self.network(tensor)

        result = embedding.cpu().numpy()
        return result[0] if single else result

    # ── Training Support ────────────────────────

    def get_network(self) -> PerceptionNetwork:
        """Get the underlying PyTorch network (for joint training)."""
        return self.network

    def train_mode(self):
        """Switch to training mode."""
        self.network.train()

    def eval_mode(self):
        """Switch to evaluation mode."""
        self.network.eval()

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)
        logger.info("Perception model saved to %s", path)

    def load(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        logger.info("Perception model loaded from %s", path)

    # ── Helpers ──────────────────────────────────

    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
