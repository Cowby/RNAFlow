"""Cell-type-conditioned translation efficiency predictor.

Trained on top of frozen RiboNN latent embeddings.
Input: (z, cell_type) -> predicted translation efficiency.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class TranslationPredictor(nn.Module):
    """MLP that predicts cell-type-specific translation efficiency from latent embeddings.

    Architecture: [z || cell_emb] -> Linear -> ReLU -> BN -> Dropout
                                   -> Linear -> ReLU -> BN -> Dropout
                                   -> Linear -> Sigmoid

    Args:
        latent_dim: Dimensionality of the input embedding (64 for RiboNN).
        n_cell_types: Number of distinct cell types.
        cell_embed_dim: Dimensionality of the learned cell type embedding.
        hidden_dims: Hidden layer sizes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_cell_types: int = 10,
        cell_embed_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.latent_dim = latent_dim
        self.n_cell_types = n_cell_types
        self.cell_embed_dim = cell_embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.cell_embedding = nn.Embedding(n_cell_types, cell_embed_dim)

        layers = []
        in_dim = latent_dim + cell_embed_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: Tensor, cell_type: Tensor) -> Tensor:
        """Predict translation efficiency.

        Args:
            z: (B, latent_dim) latent embeddings.
            cell_type: (B,) integer cell type IDs.

        Returns:
            efficiency: (B,) predicted TE in [0, 1].
        """
        cell_emb = self.cell_embedding(cell_type)  # (B, cell_embed_dim)
        h = torch.cat([z, cell_emb], dim=-1)       # (B, latent_dim + cell_embed_dim)
        return self.mlp(h).squeeze(-1)              # (B,)

    def save(self, path: str | Path):
        """Save model weights and config."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "latent_dim": self.latent_dim,
                "n_cell_types": self.n_cell_types,
                "cell_embed_dim": self.cell_embed_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TranslationPredictor":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
