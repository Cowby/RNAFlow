"""Objective functions for mRNA sequence optimization.

Two approaches:
  A) Direct RiboNN outputs: different output columns = different samples/cell types
  B) Custom predictor: trained cell-type-conditioned MLP on frozen embeddings
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class RiboNNSpecificityObjective:
    """Specificity objective using RiboNN's multi-target TE predictions directly.

    Maximizes: TE[target_col] - lambda * mean(TE[off_target_cols])

    This requires passing through the FULL RiboNN model (not just embeddings),
    so it operates on one-hot encoded sequences or uses the wrapper's encode_with_grad
    path to evaluate latent vectors.

    Args:
        wrapper: RiboNNWrapper instance.
        target_col: Index of the target cell type in RiboNN's output.
        off_target_cols: Indices of off-target cell types.
        lam: Weight for the off-target penalty.
    """

    def __init__(
        self,
        wrapper,  # RiboNNWrapper
        target_col: int,
        off_target_cols: list[int],
        lam: float = 1.0,
    ):
        self.wrapper = wrapper
        self.target_col = target_col
        self.off_target_cols = off_target_cols
        self.lam = lam

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate specificity on one-hot encoded sequences.

        Args:
            x: (N, 4, L) batch of one-hot encoded sequences.

        Returns:
            scores: (N,) specificity scores (higher is better).
        """
        te = self.wrapper.predict(x)  # (N, num_targets)
        target_eff = te[:, self.target_col]

        off_target_eff = te[:, self.off_target_cols].mean(dim=1)

        return target_eff - self.lam * off_target_eff


class LatentRiboNNObjective:
    """Specificity objective using RiboNN's head tail directly on latent vectors.

    Instead of running the full RiboNN model (which expects sequence input),
    this runs latent vectors z through the tail of the head:
        head[5] = BatchNorm1d(filters)
        head[6] = Dropout
        head[7] = Linear(filters, num_targets)

    This produces TE predictions directly from 64-dim latent space.

    Args:
        wrapper: RiboNNWrapper instance (provides access to head layers).
        target_col: Index of the target cell type in RiboNN's output.
        off_target_cols: Indices of off-target cell types.
        lam: Weight for the off-target penalty.
    """

    def __init__(
        self,
        wrapper,  # RiboNNWrapper
        target_col: int,
        off_target_cols: list[int],
        lam: float = 1.0,
    ):
        self.wrapper = wrapper
        self.target_col = target_col
        self.off_target_cols = off_target_cols
        self.lam = lam

        # Extract the tail of the head (layers 5-7: BN -> Dropout -> Linear)
        # For ensembles, use pre-extracted head_tails from all models
        if hasattr(wrapper, 'head_tails'):
            self.head_tails = wrapper.head_tails
            self.head_tail = wrapper.head_tails[0]  # compat for diffusion
        else:
            self.head_tails = None
            head = wrapper.model.head
            self.head_tail = nn.Sequential(
                head[5],  # BatchNorm1d(64)
                head[6],  # Dropout
                head[7],  # Linear(64, num_targets)
            )
            self.head_tail.eval()
            self.head_tail.to(wrapper.device)

    @torch.no_grad()
    def __call__(self, z: Tensor) -> Tensor:
        """Evaluate specificity for latent vectors.

        Args:
            z: (N, latent_dim) batch of latent vectors from CEM.

        Returns:
            scores: (N,) specificity scores (higher is better).
        """
        z = z.to(self.wrapper.device)
        if self.head_tails:
            te = torch.stack([ht(z) for ht in self.head_tails]).mean(dim=0)
        else:
            te = self.head_tail(z)  # (N, num_targets)

        target_eff = te[:, self.target_col]
        off_target_eff = te[:, self.off_target_cols].mean(dim=1)

        return target_eff - self.lam * off_target_eff


class PredictorSpecificityObjective:
    """Specificity objective using a trained cell-type-conditioned predictor.

    Operates directly in latent space — no need to decode back to sequences.

    Args:
        predictor: TranslationPredictor model.
        target_cell: Target cell type ID.
        off_target_cells: List of off-target cell type IDs.
        lam: Weight for the off-target penalty.
        device: Torch device.
    """

    def __init__(
        self,
        predictor: nn.Module,
        target_cell: int,
        off_target_cells: list[int],
        lam: float = 1.0,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.predictor.eval()
        self.target_cell = target_cell
        self.off_target_cells = off_target_cells
        self.lam = lam
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, z: Tensor) -> Tensor:
        """Evaluate specificity for latent vectors.

        Args:
            z: (N, latent_dim) batch of latent vectors.

        Returns:
            scores: (N,) specificity scores (higher is better).
        """
        z = z.to(self.device)
        N = z.shape[0]

        target_eff = self.predictor(
            z, torch.full((N,), self.target_cell, device=self.device, dtype=torch.long)
        )

        off_target_eff = torch.zeros(N, device=self.device)
        for ct in self.off_target_cells:
            off_target_eff += self.predictor(
                z, torch.full((N,), ct, device=self.device, dtype=torch.long)
            )
        off_target_eff /= len(self.off_target_cells)

        return target_eff - self.lam * off_target_eff


class CombinedObjective:
    """Combine multiple objectives with weights.

    Args:
        objectives: List of (weight, objective_fn) tuples.
    """

    def __init__(self, objectives: list[tuple[float, callable]]):
        self.objectives = objectives

    def __call__(self, z: Tensor) -> Tensor:
        total = torch.zeros(z.shape[0], device=z.device)
        for weight, obj_fn in self.objectives:
            total += weight * obj_fn(z)
        return total
