"""Ensemble wrapper for multiple RiboNN cross-validation models.

The 91 pretrained RiboNN models are nested cross-validation folds (10 test x 9
validation). Each model predicts all 78 cell types. The RiboNN authors ensemble
them by selecting the top-K models by validation R2 and averaging predictions.

This wrapper loads multiple models and averages their outputs, presenting the
same interface as RiboNNWrapper so all downstream code works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from rnaflow.data.encoding import one_hot_encode_ribonn
from rnaflow.embeddings.ribonn_wrapper import RiboNNWrapper


class EnsembleRiboNNWrapper:
    """Wraps multiple RiboNN models for ensembled prediction.

    Averages predictions across models for more robust TE estimates.
    Uses a single primary model for gradient-enabled operations (encoding,
    inversion) since backprop through multiple full CNNs is too expensive.

    Args:
        wrappers: List of RiboNNWrapper instances to ensemble.
    """

    def __init__(self, wrappers: list[RiboNNWrapper]):
        if not wrappers:
            raise ValueError("Need at least one wrapper for ensemble")
        self.wrappers = wrappers
        self.primary = wrappers[0]

        # Extract head_tails (layers 5-7) from each model for latent objective
        self.head_tails: list[nn.Sequential] = []
        for w in wrappers:
            head = w.model.head
            ht = nn.Sequential(head[5], head[6], head[7])
            ht.eval()
            ht.to(w.device)
            self.head_tails.append(ht)

    @classmethod
    def from_runs_csv(
        cls,
        runs_csv_path: str | Path,
        checkpoints_dir: str | Path,
        top_k: int = 5,
        device: str = "cpu",
    ) -> "EnsembleRiboNNWrapper":
        """Load top-K models ranked by validation R2 from runs.csv.

        This matches the RiboNN authors' ensembling strategy: select the
        best models by metrics.val_r2 and average their predictions.

        Args:
            runs_csv_path: Path to runs.csv with model metadata.
            checkpoints_dir: Directory containing <run_id>/state_dict.pth.
            top_k: Number of top models to load.
            device: Torch device.
        """
        runs_csv_path = Path(runs_csv_path)
        checkpoints_dir = Path(checkpoints_dir)

        df = pd.read_csv(runs_csv_path)
        df = df.sort_values("metrics.val_r2", ascending=False)

        # Only keep models whose state_dict actually exists
        wrappers = []
        for _, row in df.iterrows():
            if len(wrappers) >= top_k:
                break
            run_id = row["run_id"]
            state_dict_path = checkpoints_dir / run_id / "state_dict.pth"
            if state_dict_path.exists():
                wrapper = RiboNNWrapper.from_state_dict(
                    state_dict_path, device=device
                )
                wrappers.append(wrapper)

        if not wrappers:
            raise FileNotFoundError(
                f"No state_dict.pth files found in {checkpoints_dir}"
            )

        print(f"  Loaded ensemble of {len(wrappers)} models "
              f"(top {top_k} by val_r2)")

        return cls(wrappers)

    @classmethod
    def from_directory(
        cls,
        checkpoints_dir: str | Path,
        max_models: int = 5,
        device: str = "cpu",
    ) -> "EnsembleRiboNNWrapper":
        """Fallback: load first N models from directory (no runs.csv).

        Args:
            checkpoints_dir: Directory containing <run_id>/state_dict.pth.
            max_models: Maximum number of models to load.
            device: Torch device.
        """
        checkpoints_dir = Path(checkpoints_dir)
        candidates = sorted(checkpoints_dir.glob("*/state_dict.pth"))

        if not candidates:
            raise FileNotFoundError(
                f"No state_dict.pth files found in {checkpoints_dir}"
            )

        wrappers = []
        for path in candidates[:max_models]:
            wrapper = RiboNNWrapper.from_state_dict(path, device=device)
            wrappers.append(wrapper)

        print(f"  Loaded ensemble of {len(wrappers)} models from directory")

        return cls(wrappers)

    # ── Properties (delegate to primary) ──────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self.primary.model

    @property
    def device(self) -> torch.device:
        return self.primary.device

    @property
    def latent_dim(self) -> int:
        return self.primary.latent_dim

    @property
    def num_targets(self) -> int:
        return self.primary.num_targets

    @property
    def max_seq_len(self) -> int:
        return self.primary.max_seq_len

    @property
    def input_channels(self) -> int:
        return self.primary.input_channels

    @property
    def label_codons(self) -> bool:
        return self.primary.label_codons

    # ── Ensembled inference ───────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Average embeddings across all models.

        Args:
            x: (B, C, L) encoded input tensor.

        Returns:
            z: (B, latent_dim) averaged embedding.
        """
        embeddings = [w.encode(x) for w in self.wrappers]
        return torch.stack(embeddings).mean(dim=0)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Average predictions across all models.

        Args:
            x: (B, C, L) encoded input tensor.

        Returns:
            te: (B, num_targets) averaged predictions.
        """
        predictions = [w.predict(x) for w in self.wrappers]
        return torch.stack(predictions).mean(dim=0)

    @torch.no_grad()
    def encode_and_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Get both averaged embeddings and predictions."""
        embeddings = []
        predictions = []
        for w in self.wrappers:
            z, te = w.encode_and_predict(x)
            embeddings.append(z)
            predictions.append(te)
        return (
            torch.stack(embeddings).mean(dim=0),
            torch.stack(predictions).mean(dim=0),
        )

    def encode_sequence(
        self,
        seq: str,
        max_len: int | None = None,
        utr5_size: int = 0,
        cds_size: int = 0,
    ) -> Tensor:
        """Encode a raw nucleotide string, averaging across models.

        Returns:
            z: (latent_dim,) averaged embedding vector.
        """
        if max_len is None:
            max_len = self.max_seq_len
        x = one_hot_encode_ribonn(
            seq, max_len,
            utr5_size=utr5_size,
            cds_size=cds_size,
            label_codons=self.label_codons,
        ).unsqueeze(0)
        return self.encode(x).squeeze(0)

    def predict_sequence(
        self,
        seq: str,
        max_len: int | None = None,
        utr5_size: int = 0,
        cds_size: int = 0,
    ) -> Tensor:
        """Predict TE from a raw nucleotide string, averaged across models.

        Returns:
            te: (num_targets,) averaged prediction vector.
        """
        if max_len is None:
            max_len = self.max_seq_len
        x = one_hot_encode_ribonn(
            seq, max_len,
            utr5_size=utr5_size,
            cds_size=cds_size,
            label_codons=self.label_codons,
        ).unsqueeze(0)
        return self.predict(x).squeeze(0)

    # ── Gradient-enabled ops (delegate to primary) ────────────────────────

    def encode_with_grad(self, x: Tensor) -> Tensor:
        """Encode with gradients (uses primary model only)."""
        return self.primary.encode_with_grad(x)

    def predict_with_grad(self, x: Tensor) -> Tensor:
        """Predict with gradients (uses primary model only)."""
        return self.primary.predict_with_grad(x)

    def cleanup(self):
        """Clean up all wrapped models."""
        for w in self.wrappers:
            w.cleanup()
