"""Wrapper around RiboNN to extract latent embeddings and predictions.

RiboNN architecture (relevant layers):
    initial_conv -> middle_convs -> head

    head = Sequential(
        activation,                                          # 0
        Flatten,                                             # 1
        Dropout,                                             # 2
        Linear(filters * len_after_conv, filters, bias=F),   # 3  <-- bottleneck
        activation,                                          # 4  <-- HOOK HERE
        BatchNorm1d(filters),                                # 5
        Dropout,                                             # 6
        Linear(filters, num_targets),                        # 7
    )

We hook into layer 4 to extract the filters-dim (typically 64) embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from rnaflow.data.encoding import one_hot_encode_ribonn


class RiboNNWrapper:
    """Wraps a pretrained RiboNN model for embedding extraction and prediction.

    Usage:
        wrapper = RiboNNWrapper.from_checkpoint("path/to/checkpoint.ckpt")
        z = wrapper.encode(one_hot_tensor)          # (B, 64)
        te = wrapper.predict(one_hot_tensor)         # (B, num_targets)
        z, te = wrapper.encode_and_predict(tensor)   # both at once
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        # Disable stochastic shifting during inference (RiboNN calls this
        # even in eval mode if max_shift > 0)
        if hasattr(self.model, "max_shift"):
            self._original_max_shift = self.model.max_shift
            self.model.max_shift = 0

        # Read model hparams if available (PyTorch Lightning stores these)
        self._hparams = {}
        if hasattr(self.model, "hparams"):
            self._hparams = dict(self.model.hparams)

        self._embedding: Optional[Tensor] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register a forward hook on the head's bottleneck activation.

        head[4] is the activation (ReLU) after the bottleneck Linear — its
        output shape is (B, filters) which is our latent embedding.
        """
        head = self.model.head
        bottleneck_activation = head[4]

        def hook_fn(module, input, output):
            self._embedding = output.detach()

        self._hook_handle = bottleneck_activation.register_forward_hook(hook_fn)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        ribonn_class: Optional[type] = None,
    ) -> "RiboNNWrapper":
        """Load a RiboNN model from a PyTorch Lightning .ckpt checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file.
            device: Device to load model on.
            ribonn_class: The RiboNN class. If None, tries to import from RiboNN's src.
        """
        if ribonn_class is None:
            try:
                from model import RiboNN
                ribonn_class = RiboNN
            except ImportError:
                raise ImportError(
                    "Could not import RiboNN. Make sure RiboNN/src is in your PYTHONPATH. "
                    "Run: export PYTHONPATH=/path/to/RiboNN/src:$PYTHONPATH"
                )

        model = ribonn_class.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        return cls(model=model, device=device)

    @classmethod
    def from_state_dict(
        cls,
        state_dict_path: str | Path,
        config: dict | None = None,
        device: str = "cpu",
        ribonn_class: Optional[type] = None,
    ) -> "RiboNNWrapper":
        """Load a RiboNN model from a state_dict.pth file (Zenodo pretrained weights).

        Infers model architecture from the state_dict if config is not provided.

        Args:
            state_dict_path: Path to state_dict.pth file.
            config: RiboNN config dict. If None, infers from default + state_dict shapes.
            device: Device to load model on.
            ribonn_class: The RiboNN class.
        """
        import torch

        if ribonn_class is None:
            try:
                from model import RiboNN
                ribonn_class = RiboNN
            except ImportError:
                raise ImportError(
                    "Could not import RiboNN. Make sure RiboNN/src is in your PYTHONPATH. "
                    "Run: export PYTHONPATH=/path/to/RiboNN/src:$PYTHONPATH"
                )

        state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)

        if config is None:
            config = cls._infer_config_from_state_dict(state_dict)

        model = ribonn_class(**config)
        model.load_state_dict(state_dict)
        return cls(model=model, device=device)

    @staticmethod
    def _infer_config_from_state_dict(state_dict: dict) -> dict:
        """Infer RiboNN hyperparameters from state_dict tensor shapes.

        Uses the pretrained weights' shapes to reconstruct the config needed
        to instantiate the model.
        """
        # initial_conv.conv.weight: (filters, in_channels, kernel_size)
        init_conv_w = state_dict["initial_conv.conv.weight"]
        filters = init_conv_w.shape[0]
        in_channels = init_conv_w.shape[1]
        initial_kernel_size = init_conv_w.shape[2]

        # Count middle conv layers
        n_middle = sum(1 for k in state_dict if k.startswith("middle_convs.") and k.endswith(".conv.1.weight"))

        # Middle conv kernel size
        middle_conv_key = "middle_convs.0.conv.1.weight"
        kernel_size = state_dict[middle_conv_key].shape[2]

        # head.3.weight: (filters, filters * len_after_conv) — bottleneck
        head_linear_w = state_dict["head.3.weight"]
        len_after_conv = head_linear_w.shape[1] // filters

        # head.7.weight: (num_targets, filters) — output
        num_targets = state_dict["head.7.weight"].shape[0]

        # Determine input channel composition
        label_codons = (in_channels == 5)  # 4 nuc + 1 codon label

        # Compute max_seq_len from len_after_conv by reversing the convolution math.
        # Each middle conv: seq_len = ((seq_len + 2*0 - 1*(k-1) - 1) // 1 + 1) then MaxPool(2,2)
        # Initial conv: seq_len = ((seq_len + 2*0 - 1*(5-1) - 1) // 1 + 1)  [kernel=5 always]
        # Working backwards from len_after_conv:
        seq_len = len_after_conv
        for _ in range(n_middle):
            seq_len = seq_len * 2           # undo MaxPool
            seq_len = seq_len + kernel_size - 1  # undo Conv1d(padding=0)
        seq_len = seq_len + initial_kernel_size - 1  # undo initial Conv1d(padding=0)

        # Default human config values matching RiboNN's training setup
        config = {
            "filters": filters,
            "kernel_size": kernel_size,
            "num_conv_layers": n_middle,
            "num_targets": num_targets,
            "len_after_conv": len_after_conv,
            "max_seq_len": seq_len,
            "conv_stride": 1,
            "conv_padding": 0,
            "conv_dilation": 1,
            "dropout": 0.3,
            "ln_epsilon": 0.007,
            "bn_momentum": 0.9,
            "residual": False,
            "max_shift": 0,
            "symmetric_shift": True,
            "augmentation_shifts": [-3, -2, -1, 0, 1, 2, 3],
            "label_codons": label_codons,
            "label_3rd_nt_of_codons": False,
            "label_utr5": False,
            "label_utr3": False,
            "label_splice_sites": False,
            "label_up_probs": False,
            "split_utr5_cds_utr3_channels": False,
            "with_NAs": True,
            "pad_5_prime": False,
            "go_backwards": True,
            "lr": 0.0001,
            "min_lr": 1e-7,
            "l2_scale": 0.001,
            "adam_beta1": 0.90,
            "adam_beta2": 0.998,
        }

        return config

    @classmethod
    def from_model(cls, model: nn.Module, device: str = "cpu") -> "RiboNNWrapper":
        """Wrap an already-loaded RiboNN model."""
        return cls(model=model, device=device)

    @property
    def label_codons(self) -> bool:
        """Whether the model expects codon label channel."""
        return self._hparams.get("label_codons", False)

    @property
    def input_channels(self) -> int:
        """Number of input channels the model expects."""
        seq_ch = 12 if self._hparams.get("split_utr5_cds_utr3_channels", False) else 4
        label_ch = sum([
            self._hparams.get("label_codons", False),
            self._hparams.get("label_utr5", False),
            self._hparams.get("label_utr3", False),
            self._hparams.get("label_splice_sites", False),
            self._hparams.get("label_up_probs", False),
        ])
        return seq_ch + label_ch

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent embedding (typically 64)."""
        for layer in self.model.head:
            if isinstance(layer, nn.Linear):
                return layer.out_features
        raise RuntimeError("Could not determine latent dim from model head")

    @property
    def num_targets(self) -> int:
        """Number of prediction targets (cell types / samples)."""
        layers = list(self.model.head)
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        raise RuntimeError("Could not determine num_targets from model head")

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length the model was trained on."""
        return self._hparams.get("max_seq_len", 12288)

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Extract latent embeddings from encoded sequences.

        Args:
            x: (B, C, L) encoded input tensor (C=4 or 5 depending on model).

        Returns:
            z: (B, latent_dim) embedding tensor.
        """
        x = x.to(self.device)
        _ = self.model(x)  # triggers the hook
        return self._embedding.clone()

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Get translation efficiency predictions.

        Args:
            x: (B, C, L) encoded input tensor.

        Returns:
            te: (B, num_targets) predicted translation efficiencies.
        """
        x = x.to(self.device)
        return self.model(x)

    @torch.no_grad()
    def encode_and_predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Get both embeddings and predictions in a single forward pass."""
        x = x.to(self.device)
        te = self.model(x)
        return self._embedding.clone(), te

    def encode_sequence(
        self,
        seq: str,
        max_len: int | None = None,
        utr5_size: int = 0,
        cds_size: int = 0,
    ) -> Tensor:
        """Convenience: encode a raw nucleotide string to latent space.

        Args:
            seq: mRNA sequence string (5'UTR + CDS + 3'UTR).
            max_len: Pad length. Defaults to model's max_seq_len.
            utr5_size: Length of 5'UTR region.
            cds_size: Length of CDS region (including start+stop codons).

        Returns:
            z: (latent_dim,) embedding vector.
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
        """Convenience: predict TE from a raw nucleotide string.

        Returns:
            te: (num_targets,) prediction vector.
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

    def encode_with_grad(self, x: Tensor) -> Tensor:
        """Encode with gradients enabled (for gradient-based inversion).

        Args:
            x: (B, C, L) input tensor (can require grad).

        Returns:
            z: (B, latent_dim) embedding with gradient graph attached.
        """
        x = x.to(self.device)
        # Run through conv backbone
        h = self.model.initial_conv(x)
        for conv in self.model.middle_convs:
            h = conv(h)
        # Run through head layers 0-4 (up to and including bottleneck activation)
        head = self.model.head
        for i, layer in enumerate(head):
            h = layer(h)
            if i == 4:  # bottleneck activation output
                return h
        raise RuntimeError("Could not extract embedding with gradients")

    def predict_with_grad(self, x: Tensor) -> Tensor:
        """Full forward pass with gradients enabled (for objective-aware inversion).

        Args:
            x: (B, C, L) input tensor (can require grad).

        Returns:
            te: (B, num_targets) predicted translation efficiencies with grad graph.
        """
        x = x.to(self.device)
        h = self.model.initial_conv(x)
        for conv in self.model.middle_convs:
            h = conv(h)
        return self.model.head(h)

    def cleanup(self):
        """Remove the forward hook and restore original max_shift."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if hasattr(self, "_original_max_shift"):
            self.model.max_shift = self._original_max_shift


class MockRiboNN(nn.Module):
    """A mock RiboNN model for testing without the real pretrained weights.

    Mimics RiboNN's architecture: conv backbone -> head with 64-dim bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 4,
        filters: int = 64,
        kernel_size: int = 5,
        n_conv_layers: int = 4,
        seq_len: int = 2048,
        num_targets: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.filters = filters
        self.max_shift = 0

        # Simplified conv backbone (no pooling — keeps seq_len constant)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm([filters, seq_len]),
            nn.ReLU(),
        )

        self.middle_convs = nn.ModuleList()
        for _ in range(n_conv_layers - 1):
            self.middle_convs.append(nn.Sequential(
                nn.Conv1d(filters, filters, kernel_size, padding=kernel_size // 2),
                nn.LayerNorm([filters, seq_len]),
                nn.ReLU(),
            ))

        self.len_after_conv = seq_len

        # Head matching RiboNN's exact structure
        self.head = nn.Sequential(
            nn.ReLU(),                                              # 0: activation
            nn.Flatten(),                                           # 1: flatten
            nn.Dropout(dropout),                                    # 2: dropout
            nn.Linear(filters * seq_len, filters),                  # 3: bottleneck
            nn.ReLU(),                                              # 4: activation (HOOK)
            nn.BatchNorm1d(filters),                                # 5: batchnorm
            nn.Dropout(dropout),                                    # 6: dropout
            nn.Linear(filters, num_targets),                        # 7: output
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.initial_conv(x)
        for conv in self.middle_convs:
            h = conv(h)
        return self.head(h)
