"""One-hot encoding/decoding for mRNA sequences, matching RiboNN's format."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# RiboNN convention: A=0, T=1, C=2, G=3
NUC_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3, "U": 1}
IDX_TO_NUC = {0: "A", 1: "U", 2: "C", 3: "G"}


def one_hot_encode(seq: str, max_len: int | None = None) -> tuple[Tensor, Tensor]:
    """Encode an mRNA sequence as a one-hot tensor.

    Args:
        seq: Nucleotide string (A/U/G/C/T). T is treated as U.
        max_len: Pad/truncate to this length. If None, uses len(seq).

    Returns:
        one_hot: (4, L) float tensor
        mask: (L,) bool tensor — True for real positions, False for padding
    """
    seq = seq.upper().replace("T", "U") if "T" in seq.upper() else seq.upper()
    seq_len = len(seq)
    L = max_len if max_len is not None else seq_len
    seq = seq[:L]  # truncate if needed

    one_hot = torch.zeros(4, L, dtype=torch.float32)
    mask = torch.zeros(L, dtype=torch.bool)

    for i, nuc in enumerate(seq):
        idx = NUC_TO_IDX.get(nuc)
        if idx is not None:
            one_hot[idx, i] = 1.0
            mask[i] = True

    return one_hot, mask


def one_hot_encode_ribonn(
    seq: str,
    max_len: int,
    utr5_size: int = 0,
    cds_size: int = 0,
    label_codons: bool = False,
) -> Tensor:
    """Encode in RiboNN's exact format with A=0,T=1,C=2,G=3.

    Uses T instead of U in internal representation to match RiboNN's base_index.

    Args:
        seq: Full transcript sequence (5'UTR + CDS + 3'UTR).
        max_len: Pad to this total length.
        utr5_size: Length of 5'UTR (needed for codon labeling).
        cds_size: Length of CDS including start+stop codons (needed for codon labeling).
        label_codons: If True, add a 5th channel marking codon start positions in CDS.

    Returns:
        Tensor of shape (n_channels, max_len) where n_channels=4 or 5.
    """
    seq = seq.upper().replace("U", "T")[:max_len]
    base_index = {"A": 0, "T": 1, "C": 2, "G": 3}

    n_channels = 5 if label_codons else 4
    x = torch.zeros(n_channels, max_len, dtype=torch.float32)

    # One-hot encode nucleotides (channels 0-3)
    for i, nuc in enumerate(seq):
        if nuc in base_index:
            x[base_index[nuc], i] = 1.0

    # Codon labels (channel 4): mark the first nucleotide of each codon in CDS
    if label_codons and cds_size > 0:
        cds_start = utr5_size
        cds_end = utr5_size + cds_size - 3  # exclude stop codon
        for pos in range(cds_start, cds_end + 1, 3):
            if pos < max_len:
                x[4, pos] = 1.0

    return x


def decode_logits(logits: Tensor) -> str:
    """Decode logits (4, L) to a nucleotide string via argmax."""
    indices = logits.argmax(dim=0)
    return "".join(IDX_TO_NUC[i.item()] for i in indices)


def soft_to_one_hot(soft: Tensor, temperature: float = 1.0) -> Tensor:
    """Apply Gumbel-softmax to soft logits (4, L) for differentiable discretization."""
    # Transpose to (L, 4) for gumbel_softmax, then back
    return F.gumbel_softmax(
        soft.T, tau=temperature, hard=False, dim=-1
    ).T


def sequence_entropy(soft: Tensor) -> Tensor:
    """Compute per-position entropy of a soft sequence (4, L). Lower = more discrete."""
    probs = F.softmax(soft, dim=0)
    log_probs = F.log_softmax(soft, dim=0)
    entropy = -(probs * log_probs).sum(dim=0)  # (L,)
    return entropy.mean()
