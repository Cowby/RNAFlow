"""Synthetic mRNA dataset with planted cell-type-specific motifs for prototyping."""

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rnaflow.data.encoding import one_hot_encode_ribonn

# Cell-type-specific motifs: sequences containing these motifs will have
# higher translation efficiency in the corresponding cell type
DEFAULT_MOTIFS = {
    0: "AUGCAUGCAUGC",   # cell type 0 responds to this motif
    1: "GCGCGCGCGCGC",   # cell type 1
    2: "AAAUUUAAAUUU",   # cell type 2
    3: "CCGGCCGGCCGG",   # cell type 3
    4: "UGUGUGUGUG",     # cell type 4
}

NUCLEOTIDES = "AUGC"


def random_mrna(length: int) -> str:
    """Generate a random mRNA sequence of given length."""
    return "".join(random.choice(NUCLEOTIDES) for _ in range(length))


def plant_motif(seq: str, motif: str, position: Optional[int] = None) -> str:
    """Insert a motif into a sequence at a given or random position."""
    if position is None:
        position = random.randint(0, max(0, len(seq) - len(motif)))
    return seq[:position] + motif + seq[position + len(motif):]


class SyntheticMRNADataset(Dataset):
    """Dataset of synthetic mRNA sequences with cell-type-dependent efficiency.

    Each sequence may contain motifs that increase its translation efficiency
    in specific cell types. Efficiency is computed as:
        base_efficiency + motif_bonus * (motif present for this cell type)
        + noise

    Args:
        n_sequences: Number of sequences to generate.
        seq_lengths: Tuple (min_len, max_len) for variable-length sequences.
        max_seq_len: Maximum sequence length for padding (RiboNN input size).
        n_cell_types: Number of cell types to simulate.
        motifs: Dict mapping cell_type_id -> motif string. Uses defaults if None.
        motif_prob: Probability of planting each motif in a sequence.
        base_efficiency: Baseline TE for sequences without the target motif.
        motif_bonus: Additional TE when the target motif is present.
        noise_std: Standard deviation of Gaussian noise on efficiency.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_sequences: int = 5000,
        seq_lengths: tuple[int, int] = (200, 2000),
        max_seq_len: int = 2048,
        n_cell_types: int = 5,
        motifs: Optional[dict[int, str]] = None,
        motif_prob: float = 0.3,
        base_efficiency: float = 0.3,
        motif_bonus: float = 0.5,
        noise_std: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_cell_types = n_cell_types
        self.motifs = motifs or {k: v for k, v in DEFAULT_MOTIFS.items() if k < n_cell_types}

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        self.sequences: list[str] = []
        self.efficiencies: list[np.ndarray] = []  # (n_cell_types,) per sequence
        self.planted: list[set[int]] = []  # which motifs were planted

        for _ in range(n_sequences):
            length = rng.randint(*seq_lengths)
            seq = random_mrna(length)

            planted_cell_types: set[int] = set()
            for ct, motif in self.motifs.items():
                if rng.random() < motif_prob and len(motif) <= length:
                    seq = plant_motif(seq, motif, position=rng.randint(0, length - len(motif)))
                    planted_cell_types.add(ct)

            # Compute efficiency per cell type
            eff = np.full(n_cell_types, base_efficiency, dtype=np.float32)
            for ct in planted_cell_types:
                eff[ct] += motif_bonus
            eff += np_rng.normal(0, noise_std, size=n_cell_types).astype(np.float32)
            eff = np.clip(eff, 0.0, 1.0)

            self.sequences.append(seq)
            self.efficiencies.append(eff)
            self.planted.append(planted_cell_types)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        x = one_hot_encode_ribonn(seq, self.max_seq_len)
        eff = torch.from_numpy(self.efficiencies[idx])
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:len(seq)] = True

        return {
            "one_hot": x,            # (4, max_seq_len)
            "mask": mask,            # (max_seq_len,)
            "efficiency": eff,       # (n_cell_types,)
            "seq_len": len(seq),
        }

    def get_sequence(self, idx: int) -> str:
        return self.sequences[idx]
