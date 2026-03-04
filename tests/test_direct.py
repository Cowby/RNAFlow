"""Tests for the direct codon optimizer."""

import torch
from rnaflow.optim.direct import DirectOptimizer, DirectResult


class _MockWrapper:
    """Minimal wrapper that mimics RiboNNWrapper for testing."""

    def __init__(self, input_channels=4, num_targets=3, latent_dim=8):
        self._input_channels = input_channels
        self._num_targets = num_targets
        self._latent_dim = latent_dim
        # Simple linear layers for gradient flow
        self._encoder = torch.nn.Linear(100, latent_dim)
        self._predictor = torch.nn.Linear(100, num_targets)

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def num_targets(self):
        return self._num_targets

    def predict_with_grad(self, x):
        # x: (B, C, L) → pool to fixed size → predict
        pooled = torch.nn.functional.adaptive_avg_pool1d(x[:, :4], 100)
        return self._predictor(pooled.squeeze(0).mean(dim=0, keepdim=True).expand(x.shape[0], -1))

    def encode(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool1d(x[:, :4], 100)
        return self._encoder(pooled.squeeze(0).mean(dim=0, keepdim=True).expand(x.shape[0], -1))


def test_direct_basic():
    """DirectOptimizer runs and returns valid result."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=6,
        cds_size=18,
        utr3_size=6,
        cds_seq="AUGGCUAGCUAGCUAUAG",  # 6 codons
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=20,
        lr=0.05,
    )
    result = opt.optimize(verbose=False)

    assert isinstance(result, DirectResult)
    assert result.best_z is not None
    assert result.best_z.shape == (8,)
    assert len(result.history) == 20
    assert len(result.sequence) == 30  # 6 + 18 + 6
    assert result.best_score > -float("inf")


def test_direct_protein_preserved():
    """CDS should only use synonymous codons — protein is preserved."""
    from rnaflow.data.codon_table import CODON_TO_AA

    cds = "AUGGCUAGCUAGCUAUAG"  # M-A-S-*-L-*
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=0,
        cds_size=len(cds),
        utr3_size=0,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=30,
        lr=0.1,
    )
    result = opt.optimize(verbose=False)

    # Translate original and optimized CDS
    orig_protein = "".join(
        CODON_TO_AA.get(cds[i:i+3], "?") for i in range(0, len(cds), 3)
    )
    opt_cds = result.sequence[:len(cds)]
    opt_protein = "".join(
        CODON_TO_AA.get(opt_cds[i:i+3], "?") for i in range(0, len(opt_cds), 3)
    )
    assert orig_protein == opt_protein, (
        f"Protein changed: {orig_protein} -> {opt_protein}"
    )


def test_direct_result_interface():
    """DirectResult has the required fields."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=50,
        utr5_size=3,
        cds_size=9,
        utr3_size=3,
        cds_seq="AUGGCUUAG",
        target_col=0,
        off_target_cols=[1],
        n_steps=10,
    )
    result = opt.optimize(verbose=False)

    assert hasattr(result, "best_z")
    assert hasattr(result, "best_score")
    assert hasattr(result, "history")
    assert hasattr(result, "sequence")
    assert hasattr(result, "logits")


def test_direct_sequence_length():
    """Output sequence should be exactly bio_len (utr5 + cds + utr3)."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=200,
        utr5_size=12,
        cds_size=18,
        utr3_size=12,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=10,
    )
    result = opt.optimize(verbose=False)
    assert len(result.sequence) == 42  # 12 + 18 + 12


def test_direct_no_utr():
    """Works with CDS only (no UTR regions)."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=0,
        cds_size=18,
        utr3_size=0,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        target_col=0,
        off_target_cols=[1],
        n_steps=10,
    )
    result = opt.optimize(verbose=False)
    assert len(result.sequence) == 18


def test_direct_ratio_mode():
    """Ratio objective mode runs without error."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=6,
        cds_size=18,
        utr3_size=6,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        target_col=0,
        off_target_cols=[1, 2],
        obj_mode="ratio",
        n_steps=20,
        lr=0.05,
    )
    result = opt.optimize(verbose=False)

    assert isinstance(result, DirectResult)
    assert len(result.history) == 20
    assert result.best_score > -float("inf")


def test_direct_utr_seed_initialization():
    """UTR logits should be biased toward seed UTR nucleotides."""
    from rnaflow.data.codon_table import NUC_IDX

    utr5 = "ACGUAC"
    utr3 = "GUACGU"
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=6,
        cds_size=18,
        utr3_size=6,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        utr5_seq=utr5,
        utr3_seq=utr3,
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=5,
    )
    logits = opt._init_utr_logits()

    # Check 5'UTR: position 0 should have A (idx 0) as highest logit
    assert logits[NUC_IDX["A"], 0].item() == 3.0
    assert logits[NUC_IDX["C"], 1].item() == 3.0
    assert logits[NUC_IDX["G"], 2].item() == 3.0
    assert logits[NUC_IDX["U"], 3].item() == 3.0

    # Check 3'UTR: offset by utr5_size=6
    assert logits[NUC_IDX["G"], 6].item() == 3.0
    assert logits[NUC_IDX["U"], 7].item() == 3.0


def test_direct_utr_seed_fallback():
    """Without UTR seed sequences, logits should be small random values."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=6,
        cds_size=18,
        utr3_size=6,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=5,
    )
    logits = opt._init_utr_logits()

    # No position should have logit as high as 3.0 (seed bias)
    assert logits.abs().max().item() < 3.0


def test_direct_n_repeats():
    """n_repeats runs multiple times and returns the best result."""
    opt = DirectOptimizer(
        wrapper=_MockWrapper(),
        seq_len=100,
        utr5_size=6,
        cds_size=18,
        utr3_size=6,
        cds_seq="AUGGCUAGCUAGCUAUAG",
        utr5_seq="ACGUAC",
        utr3_seq="GUACGU",
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=10,
        n_repeats=3,
    )
    result = opt.optimize(verbose=False)

    assert isinstance(result, DirectResult)
    # History should be from the best single run (10 steps, not 30)
    assert len(result.history) == 10
    assert len(result.sequence) == 30  # 6 + 18 + 6
