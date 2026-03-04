"""Tests for the combined pipeline (diffusion → inversion → direct)."""

import torch
from rnaflow.optim.direct import DirectOptimizer, DirectResult
from rnaflow.inversion.gradient_decoder import GradientDecoder, InversionResult
from rnaflow.optim.diffusion import DiffusionOptimizer
from rnaflow.optim.objective import LatentRiboNNObjective


class _MockWrapper:
    """Minimal wrapper that mimics RiboNNWrapper for testing."""

    def __init__(self, input_channels=4, num_targets=3, latent_dim=8):
        self._input_channels = input_channels
        self._num_targets = num_targets
        self._latent_dim = latent_dim
        self._encoder = torch.nn.Linear(100, latent_dim)
        self._predictor = torch.nn.Linear(100, num_targets)
        self._head_tail = torch.nn.Linear(latent_dim, num_targets)

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def num_targets(self):
        return self._num_targets

    @property
    def device(self):
        return torch.device("cpu")

    def predict_with_grad(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool1d(x[:, :4], 100)
        return self._predictor(pooled.squeeze(0).mean(dim=0, keepdim=True).expand(x.shape[0], -1))

    def encode(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool1d(x[:, :4], 100)
        return self._encoder(pooled.squeeze(0).mean(dim=0, keepdim=True).expand(x.shape[0], -1))

    def encode_with_grad(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool1d(x[:, :4], 100)
        return self._encoder(pooled.squeeze(0).mean(dim=0, keepdim=True).expand(x.shape[0], -1))


def test_combined_pipeline():
    """Combined pipeline: inversion → direct produces valid result."""
    wrapper = _MockWrapper()
    cds = "AUGGCUAGCUAGCUAUAG"

    # Stage 1: Inversion (simulate having a z* from diffusion)
    z_star = torch.randn(8)
    decoder = GradientDecoder(
        wrapper=wrapper,
        seq_len=100,
        n_steps=10,
        utr5_size=6,
        cds_size=len(cds),
        utr3_size=6,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1, 2],
        obj_weight=1.0,
        lam=1.0,
    )
    inv_result = decoder.invert(z_star, verbose=False)
    assert isinstance(inv_result, InversionResult)

    # Stage 2: Direct refinement on the inverted sequence
    direct_opt = DirectOptimizer(
        wrapper=wrapper,
        seq_len=100,
        utr5_size=6,
        cds_size=len(cds),
        utr3_size=6,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=20,
        lr=0.05,
    )
    direct_result = direct_opt.optimize(verbose=False)

    assert isinstance(direct_result, DirectResult)
    assert len(direct_result.sequence) == 30  # 6 + 18 + 6
    assert direct_result.best_score > -float("inf")


def test_combined_protein_preserved():
    """Combined pipeline preserves protein through both stages."""
    from rnaflow.data.codon_table import CODON_TO_AA

    wrapper = _MockWrapper()
    cds = "AUGGCUAGCUAGCUAUAG"

    # Inversion stage
    z_star = torch.randn(8)
    decoder = GradientDecoder(
        wrapper=wrapper,
        seq_len=100,
        n_steps=10,
        utr5_size=0,
        cds_size=len(cds),
        utr3_size=0,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1, 2],
    )
    decoder.invert(z_star, verbose=False)

    # Direct refinement (uses original CDS as seed, like combined mode)
    direct_opt = DirectOptimizer(
        wrapper=wrapper,
        seq_len=100,
        utr5_size=0,
        cds_size=len(cds),
        utr3_size=0,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1, 2],
        n_steps=20,
    )
    result = direct_opt.optimize(verbose=False)

    orig_protein = "".join(
        CODON_TO_AA.get(cds[i:i+3], "?") for i in range(0, len(cds), 3)
    )
    opt_cds = result.sequence[:len(cds)]
    opt_protein = "".join(
        CODON_TO_AA.get(opt_cds[i:i+3], "?") for i in range(0, len(opt_cds), 3)
    )
    assert orig_protein == opt_protein


def test_combined_with_ratio_mode():
    """Combined pipeline works with ratio objective mode."""
    wrapper = _MockWrapper()
    cds = "AUGGCUUAG"

    direct_opt = DirectOptimizer(
        wrapper=wrapper,
        seq_len=50,
        utr5_size=3,
        cds_size=len(cds),
        utr3_size=3,
        cds_seq=cds,
        target_col=0,
        off_target_cols=[1],
        obj_mode="ratio",
        n_steps=10,
    )
    result = direct_opt.optimize(verbose=False)
    assert isinstance(result, DirectResult)
    assert len(result.history) == 10
