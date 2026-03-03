"""Tests for the ensemble RiboNN wrapper."""

import torch
import torch.nn as nn

from rnaflow.embeddings.ribonn_wrapper import RiboNNWrapper, MockRiboNN
from rnaflow.embeddings.ensemble import EnsembleRiboNNWrapper
from rnaflow.optim.objective import LatentRiboNNObjective


def _make_mock_wrapper(seq_len=128, num_targets=5, device="cpu"):
    """Create a MockRiboNN-based wrapper for testing."""
    mock = MockRiboNN(seq_len=seq_len, num_targets=num_targets)
    return RiboNNWrapper.from_model(mock, device=device)


def test_ensemble_creation():
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    assert len(ens.wrappers) == 2
    assert len(ens.head_tails) == 2
    assert ens.primary is w1


def test_ensemble_properties():
    w1 = _make_mock_wrapper(num_targets=5)
    w2 = _make_mock_wrapper(num_targets=5)
    ens = EnsembleRiboNNWrapper([w1, w2])

    assert ens.latent_dim == w1.latent_dim
    assert ens.num_targets == w1.num_targets
    assert ens.max_seq_len == w1.max_seq_len


def test_ensemble_averages_predictions():
    torch.manual_seed(42)
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    x = torch.randn(2, 4, 128)
    pred_ens = ens.predict(x)
    pred_w1 = w1.predict(x)
    pred_w2 = w2.predict(x)
    expected = (pred_w1 + pred_w2) / 2

    assert torch.allclose(pred_ens, expected, atol=1e-5)


def test_ensemble_averages_embeddings():
    torch.manual_seed(42)
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    x = torch.randn(2, 4, 128)
    z_ens = ens.encode(x)
    z_w1 = w1.encode(x)
    z_w2 = w2.encode(x)
    expected = (z_w1 + z_w2) / 2

    assert torch.allclose(z_ens, expected, atol=1e-5)


def test_ensemble_gradient_ops_delegate_to_primary():
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    x = torch.randn(1, 4, 128, requires_grad=True)
    z_ens = ens.encode_with_grad(x)
    z_w1 = w1.encode_with_grad(x.detach().requires_grad_(True))

    assert torch.allclose(z_ens, z_w1, atol=1e-5)
    assert z_ens.requires_grad


def test_ensemble_head_tails_are_correct():
    """Head tails should match layers 5-7 of each model's head."""
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    z = torch.randn(4, 64)
    for i, w in enumerate([w1, w2]):
        head = w.model.head
        expected = nn.Sequential(head[5], head[6], head[7])(z)
        actual = ens.head_tails[i](z)
        assert torch.allclose(actual, expected, atol=1e-5)


def test_ensemble_with_latent_objective():
    """LatentRiboNNObjective should average head_tails when given an ensemble."""
    torch.manual_seed(42)
    w1 = _make_mock_wrapper()
    w2 = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w1, w2])

    obj = LatentRiboNNObjective(
        wrapper=ens, target_col=0, off_target_cols=[1, 2], lam=1.0
    )

    assert obj.head_tails is not None
    assert len(obj.head_tails) == 2

    z = torch.randn(8, 64)
    scores = obj(z)
    assert scores.shape == (8,)


def test_ensemble_single_model_is_identity():
    """Ensemble of 1 model should match single model exactly."""
    torch.manual_seed(42)
    w = _make_mock_wrapper()
    ens = EnsembleRiboNNWrapper([w])

    x = torch.randn(2, 4, 128)
    assert torch.allclose(ens.predict(x), w.predict(x), atol=1e-5)
    assert torch.allclose(ens.encode(x), w.encode(x), atol=1e-5)


def test_ensemble_rejects_empty():
    try:
        EnsembleRiboNNWrapper([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
