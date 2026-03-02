"""Tests for the TranslationPredictor model."""

import tempfile
from pathlib import Path

import torch
from rnaflow.models.predictor import TranslationPredictor


def test_predictor_forward():
    model = TranslationPredictor(latent_dim=64, n_cell_types=5)
    z = torch.randn(8, 64)
    ct = torch.randint(0, 5, (8,))
    out = model(z, ct)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


def test_predictor_different_cell_types():
    model = TranslationPredictor(latent_dim=32, n_cell_types=3)
    z = torch.randn(1, 32)
    # Same sequence, different cell types -> should give different predictions
    preds = []
    for ct in range(3):
        pred = model(z, torch.tensor([ct]))
        preds.append(pred.item())
    # Not all identical (extremely unlikely with random weights)
    assert len(set(preds)) > 1


def test_predictor_gradient_flow():
    model = TranslationPredictor(latent_dim=64, n_cell_types=5)
    z = torch.randn(4, 64, requires_grad=True)
    ct = torch.randint(0, 5, (4,))
    out = model(z, ct)
    out.sum().backward()
    assert z.grad is not None
    assert z.grad.shape == (4, 64)


def test_predictor_save_load():
    model = TranslationPredictor(latent_dim=32, n_cell_types=3, cell_embed_dim=8)
    model.eval()  # disable dropout for deterministic comparison
    z = torch.randn(2, 32)
    ct = torch.tensor([0, 1])
    original_pred = model(z, ct).detach()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "predictor.pt"
        model.save(path)
        loaded = TranslationPredictor.load(path)
        loaded.eval()
        loaded_pred = loaded(z, ct).detach()

    assert torch.allclose(original_pred, loaded_pred)
