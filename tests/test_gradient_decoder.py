"""Tests for gradient-based sequence inversion."""

import torch
from rnaflow.embeddings.ribonn_wrapper import MockRiboNN, RiboNNWrapper
from rnaflow.inversion.gradient_decoder import GradientDecoder


def _make_wrapper(seq_len=128):
    model = MockRiboNN(seq_len=seq_len, num_targets=5, n_conv_layers=2)
    return RiboNNWrapper.from_model(model, device="cpu")


def test_gradient_decoder_runs():
    wrapper = _make_wrapper(seq_len=128)
    decoder = GradientDecoder(
        wrapper=wrapper, seq_len=128, n_steps=20, lr=0.1
    )

    z_target = torch.randn(64)
    result = decoder.invert(z_target, verbose=False)

    assert len(result.sequence) == 128
    assert all(c in "AUGC" for c in result.sequence)
    assert len(result.loss_history) == 20
    assert result.final_loss < result.loss_history[0]  # loss should decrease


def test_gradient_decoder_with_mask():
    wrapper = _make_wrapper(seq_len=128)
    decoder = GradientDecoder(
        wrapper=wrapper, seq_len=128, n_steps=20, lr=0.1
    )

    z_target = torch.randn(64)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:50] = True  # only first 50 positions active

    result = decoder.invert(z_target, mask=mask, verbose=False)
    assert len(result.sequence) == 50  # trimmed to active positions


def test_gradient_decoder_loss_decreases():
    wrapper = _make_wrapper(seq_len=64)
    decoder = GradientDecoder(
        wrapper=wrapper, seq_len=64, n_steps=50, lr=0.05
    )

    z_target = torch.randn(64)
    result = decoder.invert(z_target, verbose=False)

    # Loss should generally decrease (allow some noise)
    early_avg = sum(result.loss_history[:5]) / 5
    late_avg = sum(result.loss_history[-5:]) / 5
    assert late_avg < early_avg
