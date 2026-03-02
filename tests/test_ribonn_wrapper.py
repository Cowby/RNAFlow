"""Tests for the RiboNN wrapper using MockRiboNN."""

import torch
from rnaflow.embeddings.ribonn_wrapper import MockRiboNN, RiboNNWrapper
from rnaflow.data.encoding import one_hot_encode_ribonn


def _make_wrapper(seq_len=512, num_targets=5):
    model = MockRiboNN(seq_len=seq_len, num_targets=num_targets)
    return RiboNNWrapper.from_model(model, device="cpu")


def test_wrapper_encode():
    wrapper = _make_wrapper()
    x = torch.randn(4, 4, 512)
    z = wrapper.encode(x)
    assert z.shape == (4, 64)


def test_wrapper_predict():
    wrapper = _make_wrapper()
    x = torch.randn(2, 4, 512)
    te = wrapper.predict(x)
    assert te.shape == (2, 5)


def test_wrapper_encode_and_predict():
    wrapper = _make_wrapper()
    x = torch.randn(3, 4, 512)
    z, te = wrapper.encode_and_predict(x)
    assert z.shape == (3, 64)
    assert te.shape == (3, 5)


def test_wrapper_encode_sequence():
    wrapper = _make_wrapper(seq_len=256)
    seq = "AUGCAUGCAUGCAUGC" * 16  # 256 nt
    z = wrapper.encode_sequence(seq, max_len=256)
    assert z.shape == (64,)


def test_wrapper_latent_dim():
    wrapper = _make_wrapper()
    assert wrapper.latent_dim == 64


def test_wrapper_num_targets():
    wrapper = _make_wrapper(num_targets=3)
    assert wrapper.num_targets == 3


def test_wrapper_encode_with_grad():
    wrapper = _make_wrapper()
    x = torch.randn(2, 4, 512, requires_grad=True)
    z = wrapper.encode_with_grad(x)
    assert z.shape == (2, 64)
    z.sum().backward()
    assert x.grad is not None


def test_wrapper_deterministic():
    wrapper = _make_wrapper()
    x = torch.randn(2, 4, 512)
    z1 = wrapper.encode(x)
    z2 = wrapper.encode(x)
    assert torch.allclose(z1, z2)


def test_wrapper_cleanup():
    wrapper = _make_wrapper()
    wrapper.cleanup()
    assert wrapper._hook_handle is None
