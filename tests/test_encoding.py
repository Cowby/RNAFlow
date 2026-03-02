"""Tests for mRNA sequence encoding/decoding."""

import torch
from rnaflow.data.encoding import (
    one_hot_encode,
    one_hot_encode_ribonn,
    decode_logits,
    soft_to_one_hot,
    sequence_entropy,
)


def test_one_hot_encode_basic():
    seq = "AUGC"
    oh, mask = one_hot_encode(seq)
    assert oh.shape == (4, 4)
    assert mask.all()
    # A at pos 0
    assert oh[0, 0] == 1.0
    # U at pos 1
    assert oh[1, 1] == 1.0
    # G at pos 2
    assert oh[3, 2] == 1.0
    # C at pos 3
    assert oh[2, 3] == 1.0


def test_one_hot_encode_with_padding():
    seq = "AUG"
    oh, mask = one_hot_encode(seq, max_len=10)
    assert oh.shape == (4, 10)
    assert mask[:3].all()
    assert not mask[3:].any()
    # Padded positions should be all zeros
    assert oh[:, 3:].sum() == 0


def test_one_hot_encode_truncation():
    seq = "AUGCAUGCAUGC"
    oh, mask = one_hot_encode(seq, max_len=4)
    assert oh.shape == (4, 4)
    assert mask.all()


def test_one_hot_encode_ribonn_format():
    seq = "AUGC"
    x = one_hot_encode_ribonn(seq, max_len=8)
    assert x.shape == (4, 8)
    # RiboNN uses T instead of U: A=0, T=1, C=2, G=3
    assert x[0, 0] == 1.0  # A
    assert x[1, 1] == 1.0  # U->T
    assert x[3, 2] == 1.0  # G
    assert x[2, 3] == 1.0  # C


def test_decode_logits_roundtrip():
    seq = "AUGCAUGC"
    oh, _ = one_hot_encode(seq)
    recovered = decode_logits(oh)
    assert recovered == seq


def test_soft_to_one_hot_shape():
    logits = torch.randn(4, 100)
    soft = soft_to_one_hot(logits, temperature=1.0)
    assert soft.shape == (4, 100)
    # Should sum to ~1 per position
    sums = soft.sum(dim=0)
    assert torch.allclose(sums, torch.ones(100), atol=0.01)


def test_sequence_entropy():
    # Uniform distribution -> max entropy
    uniform = torch.zeros(4, 10)
    high_ent = sequence_entropy(uniform)

    # One-hot (very peaked) -> low entropy
    peaked = torch.zeros(4, 10)
    peaked[0, :] = 100.0  # strongly favor A
    low_ent = sequence_entropy(peaked)

    assert high_ent > low_ent
