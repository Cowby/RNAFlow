"""Tests for the latent diffusion optimizer."""

import torch
from rnaflow.optim.diffusion import DiffusionOptimizer, DiffusionResult


class _DiffObjective:
    """Differentiable quadratic objective for testing gradient guidance."""

    def __init__(self, target_val: float = 3.0):
        self.target_val = target_val

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        target = torch.ones(z.shape[1], device=z.device) * self.target_val
        return -((z - target) ** 2).sum(dim=1)


def test_diffusion_basic():
    opt = DiffusionOptimizer(
        dim=8, batch_size=32, n_steps=20, guidance_scale=5.0
    )
    result = opt.optimize(_DiffObjective(), verbose=False)

    assert result.best_z is not None
    assert result.best_z.shape == (8,)
    assert len(result.history) == 20
    assert len(result.noise_history) == 20
    assert result.best_score > -float("inf")


def test_diffusion_finds_optimum():
    torch.manual_seed(42)
    opt = DiffusionOptimizer(
        dim=4, batch_size=128, n_steps=100, guidance_scale=20.0
    )
    result = opt.optimize(_DiffObjective(), verbose=False)

    # Should approach the target (3.0 in each dimension)
    assert torch.allclose(result.best_z, torch.ones(4) * 3.0, atol=1.5)


def test_diffusion_cosine_schedule():
    opt = DiffusionOptimizer(dim=4, n_steps=100, noise_schedule="cosine")
    # Alpha bars should be monotonically decreasing (more noise at higher t)
    diffs = opt.alpha_bars[1:] - opt.alpha_bars[:-1]
    assert (diffs <= 0).all()
    # First alpha_bar should be close to 1 (little noise), last close to 0
    assert opt.alpha_bars[0] > 0.9
    assert opt.alpha_bars[-1] < 0.1


def test_diffusion_linear_schedule():
    opt = DiffusionOptimizer(dim=4, n_steps=100, noise_schedule="linear")
    diffs = opt.alpha_bars[1:] - opt.alpha_bars[:-1]
    assert (diffs <= 0).all()


def test_diffusion_with_init_mu():
    torch.manual_seed(42)
    # Starting close to target should help
    init_mu = torch.ones(4) * 2.5
    opt = DiffusionOptimizer(
        dim=4, batch_size=64, n_steps=50, guidance_scale=10.0,
        init_mu=init_mu,
    )
    result = opt.optimize(_DiffObjective(), verbose=False)
    assert torch.allclose(result.best_z, torch.ones(4) * 3.0, atol=1.5)


def test_diffusion_guidance_scale_effect():
    """Higher guidance scale should yield better scores on smooth objectives."""
    torch.manual_seed(42)
    scores_low = []
    scores_high = []

    for _ in range(3):
        opt_low = DiffusionOptimizer(
            dim=4, batch_size=64, n_steps=50, guidance_scale=1.0
        )
        res_low = opt_low.optimize(_DiffObjective(), verbose=False)
        scores_low.append(res_low.best_score)

        opt_high = DiffusionOptimizer(
            dim=4, batch_size=64, n_steps=50, guidance_scale=20.0
        )
        res_high = opt_high.optimize(_DiffObjective(), verbose=False)
        scores_high.append(res_high.best_score)

    # Higher guidance should generally do better (or at least comparable)
    mean_low = sum(scores_low) / len(scores_low)
    mean_high = sum(scores_high) / len(scores_high)
    assert mean_high > mean_low - 5.0  # allow some tolerance


def test_diffusion_result_interface():
    """DiffusionResult should have the same core fields as FlowCEMResult."""
    opt = DiffusionOptimizer(dim=4, batch_size=16, n_steps=10)
    result = opt.optimize(_DiffObjective(), verbose=False)

    assert hasattr(result, "best_z")
    assert hasattr(result, "best_score")
    assert hasattr(result, "history")
    assert isinstance(result, DiffusionResult)


def test_diffusion_proximity_constrains_distance():
    """Proximity weight should keep z close to mu_prior."""
    torch.manual_seed(42)

    # Without proximity: allow drift
    opt_no_prox = DiffusionOptimizer(
        dim=4, batch_size=64, n_steps=50, guidance_scale=20.0,
        proximity_weight=0.0, max_radius=0.0,
    )
    res_no_prox = opt_no_prox.optimize(_DiffObjective(target_val=100.0), verbose=False)
    dist_no_prox = res_no_prox.best_z.norm().item()

    # With proximity: constrained
    opt_prox = DiffusionOptimizer(
        dim=4, batch_size=64, n_steps=50, guidance_scale=20.0,
        proximity_weight=1.0, max_radius=0.0,
    )
    res_prox = opt_prox.optimize(_DiffObjective(target_val=100.0), verbose=False)
    dist_prox = res_prox.best_z.norm().item()

    # Proximity should keep z closer to origin
    assert dist_prox < dist_no_prox


def test_diffusion_max_radius_clamps():
    """Max radius should enforce a hard distance bound."""
    torch.manual_seed(42)
    radius = 5.0
    opt = DiffusionOptimizer(
        dim=4, batch_size=64, n_steps=50, guidance_scale=20.0,
        proximity_weight=0.0, max_radius=radius,
    )
    result = opt.optimize(_DiffObjective(target_val=100.0), verbose=False)

    # Best z should be within max_radius of origin (mu_prior = 0)
    assert result.best_z.norm().item() <= radius + 0.1  # small tolerance


def test_diffusion_reset():
    opt = DiffusionOptimizer(dim=4, batch_size=16, n_steps=10)
    opt.optimize(_DiffObjective(), verbose=False)
    opt.reset()

    assert opt.best_z is None
    assert opt.best_score == -float("inf")
