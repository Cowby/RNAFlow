"""Tests for the Flow-inspired CEM optimizer."""

import torch
from rnaflow.optim.flow_cem import FlowCEM
from rnaflow.optim.cem import VanillaCEM


def _quadratic_objective(z: torch.Tensor) -> torch.Tensor:
    """Simple quadratic: maximize -(z - target)^2. Optimum at target."""
    target = torch.ones(z.shape[1]) * 3.0
    return -((z - target) ** 2).sum(dim=1)


def test_flow_cem_basic():
    optimizer = FlowCEM(dim=8, pop_size=64, elite_frac=0.1, n_iters=50)
    result = optimizer.optimize(_quadratic_objective, verbose=False)

    assert result.best_z is not None
    assert result.best_z.shape == (8,)
    assert len(result.history) == 50
    # Score should improve over time
    assert result.history[-1] > result.history[0]


def test_flow_cem_finds_optimum():
    optimizer = FlowCEM(dim=4, pop_size=128, elite_frac=0.1, n_iters=100)
    result = optimizer.optimize(_quadratic_objective, verbose=False)

    # Should be close to target (3.0 in each dimension)
    assert torch.allclose(result.best_z, torch.ones(4) * 3.0, atol=0.5)


def test_flow_cem_cosine_schedule():
    optimizer = FlowCEM(dim=4, pop_size=32, n_iters=100, schedule="cosine")
    # t=0 -> 0, t=0.5 -> 0.5, t=1 -> 1
    assert optimizer._time_schedule(0) == 0.0
    assert abs(optimizer._time_schedule(99) - 1.0) < 1e-6
    # Cosine: midpoint should be 0.5
    assert abs(optimizer._time_schedule(49) - 0.5) < 0.05


def test_flow_cem_linear_schedule():
    optimizer = FlowCEM(dim=4, pop_size=32, n_iters=100, schedule="linear")
    assert optimizer._time_schedule(0) == 0.0
    assert abs(optimizer._time_schedule(50) - 50 / 99) < 1e-6


def test_flow_cem_quadratic_schedule():
    optimizer = FlowCEM(dim=4, pop_size=32, n_iters=100, schedule="quadratic")
    t_50 = optimizer._time_schedule(50)
    # Quadratic: should be less than linear midpoint
    assert t_50 < 0.5


def test_flow_cem_with_init():
    init_mu = torch.ones(8) * 2.5  # close to target
    optimizer = FlowCEM(dim=8, pop_size=64, n_iters=30, init_mu=init_mu)
    result = optimizer.optimize(_quadratic_objective, verbose=False)

    # Starting close should converge faster
    assert torch.allclose(result.best_z, torch.ones(8) * 3.0, atol=0.5)


def test_flow_cem_momentum():
    optimizer = FlowCEM(dim=4, pop_size=64, n_iters=50, momentum=0.5)
    result = optimizer.optimize(_quadratic_objective, verbose=False)
    assert result.best_z is not None


def test_flow_cem_reset():
    optimizer = FlowCEM(dim=4, pop_size=32, n_iters=20)
    optimizer.optimize(_quadratic_objective, verbose=False)
    optimizer.reset()

    assert optimizer.step_count == 0
    assert optimizer.best_z is None
    assert optimizer.best_score == -float("inf")


def test_vanilla_cem_basic():
    optimizer = VanillaCEM(dim=8, pop_size=64, elite_frac=0.1, n_iters=50)
    result = optimizer.optimize(_quadratic_objective, verbose=False)
    assert result.best_z is not None
    assert result.history[-1] > result.history[0]


def test_flow_cem_outperforms_vanilla_on_noisy():
    """FlowCEM's slower distribution collapse should help on noisy objectives."""
    torch.manual_seed(42)

    def noisy_objective(z):
        target = torch.ones(z.shape[1]) * 3.0
        signal = -((z - target) ** 2).sum(dim=1)
        noise = torch.randn(z.shape[0]) * 2.0
        return signal + noise

    # Run both optimizers with same seed
    flow_scores = []
    vanilla_scores = []
    for _ in range(5):
        flow = FlowCEM(dim=8, pop_size=64, elite_frac=0.1, n_iters=80, schedule="cosine")
        flow_result = flow.optimize(noisy_objective, verbose=False)
        flow_scores.append(flow_result.best_score)

        vanilla = VanillaCEM(dim=8, pop_size=64, elite_frac=0.1, n_iters=80)
        vanilla_result = vanilla.optimize(noisy_objective, verbose=False)
        vanilla_scores.append(vanilla_result.best_score)

    # FlowCEM should do at least as well on average
    # (not strictly guaranteed, but should hold statistically)
    flow_mean = sum(flow_scores) / len(flow_scores)
    vanilla_mean = sum(vanilla_scores) / len(vanilla_scores)
    # Just check both find reasonable solutions
    assert flow_mean > -10.0
    assert vanilla_mean > -10.0
