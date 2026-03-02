"""Flow-inspired Cross-Entropy Method (FlowCEM) optimizer.

The key innovation: instead of letting the CEM distribution collapse freely,
we interpolate between the initial (broad) prior distribution and the
elite-fitted distribution using a time parameter t in [0, 1].

This creates a smooth "flow" from exploration to exploitation, analogous to
the probability path in flow matching / continuous normalizing flows:

    mu_t  = (1 - t) * mu_0  + t * mu_elite
    sig_t = (1 - t) * sig_0 + t * sig_elite

With a cosine schedule for t, we get slow transition at the start (exploration)
and end (fine-tuning), with faster transition in the middle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class FlowCEMResult:
    """Result of a FlowCEM optimization run."""
    best_z: Tensor
    best_score: float
    history: list[float] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)
    sigma_history: list[float] = field(default_factory=list)
    final_mu: Tensor | None = None
    final_sigma: Tensor | None = None


class FlowCEM:
    """Cross-Entropy Method with flow-matching-inspired distribution interpolation.

    At each step k:
        t_k = schedule(k / (n_iters - 1))
        mu_t  = (1 - t_k) * mu_0  + t_k * mu_elite
        sig_t = (1 - t_k) * sig_0 + t_k * sig_elite
        samples ~ N(mu_t, diag(sig_t^2))

    This prevents premature collapse and provides smoother optimization.

    Args:
        dim: Latent space dimensionality.
        pop_size: Number of candidate samples per iteration.
        elite_frac: Fraction of top samples to keep.
        n_iters: Number of optimization iterations.
        init_mu: Initial mean of the prior distribution. Defaults to zeros.
        init_sigma: Initial std of the prior distribution. Defaults to ones.
        schedule: Time interpolation schedule ('linear', 'cosine', 'quadratic').
        min_sigma: Minimum standard deviation to prevent collapse.
        momentum: Exponential moving average factor for elite updates.
            0 = no momentum (standard CEM), 0.9 = heavy smoothing.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        pop_size: int = 512,
        elite_frac: float = 0.05,
        n_iters: int = 200,
        init_mu: Tensor | None = None,
        init_sigma: Tensor | None = None,
        schedule: str = "cosine",
        min_sigma: float = 1e-4,
        momentum: float = 0.0,
        device: str = "cpu",
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.n_elite = max(1, int(pop_size * elite_frac))
        self.n_iters = n_iters
        self.schedule = schedule
        self.min_sigma = min_sigma
        self.momentum = momentum
        self.device = torch.device(device)

        # Prior distribution (preserved throughout optimization)
        self.mu_0 = (init_mu if init_mu is not None else torch.zeros(dim)).to(self.device)
        self.sigma_0 = (init_sigma if init_sigma is not None else torch.ones(dim)).to(self.device)

        # Elite-fitted distribution (updated each iteration)
        self.mu_elite = self.mu_0.clone()
        self.sigma_elite = self.sigma_0.clone()

        self.best_z: Tensor | None = None
        self.best_score = -float("inf")
        self.step_count = 0

    def _time_schedule(self, k: int) -> float:
        """Map iteration index k to interpolation parameter t in [0, 1].

        Args:
            k: Current iteration (0-indexed).

        Returns:
            t: Interpolation weight. t=0 means pure prior, t=1 means pure elite.
        """
        t_raw = k / max(self.n_iters - 1, 1)

        if self.schedule == "linear":
            return t_raw
        elif self.schedule == "cosine":
            # Slow at both ends, fast in the middle — like cosine annealing
            return 0.5 * (1.0 - math.cos(math.pi * t_raw))
        elif self.schedule == "quadratic":
            # Slow start, faster later — more exploration early
            return t_raw ** 2
        elif self.schedule == "sqrt":
            # Fast start, slower later — more exploitation early
            return math.sqrt(t_raw)
        else:
            return t_raw

    def step(self, objective_fn: callable) -> tuple[Tensor, float]:
        """Run one FlowCEM iteration.

        Args:
            objective_fn: Callable taking (N, dim) tensor, returning (N,) scores.

        Returns:
            (best_z, best_score) seen so far.
        """
        t = self._time_schedule(self.step_count)

        # Interpolate between prior and elite-fitted distribution
        mu_t = (1.0 - t) * self.mu_0 + t * self.mu_elite
        sigma_t = (1.0 - t) * self.sigma_0 + t * self.sigma_elite

        # Sample candidates from the interpolated distribution
        eps = torch.randn(self.pop_size, self.dim, device=self.device)
        samples = mu_t.unsqueeze(0) + eps * sigma_t.unsqueeze(0)

        # Evaluate objective
        scores = objective_fn(samples)

        # Select elites
        elite_idx = scores.topk(self.n_elite).indices
        elites = samples[elite_idx]

        # Fit new Gaussian to elites (with optional momentum)
        new_mu = elites.mean(dim=0)
        new_sigma = elites.std(dim=0).clamp(min=self.min_sigma)

        if self.momentum > 0 and self.step_count > 0:
            self.mu_elite = self.momentum * self.mu_elite + (1.0 - self.momentum) * new_mu
            self.sigma_elite = self.momentum * self.sigma_elite + (1.0 - self.momentum) * new_sigma
        else:
            self.mu_elite = new_mu
            self.sigma_elite = new_sigma

        # Track global best
        best_idx = scores.argmax()
        if scores[best_idx].item() > self.best_score:
            self.best_score = scores[best_idx].item()
            self.best_z = samples[best_idx].clone()

        self.step_count += 1
        return self.best_z, self.best_score

    def optimize(self, objective_fn: callable, verbose: bool = True) -> FlowCEMResult:
        """Run full FlowCEM optimization.

        Args:
            objective_fn: Callable taking (N, dim) tensor, returning (N,) scores.
            verbose: Print progress every 10 iterations.

        Returns:
            FlowCEMResult with optimization trajectory.
        """
        history = []
        time_history = []
        sigma_history = []

        for k in range(self.n_iters):
            t = self._time_schedule(k)
            z, score = self.step(objective_fn)

            history.append(score)
            time_history.append(t)
            sigma_history.append(self.sigma_elite.mean().item())

            if verbose and k % 10 == 0:
                print(
                    f"[FlowCEM] Iter {k:4d} | t={t:.3f} | score={score:.4f} | "
                    f"sigma_mean={self.sigma_elite.mean():.4f}"
                )

        return FlowCEMResult(
            best_z=self.best_z,
            best_score=self.best_score,
            history=history,
            time_history=time_history,
            sigma_history=sigma_history,
            final_mu=self.mu_elite.clone(),
            final_sigma=self.sigma_elite.clone(),
        )

    def reset(self):
        """Reset optimizer state for a new run."""
        self.mu_elite = self.mu_0.clone()
        self.sigma_elite = self.sigma_0.clone()
        self.best_z = None
        self.best_score = -float("inf")
        self.step_count = 0
