"""Vanilla Cross-Entropy Method (CEM) optimizer in latent space."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class CEMResult:
    """Result of a CEM optimization run."""
    best_z: Tensor
    best_score: float
    history: list[float] = field(default_factory=list)
    final_mu: Tensor | None = None
    final_sigma: Tensor | None = None


class VanillaCEM:
    """Standard Cross-Entropy Method for latent space optimization.

    Maintains a diagonal Gaussian distribution over the latent space,
    iteratively refitting it to elite samples.

    Args:
        dim: Dimensionality of the latent space.
        pop_size: Number of candidate samples per iteration.
        elite_frac: Fraction of top samples used to update the distribution.
        n_iters: Number of CEM iterations.
        init_mu: Initial mean. Defaults to zeros.
        init_sigma: Initial std. Defaults to ones.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        n_iters: int = 100,
        init_mu: Tensor | None = None,
        init_sigma: Tensor | None = None,
        device: str = "cpu",
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.n_elite = max(1, int(pop_size * elite_frac))
        self.n_iters = n_iters
        self.device = torch.device(device)

        self.mu = (init_mu if init_mu is not None else torch.zeros(dim)).to(self.device)
        self.sigma = (init_sigma if init_sigma is not None else torch.ones(dim)).to(self.device)

        self.best_z: Tensor | None = None
        self.best_score = -float("inf")

    def step(self, objective_fn: callable) -> tuple[Tensor, float]:
        """Run one CEM iteration.

        Args:
            objective_fn: Callable taking (N, dim) tensor, returning (N,) scores.

        Returns:
            (best_z, best_score) from this iteration.
        """
        # Sample candidates
        eps = torch.randn(self.pop_size, self.dim, device=self.device)
        samples = self.mu.unsqueeze(0) + eps * self.sigma.unsqueeze(0)

        # Evaluate
        scores = objective_fn(samples)

        # Select elites
        elite_idx = scores.topk(self.n_elite).indices
        elites = samples[elite_idx]

        # Update distribution
        self.mu = elites.mean(dim=0)
        self.sigma = elites.std(dim=0).clamp(min=1e-6)

        # Track best
        best_idx = scores.argmax()
        if scores[best_idx].item() > self.best_score:
            self.best_score = scores[best_idx].item()
            self.best_z = samples[best_idx].clone()

        return self.best_z, self.best_score

    def optimize(self, objective_fn: callable, verbose: bool = True) -> CEMResult:
        """Run full CEM optimization.

        Args:
            objective_fn: Callable taking (N, dim) tensor, returning (N,) scores.
            verbose: Print progress every 10 iterations.

        Returns:
            CEMResult with best latent vector, score, and history.
        """
        history = []
        for k in range(self.n_iters):
            z, score = self.step(objective_fn)
            history.append(score)

            if verbose and k % 10 == 0:
                print(
                    f"[VanillaCEM] Iter {k:4d} | score={score:.4f} | "
                    f"sigma_mean={self.sigma.mean():.4f}"
                )

        return CEMResult(
            best_z=self.best_z,
            best_score=self.best_score,
            history=history,
            final_mu=self.mu.clone(),
            final_sigma=self.sigma.clone(),
        )
