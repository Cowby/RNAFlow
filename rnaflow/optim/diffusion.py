"""Latent Diffusion Optimizer with classifier guidance.

Uses DDPM-style reverse diffusion in the 64-dim latent space, guided by
gradients from the differentiable objective function (LatentRiboNNObjective
or PredictorSpecificityObjective).

Key difference from FlowCEM: this optimizer uses gradient information from
the objective to steer sampling, rather than gradient-free elite selection.
The analytical Gaussian prior (centered on the seed embedding) provides
regularization, while classifier guidance navigates toward high-TE regions.

Algorithm:
    1. Precompute cosine/linear noise schedule (betas, alpha_bars)
    2. Sample z_T ~ N(init_mu, I) or N(0, I)
    3. Reverse diffusion with classifier guidance:
       - Analytical noise estimate from Gaussian prior
       - Gradient of objective w.r.t. z_t for guidance
       - DDPM reverse step with guided noise prediction
    4. Return best sample by objective score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class DiffusionResult:
    """Result of a diffusion optimization run."""
    best_z: Tensor
    best_score: float
    history: list[float] = field(default_factory=list)
    noise_history: list[float] = field(default_factory=list)


class DiffusionOptimizer:
    """DDPM-based latent space optimizer with classifier guidance.

    Instead of population-based elite selection (CEM), this optimizer uses
    reverse diffusion guided by the gradient of the objective function.
    The objective must be differentiable (e.g., LatentRiboNNObjective).

    Args:
        dim: Latent space dimensionality.
        batch_size: Number of parallel samples (analogous to pop_size in CEM).
        n_steps: Number of diffusion timesteps T.
        guidance_scale: Classifier guidance weight. Higher = stronger gradient signal.
        noise_schedule: Beta schedule type ('cosine' or 'linear').
        init_mu: Seed embedding to center the prior. Defaults to zeros.
        init_sigma: Prior std (unused in diffusion, kept for interface compat).
        n_repeats: Number of independent reverse diffusion runs (keep best).
        clip_grad_norm: Maximum gradient norm per sample for stability.
        proximity_weight: Penalizes ||z - mu_prior||² / dim during guidance.
            Prevents drift into unrealistic latent regions. 0 = disabled.
        max_radius: Hard clamp on distance from mu_prior. Samples beyond this
            are projected back. 0 = disabled.
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        batch_size: int = 512,
        n_steps: int = 200,
        guidance_scale: float = 10.0,
        noise_schedule: str = "cosine",
        init_mu: Tensor | None = None,
        init_sigma: Tensor | None = None,
        n_repeats: int = 1,
        clip_grad_norm: float = 1.0,
        proximity_weight: float = 0.1,
        max_radius: float = 50.0,
        device: str = "cpu",
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.guidance_scale = guidance_scale
        self.noise_schedule = noise_schedule
        self.n_repeats = n_repeats
        self.clip_grad_norm = clip_grad_norm
        self.proximity_weight = proximity_weight
        self.max_radius = max_radius
        self.device = torch.device(device)

        self.mu_prior = (
            init_mu.to(self.device) if init_mu is not None
            else torch.zeros(dim, device=self.device)
        )

        # Precompute noise schedule
        if noise_schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            self.betas = self._linear_beta_schedule()

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - self.alpha_bars).sqrt()

        self.best_z: Tensor | None = None
        self.best_score = -float("inf")

    def _cosine_beta_schedule(self) -> Tensor:
        """Cosine schedule from Nichol & Dhariwal (2021)."""
        steps = self.n_steps + 1
        s = 0.008
        t = torch.linspace(0, 1, steps, device=self.device)
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(min=1e-6, max=0.999)

    def _linear_beta_schedule(self) -> Tensor:
        """Linear schedule with standard DDPM range."""
        return torch.linspace(1e-4, 0.02, self.n_steps, device=self.device)

    def _objective_with_grad(self, z: Tensor, objective_fn: callable) -> Tensor:
        """Evaluate the objective with gradients enabled.

        The existing objective classes use @torch.no_grad(). This method
        reaches into the underlying nn.Module to compute scores with
        gradient tracking for classifier guidance.

        Args:
            z: (N, dim) latent vectors requiring grad.
            objective_fn: The objective callable (LatentRiboNN or Predictor).

        Returns:
            scores: (N,) objective scores with grad graph attached.
        """
        # LatentRiboNNObjective — access head_tail(s) directly
        if hasattr(objective_fn, "head_tail"):
            if hasattr(objective_fn, "head_tails") and objective_fn.head_tails:
                te = torch.stack(
                    [ht(z) for ht in objective_fn.head_tails]
                ).mean(dim=0)
            else:
                te = objective_fn.head_tail(z)  # (N, num_targets)
            target_eff = te[:, objective_fn.target_col]
            off_target_eff = te[:, objective_fn.off_target_cols].mean(dim=1)
            return target_eff - objective_fn.lam * off_target_eff

        # PredictorSpecificityObjective — call predictor directly
        if hasattr(objective_fn, "predictor"):
            N = z.shape[0]
            dev = z.device
            target_eff = objective_fn.predictor(
                z, torch.full((N,), objective_fn.target_cell, device=dev, dtype=torch.long)
            )
            off_target_eff = torch.zeros(N, device=dev)
            for ct in objective_fn.off_target_cells:
                off_target_eff += objective_fn.predictor(
                    z, torch.full((N,), ct, device=dev, dtype=torch.long)
                )
            off_target_eff /= len(objective_fn.off_target_cells)
            return target_eff - objective_fn.lam * off_target_eff

        # Fallback: call directly (assumes grad context is enabled)
        return objective_fn(z)

    def _reverse_step(
        self, z_t: Tensor, t: int, objective_fn: callable
    ) -> Tensor:
        """Single DDPM reverse step with classifier guidance.

        Without a trained denoiser, we use an analytical noise estimate
        from the Gaussian prior and add classifier guidance via the
        objective gradient.

        Args:
            z_t: (N, dim) noisy latent vectors at timestep t.
            t: Current timestep (counting down from T-1 to 0).
            objective_fn: Differentiable objective for guidance.

        Returns:
            z_{t-1}: (N, dim) denoised latent vectors.
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].clamp(min=1e-8)

        # Analytical noise estimate from Gaussian prior
        eps_hat = (z_t - sqrt_ab * self.mu_prior.unsqueeze(0)) / sqrt_1_ab

        # Classifier guidance: compute gradient of objective w.r.t. z_t
        # Include proximity penalty to prevent drift from realistic latent region
        z_t_grad = z_t.detach().requires_grad_(True)
        scores = self._objective_with_grad(z_t_grad, objective_fn)
        if self.proximity_weight > 0:
            displacement = z_t_grad - self.mu_prior.unsqueeze(0)
            proximity_penalty = (displacement ** 2).sum(dim=1) / self.dim
            scores = scores - self.proximity_weight * proximity_penalty
        grad = torch.autograd.grad(scores.sum(), z_t_grad)[0]

        # Per-sample gradient clipping for stability
        if self.clip_grad_norm > 0:
            grad_norms = grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
            clip_coef = (self.clip_grad_norm / grad_norms).clamp(max=1.0)
            grad = grad * clip_coef

        # Apply guidance directly to the reverse mean (more stable than
        # modifying the noise estimate, which gets amplified)
        eps_unguided = eps_hat

        # DDPM reverse mean (unguided denoising)
        mean = (1.0 / alpha_t.sqrt()) * (
            z_t - (beta_t / sqrt_1_ab) * eps_unguided
        )

        # Add guidance as a step-size-modulated gradient ascent term
        mean = mean + self.guidance_scale * beta_t * grad

        # Add stochastic noise (except at final step)
        if t > 0:
            sigma_t = beta_t.sqrt()
            noise = torch.randn_like(z_t)
            z_next = mean + sigma_t * noise
        else:
            z_next = mean

        # Hard clamp: project back if beyond max_radius from prior
        if self.max_radius > 0:
            displacement = z_next - self.mu_prior.unsqueeze(0)
            dist = displacement.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (self.max_radius / dist).clamp(max=1.0)
            z_next = self.mu_prior.unsqueeze(0) + displacement * scale

        return z_next.detach()

    def optimize(
        self, objective_fn: callable, verbose: bool = True
    ) -> DiffusionResult:
        """Run full reverse diffusion optimization.

        Args:
            objective_fn: Callable taking (N, dim) tensor, returning (N,) scores.
            verbose: Print progress every 10 steps.

        Returns:
            DiffusionResult with optimization trajectory.
        """
        history: list[float] = []
        noise_history: list[float] = []

        global_best_z: Tensor | None = None
        global_best_score = -float("inf")

        for repeat in range(self.n_repeats):
            # Start from noisy samples around the prior
            z_t = self.mu_prior.unsqueeze(0) + torch.randn(
                self.batch_size, self.dim, device=self.device
            )

            for t in reversed(range(self.n_steps)):
                z_t = self._reverse_step(z_t, t, objective_fn)

                # Evaluate and track best
                with torch.no_grad():
                    scores = objective_fn(z_t)
                best_idx = scores.argmax()
                step_best = scores[best_idx].item()

                if step_best > global_best_score:
                    global_best_score = step_best
                    global_best_z = z_t[best_idx].detach().clone()

                history.append(global_best_score)
                noise_history.append(
                    self.sqrt_one_minus_alpha_bars[t].item()
                )

                if verbose and (self.n_steps - 1 - t) % 10 == 0:
                    mean_dist = (z_t - self.mu_prior.unsqueeze(0)).norm(
                        dim=1
                    ).mean().item()
                    print(
                        f"[Diffusion] Step {self.n_steps - t:4d}/{self.n_steps} "
                        f"(repeat {repeat + 1}/{self.n_repeats}) | "
                        f"score={global_best_score:.4f} | "
                        f"dist={mean_dist:.1f} | "
                        f"noise={self.sqrt_one_minus_alpha_bars[t]:.4f}"
                    )

        return DiffusionResult(
            best_z=global_best_z,
            best_score=global_best_score,
            history=history,
            noise_history=noise_history,
        )

    def reset(self):
        """Reset optimizer state for a new run."""
        self.best_z = None
        self.best_score = -float("inf")
