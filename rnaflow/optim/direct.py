"""Direct codon optimizer: gradient descent on the full CNN prediction.

Skips latent space optimization entirely. Starts from the seed sequence's
codons and directly optimizes cell-type specificity through predict_with_grad.

Complementary to diffusion/FlowCEM:
- Diffusion/FlowCEM: global latent exploration (structural motifs, holistic patterns)
- Direct: local gradient-based codon optimization (targeted TE improvement)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rnaflow.data.encoding import decode_logits, sequence_entropy
from rnaflow.data.codon_table import CODON_TO_AA, AA_TO_CODONS, NUC_IDX
from rnaflow.inversion.gradient_decoder import (
    _build_synonymous_tables,
    _codon_logits_to_soft_seq,
)
from rnaflow.optim.objective import _compute_specificity


@dataclass
class DirectCandidate:
    """A single candidate from direct optimization."""
    sequence: str
    score: float
    z: Tensor
    logits: Tensor | None = None


@dataclass
class DirectResult:
    """Result of direct codon optimization."""
    best_z: Tensor
    best_score: float
    history: list[float] = field(default_factory=list)
    sequence: str = ""
    logits: Tensor | None = None
    candidates: list[DirectCandidate] = field(default_factory=list)


class DirectOptimizer:
    """Optimize codon choices directly through the full CNN.

    Bypasses latent space — uses predict_with_grad to compute TE predictions
    and backpropagates through the full model to update codon logits.

    Args:
        wrapper: RiboNNWrapper or EnsembleRiboNNWrapper.
        seq_len: Model's max_seq_len (for zero-padding).
        utr5_size: Length of 5'UTR region.
        cds_size: Length of CDS region.
        utr3_size: Length of 3'UTR region.
        cds_seq: Original CDS nucleotide sequence (for synonymous constraints).
        utr5_seq: Original 5'UTR nucleotide sequence (for seed initialization).
        utr3_seq: Original 3'UTR nucleotide sequence (for seed initialization).
        target_col: Index of target cell type in RiboNN output.
        off_target_cols: Indices of off-target cell types.
        lam: Off-target penalty weight.
        obj_mode: "linear" or "ratio" specificity formula.
        n_steps: Optimization steps.
        n_repeats: Independent optimization runs (keep best).
        top_k: Number of top candidates to keep across repeats.
        lr: Learning rate for Adam.
        temp_start: Initial Gumbel-softmax temperature.
        temp_end: Final Gumbel-softmax temperature.
        entropy_weight: Entropy regularizer weight on UTR logits.
        device: Torch device.
    """

    def __init__(
        self,
        wrapper,
        seq_len: int,
        utr5_size: int = 0,
        cds_size: int = 0,
        utr3_size: int = 0,
        cds_seq: str | None = None,
        utr5_seq: str | None = None,
        utr3_seq: str | None = None,
        target_col: int = 0,
        off_target_cols: list[int] | None = None,
        lam: float = 1.0,
        obj_mode: str = "linear",
        n_steps: int = 500,
        n_repeats: int = 1,
        top_k: int = 1,
        lr: float = 0.05,
        temp_start: float = 2.0,
        temp_end: float = 0.1,
        entropy_weight: float = 0.1,
        device: str = "cpu",
    ):
        self.wrapper = wrapper
        self.seq_len = seq_len
        self.utr5_size = utr5_size
        self.cds_size = cds_size
        self.utr3_size = utr3_size
        self.target_col = target_col
        self.off_target_cols = off_target_cols or []
        self.lam = lam
        self.obj_mode = obj_mode
        self.n_steps = n_steps
        self.n_repeats = n_repeats
        self.top_k = top_k
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.entropy_weight = entropy_weight
        self.device = torch.device(device)

        # UTR seed sequences (for biased initialization)
        self.utr5_seq = utr5_seq.upper().replace("T", "U") if utr5_seq else None
        self.utr3_seq = utr3_seq.upper().replace("T", "U") if utr3_seq else None

        # CDS codon constraint tables
        self.cds_seq = cds_seq.upper().replace("T", "U") if cds_seq else None
        self._codon_onehots = None
        self._codon_masks = None
        self._max_syn = 0

        if self.cds_seq and len(self.cds_seq) >= 3:
            usable_len = (len(self.cds_seq) // 3) * 3
            self.cds_seq = self.cds_seq[:usable_len]
            self.cds_size = usable_len
            self._codon_onehots, self._codon_masks, self._max_syn = (
                _build_synonymous_tables(self.cds_seq, self.device)
            )
            n_codons = usable_len // 3
            print(f"  CDS codon constraint: {n_codons} codons, "
                  f"max {self._max_syn} synonymous alternatives per codon")

        # Pre-build codon label channel
        self._codon_channel = self._build_codon_channel()

    def _build_codon_channel(self) -> Tensor | None:
        """Build the codon label channel marking first nt of each codon."""
        if self.wrapper.input_channels <= 4:
            return None
        extra = torch.zeros(
            self.wrapper.input_channels - 4, self.seq_len,
            device=self.device,
        )
        if self.cds_size > 0:
            cds_start = self.utr5_size
            cds_end = self.utr5_size + self.cds_size - 3
            for pos in range(cds_start, min(cds_end + 1, self.seq_len), 3):
                extra[0, pos] = 1.0
        return extra

    def _temperature_schedule(self, step: int) -> float:
        """Exponential annealing from temp_start to temp_end."""
        ratio = step / max(self.n_steps - 1, 1)
        return self.temp_start * (self.temp_end / self.temp_start) ** ratio

    def _init_codon_logits(self) -> Tensor:
        """Initialize codon logits biased toward the original codons."""
        n_codons = len(self.cds_seq) // 3
        logits = torch.zeros(n_codons, self._max_syn, device=self.device)
        for i in range(n_codons):
            codon = self.cds_seq[i * 3:(i + 1) * 3]
            aa = CODON_TO_AA.get(codon, None)
            if aa is None:
                logits[i, 0] = 3.0
                continue
            synonymous = AA_TO_CODONS[aa]
            for j, syn in enumerate(synonymous):
                if syn == codon:
                    logits[i, j] = 3.0
                    break
        return logits

    def _init_utr_logits(self) -> Tensor:
        """Initialize UTR logits biased toward the seed UTR nucleotides.

        If seed UTR sequences are provided, the logit for the original
        nucleotide at each position is set to 3.0 (same bias as CDS codons).
        Otherwise falls back to small random initialization.
        """
        utr_total = self.utr5_size + self.utr3_size
        logits = torch.randn(4, utr_total, device=self.device) * 0.1

        # Bias 5'UTR positions toward seed nucleotides
        if self.utr5_seq and self.utr5_size > 0:
            seed_len = min(len(self.utr5_seq), self.utr5_size)
            for pos in range(seed_len):
                nuc = self.utr5_seq[pos]
                idx = NUC_IDX.get(nuc)
                if idx is not None:
                    logits[:, pos] = 0.0
                    logits[idx, pos] = 3.0

        # Bias 3'UTR positions toward seed nucleotides
        if self.utr3_seq and self.utr3_size > 0:
            seed_len = min(len(self.utr3_seq), self.utr3_size)
            offset = self.utr5_size
            for pos in range(seed_len):
                nuc = self.utr3_seq[pos]
                idx = NUC_IDX.get(nuc)
                if idx is not None:
                    logits[:, offset + pos] = 0.0
                    logits[idx, offset + pos] = 3.0

        return logits

    def _optimize_once(self, verbose: bool = True) -> DirectResult:
        """Run a single optimization pass.

        Returns:
            DirectResult with optimized sequence and trajectory.
        """
        has_codon_constraint = self._codon_onehots is not None and self.cds_size > 0
        utr3_opt_len = self.utr3_size
        pad_len = self.seq_len - (self.utr5_size + self.cds_size + utr3_opt_len)
        bio_len = self.utr5_size + self.cds_size + utr3_opt_len

        # Initialize learnable parameters
        params = []

        # UTR logits (5'UTR + 3'UTR only) — biased toward seed UTRs
        utr_total = self.utr5_size + utr3_opt_len
        if utr_total > 0:
            utr_logits = nn.Parameter(self._init_utr_logits())
            params.append(utr_logits)
        else:
            utr_logits = None

        # CDS codon logits
        if has_codon_constraint:
            codon_logits = nn.Parameter(self._init_codon_logits())
            params.append(codon_logits)
        else:
            codon_logits = None

        optimizer = torch.optim.Adam(params, lr=self.lr)
        history: list[float] = []
        best_score = -float("inf")
        best_step_logits = None

        for step in range(self.n_steps):
            optimizer.zero_grad()
            temp = self._temperature_schedule(step)

            # Build soft sequence
            parts = []

            if self.utr5_size > 0 and utr_logits is not None:
                utr5_soft = F.gumbel_softmax(
                    utr_logits[:, :self.utr5_size].T,
                    tau=temp, hard=True, dim=-1
                ).T
                parts.append(utr5_soft)

            if has_codon_constraint:
                cds_soft = _codon_logits_to_soft_seq(
                    codon_logits, self._codon_onehots,
                    self._codon_masks, temp,
                )
                parts.append(cds_soft)

            if utr3_opt_len > 0 and utr_logits is not None:
                tail_soft = F.gumbel_softmax(
                    utr_logits[:, self.utr5_size:].T,
                    tau=temp, hard=True, dim=-1
                ).T
                parts.append(tail_soft)

            if parts:
                soft_seq = torch.cat(parts, dim=1)
            else:
                soft_seq = torch.zeros(4, self.seq_len, device=self.device)

            # Zero-pad to model's seq_len
            if pad_len > 0 and len(parts) > 0:
                padding = torch.zeros(4, pad_len, device=self.device)
                soft_seq = torch.cat([soft_seq, padding], dim=1)

            # Add codon label channel if needed
            input_seq = soft_seq
            if self._codon_channel is not None:
                input_seq = torch.cat([soft_seq, self._codon_channel], dim=0)

            # Forward pass through full CNN
            te = self.wrapper.predict_with_grad(input_seq.unsqueeze(0))
            te = te.squeeze(0)
            target_te = te[self.target_col]
            off_target_te = te[self.off_target_cols].mean()
            specificity = _compute_specificity(
                target_te.unsqueeze(0), off_target_te.unsqueeze(0),
                self.lam, self.obj_mode,
            ).squeeze(0)

            # Loss = maximize specificity + entropy regularizer
            loss = -specificity
            if utr_logits is not None and self.entropy_weight > 0:
                loss = loss + self.entropy_weight * sequence_entropy(utr_logits)

            loss.backward()
            optimizer.step()

            score = specificity.item()
            history.append(score)

            if score > best_score:
                best_score = score
                # Save the logit state at best score
                best_step_logits = {
                    "codon": codon_logits.detach().clone() if codon_logits is not None else None,
                    "utr": utr_logits.detach().clone() if utr_logits is not None else None,
                }

            if verbose and step % 50 == 0:
                nuc_fracs = soft_seq[:, :bio_len].mean(dim=1) if bio_len > 0 else soft_seq.mean(dim=1)
                a_f, u_f, c_f, g_f = nuc_fracs.detach().cpu().tolist()
                print(
                    f"  [Direct] Step {step:4d}/{self.n_steps} | "
                    f"spec={score:.4f} TE={target_te.item():.4f} | "
                    f"A={a_f:.1%} U={u_f:.1%} C={c_f:.1%} G={g_f:.1%} | "
                    f"temp={temp:.3f}"
                )

        # Restore best logits
        if best_step_logits is not None:
            if codon_logits is not None and best_step_logits["codon"] is not None:
                codon_logits.data.copy_(best_step_logits["codon"])
            if utr_logits is not None and best_step_logits["utr"] is not None:
                utr_logits.data.copy_(best_step_logits["utr"])

        # Final discrete sequence
        with torch.no_grad():
            final_parts = []

            if self.utr5_size > 0 and utr_logits is not None:
                utr5_hard = F.one_hot(
                    utr_logits[:, :self.utr5_size].detach().argmax(dim=0), 4
                ).T.float()
                final_parts.append(utr5_hard)

            if has_codon_constraint:
                masked = codon_logits.detach() + (1 - self._codon_masks) * (-1e9)
                best_codon_idx = masked.argmax(dim=1)
                n_codons = len(self.cds_seq) // 3
                cds_hard = torch.zeros(4, self.cds_size, device=self.device)
                for i in range(n_codons):
                    j = best_codon_idx[i].item()
                    cds_hard[:, i * 3:(i + 1) * 3] = self._codon_onehots[i, j]
                final_parts.append(cds_hard)

            if utr3_opt_len > 0 and utr_logits is not None:
                tail_hard = F.one_hot(
                    utr_logits[:, self.utr5_size:].detach().argmax(dim=0), 4
                ).T.float()
                final_parts.append(tail_hard)

            if final_parts:
                hard_seq = torch.cat(final_parts, dim=1)
            else:
                hard_seq = torch.zeros(4, bio_len, device=self.device)

            # Decode biological positions only
            sequence = decode_logits(hard_seq)

            # Compute final embedding for pipeline compatibility
            full_hard = hard_seq
            if pad_len > 0:
                full_hard = torch.cat(
                    [hard_seq, torch.zeros(4, pad_len, device=self.device)], dim=1
                )
            encode_input = full_hard
            if self._codon_channel is not None:
                encode_input = torch.cat([full_hard, self._codon_channel], dim=0)
            z_final = self.wrapper.encode(encode_input.unsqueeze(0)).squeeze(0)

        return DirectResult(
            best_z=z_final,
            best_score=best_score,
            history=history,
            sequence=sequence,
            logits=hard_seq.cpu(),
        )

    def optimize(self, verbose: bool = True) -> DirectResult:
        """Run direct codon optimization with optional repeats.

        If n_repeats > 1, runs multiple independent optimization passes
        (each with fresh random Gumbel noise) and returns the best result.
        When top_k > 1, collects the top-K candidates across all repeats.

        Returns:
            DirectResult with optimized sequence, trajectory, and top-K candidates.
        """
        if self.n_repeats <= 1:
            result = self._optimize_once(verbose=verbose)
            if self.top_k > 1:
                result.candidates = [DirectCandidate(
                    sequence=result.sequence,
                    score=result.best_score,
                    z=result.best_z,
                    logits=result.logits,
                )]
            return result

        # Collect all repeat results as candidates
        all_candidates: list[DirectCandidate] = []
        best_result: DirectResult | None = None

        for rep in range(self.n_repeats):
            if verbose:
                print(f"\n--- Direct repeat {rep + 1}/{self.n_repeats} ---")
            result = self._optimize_once(verbose=verbose)

            all_candidates.append(DirectCandidate(
                sequence=result.sequence,
                score=result.best_score,
                z=result.best_z,
                logits=result.logits,
            ))

            if best_result is None or result.best_score > best_result.best_score:
                best_result = result
                if verbose:
                    print(f"  New best: {result.best_score:.4f}")

        # Sort by score descending and keep top-K
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        keep_k = min(self.top_k, len(all_candidates))
        best_result.candidates = all_candidates[:keep_k]

        if verbose:
            print(f"\nBest across {self.n_repeats} repeats: {best_result.best_score:.4f}")
            if keep_k > 1:
                print(f"Kept top {keep_k} candidates (scores: "
                      f"{', '.join(f'{c.score:.4f}' for c in best_result.candidates)})")
        return best_result
