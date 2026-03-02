"""Gradient-based sequence inversion: find the mRNA sequence whose RiboNN
embedding is closest to a target latent vector z*.

Strategy:
    1. For UTR regions: optimize (4, L_utr) nucleotide logits freely
    2. For CDS region: optimize codon-level logits over synonymous codons only,
       guaranteeing the encoded protein is preserved
    3. Apply Gumbel-softmax to get a differentiable soft one-hot encoding
    4. Pass through RiboNN encoder (with gradients) to get z
    5. Minimize ||z - z*||^2 + composition penalties + entropy regularizer
    6. Anneal Gumbel temperature from high (soft) to low (discrete)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rnaflow.data.encoding import decode_logits, sequence_entropy
from rnaflow.data.codon_table import CODON_TO_AA, AA_TO_CODONS, NUC_IDX


@dataclass
class InversionResult:
    """Result of gradient-based sequence inversion."""
    sequence: str
    logits: Tensor
    final_loss: float
    latent_distance: float
    loss_history: list[float] = field(default_factory=list)


def _build_synonymous_tables(
    cds_seq: str,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """Build lookup tables for synonymous codon optimization.

    For each codon in the CDS, creates:
    - codon_onehots: (n_codons, max_syn, 4, 3) one-hot encoding of each
      synonymous codon alternative. Padded with zeros.
    - codon_masks: (n_codons, max_syn) binary mask for valid synonymous codons.

    Args:
        cds_seq: CDS nucleotide sequence (RNA, multiple of 3).
        device: Torch device.

    Returns:
        codon_onehots, codon_masks, max_syn
    """
    cds_seq = cds_seq.upper().replace("T", "U")
    n_codons = len(cds_seq) // 3

    # Find max number of synonymous codons across all codons
    max_syn = 1
    for i in range(n_codons):
        codon = cds_seq[i*3:(i+1)*3]
        aa = CODON_TO_AA.get(codon, None)
        if aa is not None:
            max_syn = max(max_syn, len(AA_TO_CODONS[aa]))

    codon_onehots = torch.zeros(n_codons, max_syn, 4, 3, device=device)
    codon_masks = torch.zeros(n_codons, max_syn, device=device)

    for i in range(n_codons):
        codon = cds_seq[i*3:(i+1)*3]
        aa = CODON_TO_AA.get(codon, None)
        if aa is None:
            # Unknown codon — fix it as-is
            for p, nuc in enumerate(codon):
                if nuc in NUC_IDX:
                    codon_onehots[i, 0, NUC_IDX[nuc], p] = 1.0
            codon_masks[i, 0] = 1.0
            continue

        synonymous = AA_TO_CODONS[aa]
        for j, syn_codon in enumerate(synonymous):
            for p, nuc in enumerate(syn_codon):
                codon_onehots[i, j, NUC_IDX[nuc], p] = 1.0
            codon_masks[i, j] = 1.0

    return codon_onehots, codon_masks, max_syn


def _codon_logits_to_soft_seq(
    codon_logits: Tensor,
    codon_onehots: Tensor,
    codon_masks: Tensor,
    temperature: float,
) -> Tensor:
    """Convert codon-level logits to a soft nucleotide sequence.

    Args:
        codon_logits: (n_codons, max_syn) raw logits over synonymous codons.
        codon_onehots: (n_codons, max_syn, 4, 3) one-hot codon encodings.
        codon_masks: (n_codons, max_syn) valid codon mask.
        temperature: Gumbel-softmax temperature.

    Returns:
        soft_seq: (4, n_codons*3) soft one-hot nucleotide sequence.
    """
    # Mask invalid synonymous codons with -inf
    masked_logits = codon_logits + (1 - codon_masks) * (-1e9)

    # Gumbel-softmax over synonymous codons
    codon_probs = F.gumbel_softmax(masked_logits, tau=temperature, hard=False, dim=-1)
    # codon_probs: (n_codons, max_syn)

    # Weighted sum of codon one-hots: (n_codons, 4, 3)
    soft_codons = torch.einsum("ns,nspq->npq", codon_probs, codon_onehots)

    # Reshape to (4, n_codons*3)
    n_codons = codon_logits.shape[0]
    soft_seq = soft_codons.permute(1, 0, 2).reshape(4, n_codons * 3)

    return soft_seq


class GradientDecoder:
    """Recover an mRNA sequence from a target latent vector via gradient descent.

    For CDS positions, optimization is constrained to synonymous codon
    substitutions — the encoded protein is always preserved. UTR positions
    are optimized freely.

    When objective parameters (target_col, off_target_cols) are provided,
    the inversion also optimizes the RiboNN prediction directly — codon
    choices are guided by what the model predicts is best for cell-type
    specificity, not just by latent distance.

    Args:
        wrapper: RiboNNWrapper with encode_with_grad and predict_with_grad.
        seq_len: Length of the sequence to generate.
        n_steps: Number of optimization steps.
        lr: Learning rate for the logits optimizer.
        temp_start: Initial Gumbel-softmax temperature (high = soft).
        temp_end: Final Gumbel-softmax temperature (low = discrete).
        entropy_weight: Weight for the entropy regularizer.
        l2_weight: Optional L2 regularization on logits.
        utr5_size: Length of 5'UTR region.
        cds_size: Length of CDS including start+stop codons.
        cds_seq: Actual CDS nucleotide sequence. When provided, CDS positions
            are constrained to synonymous codon substitutions only.
        nuc_targets: Target fraction for each nucleotide [A, U, C, G].
            Default [0.25, 0.25, 0.25, 0.25] (balanced).
        composition_weight: Penalty for nucleotide composition deviation.
            Loss = weight * sum((actual_i - target_i)^2 for each nucleotide).
        target_col: Index of the target cell type in RiboNN output.
        off_target_cols: Indices of off-target cell types.
        obj_weight: Weight for the objective loss during inversion.
            Higher values make codon choices follow the model's predictions
            more strongly (vs. just matching the latent vector).
        lam: Off-target penalty weight (same as CEM objective).
        device: Torch device.
    """

    def __init__(
        self,
        wrapper,  # RiboNNWrapper
        seq_len: int = 2048,
        n_steps: int = 500,
        lr: float = 0.05,
        temp_start: float = 2.0,
        temp_end: float = 0.1,
        entropy_weight: float = 0.1,
        l2_weight: float = 0.0,
        utr5_size: int = 0,
        cds_size: int = 0,
        cds_seq: str | None = None,
        nuc_targets: list[float] | None = None,
        composition_weight: float = 5.0,
        target_col: int | None = None,
        off_target_cols: list[int] | None = None,
        obj_weight: float = 1.0,
        lam: float = 1.0,
        device: str = "cpu",
    ):
        self.wrapper = wrapper
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.entropy_weight = entropy_weight
        self.l2_weight = l2_weight
        self.utr5_size = utr5_size
        self.cds_size = cds_size
        # Nucleotide composition targets: [A, U, C, G]
        self.nuc_targets = torch.tensor(
            nuc_targets if nuc_targets else [0.25, 0.25, 0.25, 0.25],
            dtype=torch.float32,
        )
        self.composition_weight = composition_weight
        # Objective-aware inversion parameters
        self.target_col = target_col
        self.off_target_cols = off_target_cols or []
        self.obj_weight = obj_weight
        self.lam = lam
        self.use_obj = target_col is not None and obj_weight > 0
        self.device = torch.device(device)

        # CDS codon constraint tables
        self.cds_seq = cds_seq.upper().replace("T", "U") if cds_seq else None
        self._codon_onehots = None
        self._codon_masks = None
        self._max_syn = 0

        if self.cds_seq and len(self.cds_seq) >= 3:
            # Ensure CDS length is multiple of 3
            usable_len = (len(self.cds_seq) // 3) * 3
            self.cds_seq = self.cds_seq[:usable_len]
            self.cds_size = usable_len

            self._codon_onehots, self._codon_masks, self._max_syn = (
                _build_synonymous_tables(self.cds_seq, self.device)
            )
            n_codons = usable_len // 3
            print(f"  CDS codon constraint: {n_codons} codons, "
                  f"max {self._max_syn} synonymous alternatives per codon")

        # Pre-build the codon label channel (static, no gradients needed)
        self._codon_channel = self._build_codon_channel()

    def _build_codon_channel(self) -> Tensor | None:
        """Build the codon label channel marking first nt of each codon in CDS."""
        if self.wrapper.input_channels <= 4:
            return None

        extra = torch.zeros(
            self.wrapper.input_channels - 4, self.seq_len,
            device=self.device,
        )

        if self.cds_size > 0:
            cds_start = self.utr5_size
            cds_end = self.utr5_size + self.cds_size - 3  # exclude stop codon
            for pos in range(cds_start, min(cds_end + 1, self.seq_len), 3):
                extra[0, pos] = 1.0

        return extra

    def _temperature_schedule(self, step: int) -> float:
        """Exponential annealing from temp_start to temp_end."""
        ratio = step / max(self.n_steps - 1, 1)
        return self.temp_start * (self.temp_end / self.temp_start) ** ratio

    def _init_codon_logits(self) -> Tensor:
        """Initialize codon logits biased toward the original codons.

        The original codon gets a logit of 3.0 (strong bias), others get 0.0.
        This ensures the optimizer starts from the seed sequence and only
        changes codons when it clearly helps the objective.
        """
        n_codons = len(self.cds_seq) // 3
        logits = torch.zeros(n_codons, self._max_syn, device=self.device)

        for i in range(n_codons):
            codon = self.cds_seq[i*3:(i+1)*3]
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

    def invert(
        self,
        z_target: Tensor,
        init_logits: Tensor | None = None,
        mask: Tensor | None = None,
        verbose: bool = True,
    ) -> InversionResult:
        """Find the sequence whose embedding is closest to z_target.

        If a CDS sequence was provided, the CDS region is constrained to
        synonymous codon substitutions. UTR regions are optimized freely.

        Args:
            z_target: (latent_dim,) target latent vector.
            init_logits: Optional (4, seq_len) initialization for UTR logits.
            mask: Optional (seq_len,) boolean mask.
            verbose: Print progress.

        Returns:
            InversionResult with the recovered sequence and diagnostics.
        """
        z_target = z_target.to(self.device).detach()

        cds_start = self.utr5_size
        cds_end = cds_start + self.cds_size
        utr_tail_len = self.seq_len - cds_end  # positions after CDS (3'UTR + padding)

        has_codon_constraint = (
            self._codon_onehots is not None and self.cds_size > 0
        )

        # Initialize learnable parameters
        params = []

        # UTR logits: all non-CDS positions (5'UTR + 3'UTR + padding)
        utr_total = self.utr5_size + utr_tail_len
        if utr_total > 0:
            utr_logits = torch.randn(4, utr_total, device=self.device) * 0.1
            utr_logits = nn.Parameter(utr_logits)
            params.append(utr_logits)
        else:
            utr_logits = None

        # CDS logits: codon-level or nucleotide-level
        if has_codon_constraint:
            codon_logits = nn.Parameter(self._init_codon_logits())
            params.append(codon_logits)
            cds_nuc_logits = None
        elif self.cds_size > 0:
            cds_nuc_logits = torch.randn(4, self.cds_size, device=self.device) * 0.1
            cds_nuc_logits = nn.Parameter(cds_nuc_logits)
            params.append(cds_nuc_logits)
            codon_logits = None
        else:
            cds_nuc_logits = None
            codon_logits = None

        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_history = []

        for step in range(self.n_steps):
            optimizer.zero_grad()

            temp = self._temperature_schedule(step)

            # Build the full soft sequence by assembling regions
            parts = []

            # 5'UTR
            if self.utr5_size > 0 and utr_logits is not None:
                utr5_soft = F.gumbel_softmax(
                    utr_logits[:, :self.utr5_size].T,
                    tau=temp, hard=False, dim=-1
                ).T  # (4, utr5_size)
                parts.append(utr5_soft)

            # CDS
            if has_codon_constraint:
                cds_soft = _codon_logits_to_soft_seq(
                    codon_logits, self._codon_onehots,
                    self._codon_masks, temp,
                )  # (4, cds_size)
                parts.append(cds_soft)
            elif cds_nuc_logits is not None:
                cds_soft = F.gumbel_softmax(
                    cds_nuc_logits.T, tau=temp, hard=False, dim=-1
                ).T
                parts.append(cds_soft)

            # 3'UTR + padding
            if utr_tail_len > 0 and utr_logits is not None:
                tail_soft = F.gumbel_softmax(
                    utr_logits[:, self.utr5_size:].T,
                    tau=temp, hard=False, dim=-1
                ).T  # (4, utr_tail_len)
                parts.append(tail_soft)

            if parts:
                soft_seq = torch.cat(parts, dim=1)  # (4, seq_len)
            else:
                soft_seq = torch.zeros(4, self.seq_len, device=self.device)

            # Apply mask if provided
            if mask is not None:
                soft_seq = soft_seq * mask.float().unsqueeze(0).to(self.device)

            # Add codon label channel if model expects it
            input_seq = soft_seq
            if self._codon_channel is not None:
                input_seq = torch.cat([soft_seq, self._codon_channel], dim=0)

            # Encode through RiboNN (with gradients)
            z = self.wrapper.encode_with_grad(input_seq.unsqueeze(0))
            z = z.squeeze(0)

            # Reconstruction loss
            recon_loss = F.mse_loss(z, z_target)

            # Objective-aware loss: run full forward pass to get TE predictions
            # and optimize for cell-type specificity directly
            obj_loss = torch.tensor(0.0, device=self.device)
            if self.use_obj:
                te = self.wrapper.predict_with_grad(input_seq.unsqueeze(0))
                te = te.squeeze(0)  # (num_targets,)
                target_te = te[self.target_col]
                off_target_te = te[self.off_target_cols].mean()
                specificity = target_te - self.lam * off_target_te
                # Negate because we want to MAXIMIZE specificity
                obj_loss = -specificity

            # Entropy regularizer on UTR logits only (CDS is codon-level)
            ent_loss = torch.tensor(0.0, device=self.device)
            if utr_logits is not None:
                ent_loss = sequence_entropy(utr_logits)

            # Nucleotide composition loss — UTR positions only
            # Uses LOCAL (windowed) composition to prevent homopolymer runs.
            # A global average allows the optimizer to concentrate one base in
            # long stretches while compensating elsewhere.
            # Channel order: A=0, U=1, C=2, G=3
            targets = self.nuc_targets.to(self.device)
            if utr_logits is not None and utr_total > 0:
                utr_soft = F.softmax(utr_logits, dim=0)  # (4, utr_total)

                # Local composition in sliding windows
                window = min(100, utr_total)
                if utr_total >= window:
                    # avg_pool1d expects (N, C, L) — use utr_soft as (1, 4, L)
                    local_fracs = F.avg_pool1d(
                        utr_soft.unsqueeze(0), kernel_size=window,
                        stride=window // 2, padding=0,
                    ).squeeze(0)  # (4, n_windows)
                    # Penalize deviation in each window
                    comp_loss = ((local_fracs - targets.unsqueeze(1)) ** 2).sum(dim=0).mean()
                else:
                    utr_nuc_fracs = utr_soft.mean(dim=1)
                    comp_loss = ((utr_nuc_fracs - targets) ** 2).sum()
            else:
                utr_nuc_fracs = soft_seq.mean(dim=1)
                comp_loss = ((utr_nuc_fracs - targets) ** 2).sum()

            # Overall composition for monitoring
            nuc_fracs = soft_seq.mean(dim=1)  # (4,) full sequence

            # Total loss
            loss = recon_loss + self.entropy_weight * ent_loss
            if self.use_obj:
                loss = loss + self.obj_weight * obj_loss
            if self.composition_weight > 0:
                loss = loss + self.composition_weight * comp_loss
            if self.l2_weight > 0 and utr_logits is not None:
                loss = loss + self.l2_weight * (utr_logits ** 2).mean()

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and step % 50 == 0:
                a_f, u_f, c_f, g_f = nuc_fracs.detach().cpu().tolist()
                obj_str = ""
                if self.use_obj:
                    obj_str = f" | obj={-obj_loss.item():.4f}"
                print(
                    f"  [Inversion] Step {step:4d} | loss={loss.item():.6f} | "
                    f"recon={recon_loss.item():.6f}{obj_str} | "
                    f"A={a_f:.1%} U={u_f:.1%} C={c_f:.1%} G={g_f:.1%} | "
                    f"temp={temp:.3f}"
                )

        # Final discrete sequence
        with torch.no_grad():
            # Build the final sequence
            final_parts = []

            # 5'UTR: argmax
            if self.utr5_size > 0 and utr_logits is not None:
                utr5_logits_final = utr_logits[:, :self.utr5_size].detach()
                utr5_hard = F.one_hot(utr5_logits_final.argmax(dim=0), 4).T.float()
                final_parts.append(utr5_hard)

            # CDS: pick best synonymous codon per position
            if has_codon_constraint:
                masked = codon_logits.detach() + (1 - self._codon_masks) * (-1e9)
                best_codon_idx = masked.argmax(dim=1)  # (n_codons,)
                n_codons = len(self.cds_seq) // 3
                cds_hard = torch.zeros(4, self.cds_size, device=self.device)
                for i in range(n_codons):
                    j = best_codon_idx[i].item()
                    cds_hard[:, i*3:(i+1)*3] = self._codon_onehots[i, j]
                final_parts.append(cds_hard)
            elif cds_nuc_logits is not None:
                cds_hard = F.one_hot(cds_nuc_logits.detach().argmax(dim=0), 4).T.float()
                final_parts.append(cds_hard)

            # 3'UTR + padding
            if utr_tail_len > 0 and utr_logits is not None:
                tail_logits_final = utr_logits[:, self.utr5_size:].detach()
                tail_hard = F.one_hot(tail_logits_final.argmax(dim=0), 4).T.float()
                final_parts.append(tail_hard)

            if final_parts:
                hard_seq = torch.cat(final_parts, dim=1)
            else:
                hard_seq = torch.zeros(4, self.seq_len, device=self.device)

            if mask is not None:
                hard_seq = hard_seq * mask.float().unsqueeze(0).to(self.device)

            # Decode to string
            sequence = decode_logits(hard_seq)

            # Compute final latent distance
            encode_input = hard_seq
            if self._codon_channel is not None:
                encode_input = torch.cat([hard_seq, self._codon_channel], dim=0)
            z_final = self.wrapper.encode(encode_input.unsqueeze(0)).squeeze(0)
            latent_dist = (z_final - z_target.cpu()).norm().item()

        # Trim padded positions if mask is provided
        if mask is not None:
            active_len = mask.sum().item()
            sequence = sequence[:active_len]

        return InversionResult(
            sequence=sequence,
            logits=hard_seq.cpu(),
            final_loss=loss_history[-1],
            latent_distance=latent_dist,
            loss_history=loss_history,
        )


class BatchGradientDecoder:
    """Invert multiple latent vectors in parallel.

    Useful for decoding the top-K candidates from CEM optimization.
    """

    def __init__(self, wrapper, seq_len: int = 2048, **kwargs):
        self.decoder = GradientDecoder(wrapper, seq_len=seq_len, **kwargs)

    def invert_batch(
        self,
        z_targets: Tensor,
        masks: Tensor | None = None,
        verbose: bool = False,
    ) -> list[InversionResult]:
        """Invert a batch of latent vectors sequentially."""
        results = []
        for i in range(z_targets.shape[0]):
            mask = masks[i] if masks is not None else None
            result = self.decoder.invert(z_targets[i], mask=mask, verbose=verbose)
            results.append(result)
            if not verbose:
                print(f"  Inverted {i+1}/{z_targets.shape[0]} | "
                      f"dist={result.latent_distance:.4f}")
        return results
