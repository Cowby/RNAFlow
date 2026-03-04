"""Run FlowCEM optimization to find mRNA sequences with high target cell specificity.

Usage with sequence input (recommended):
    python scripts/optimize.py \
        --ribonn-state-dict checkpoints/human/.../state_dict.pth \
        --utr5 "ACUGGCUA..." \
        --cds "AUGAAAGGG...UAA" \
        --utr3 "AAUAAACCC..." \
        --target-cell HeLa \
        --off-target-cells HEK293T K562 A549 MCF7 \
        --objective ribonn \
        --n-iters 200

    python scripts/optimize.py --list-cell-types  # show all available cell types

Or from FASTA file:
    python scripts/optimize.py \
        --seed-fasta my_mrna.fasta \
        --utr5-len 150 --cds-len 900 \
        ...

Pipeline:
    1. Load pretrained RiboNN + optional custom predictor
    2. Assemble mRNA from UTR/CDS parts, encode, and embed as CEM starting point
    3. Run CEM optimization in 64-dim latent space
    4. Gradient-based inversion: z* -> optimized mRNA sequence
    5. Report results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add RiboNN to path if available
ribonn_root = PROJECT_ROOT / "external" / "RiboNN"
if ribonn_root.exists():
    sys.path.insert(0, str(ribonn_root))
    sys.path.insert(0, str(ribonn_root / "src"))

from rnaflow.data.cell_types import (
    HUMAN_CELL_TYPES, cell_type_to_index, index_to_cell_type,
)
from rnaflow.data.encoding import one_hot_encode_ribonn
from rnaflow.embeddings.ribonn_wrapper import RiboNNWrapper, MockRiboNN
from rnaflow.embeddings.ensemble import EnsembleRiboNNWrapper
from rnaflow.inversion.gradient_decoder import GradientDecoder, InversionResult
from rnaflow.models.predictor import TranslationPredictor
from rnaflow.optim.flow_cem import FlowCEM
from rnaflow.optim.cem import VanillaCEM
from rnaflow.optim.diffusion import DiffusionOptimizer
from rnaflow.optim.direct import DirectOptimizer
from rnaflow.optim.objective import PredictorSpecificityObjective, LatentRiboNNObjective
from rnaflow.data.codon_table import translate
from rnaflow.utils.config import load_config


def resolve_cell_type(value: str) -> int:
    """Convert a cell type name or integer string to a target index."""
    try:
        return int(value)
    except ValueError:
        return cell_type_to_index(value)


def read_fasta_sequence(path: str) -> str:
    """Read the first sequence from a FASTA file."""
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_parts:
                    break  # only read first sequence
                continue
            seq_parts.append(line)
    return "".join(seq_parts)


def clean_seq(s: str) -> str:
    """Normalize a nucleotide sequence: uppercase, strip whitespace."""
    return s.upper().replace(" ", "").replace("\n", "").replace("\r", "")


def plot_optimization(result, output_path: str):
    """Plot optimization trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Objective score over iterations
    axes[0].plot(result.history)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective Score")
    axes[0].set_title("Optimization Progress")

    # Time schedule (if FlowCEM)
    if hasattr(result, "time_history") and result.time_history:
        axes[1].plot(result.time_history)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("t (interpolation)")
        axes[1].set_title("Flow Schedule")

    # Sigma evolution (if FlowCEM)
    if hasattr(result, "sigma_history") and result.sigma_history:
        axes[2].plot(result.sigma_history)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Mean sigma")
        axes[2].set_title("Distribution Width")

    # Noise level (if Diffusion)
    if hasattr(result, "noise_history") and result.noise_history:
        axes[2].plot(result.noise_history)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Noise level")
        axes[2].set_title("Diffusion Noise Schedule")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="FlowCEM mRNA optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--list-cell-types", action="store_true",
                        help="Print all available cell type names and exit")

    # ── Model paths ───────────────────────────────────────────────────────
    model_group = parser.add_argument_group("Model selection")
    model_group.add_argument("--ribonn-checkpoint", type=str, default=None,
                             help="Path to RiboNN .ckpt (Lightning) checkpoint")
    model_group.add_argument("--ribonn-state-dict", type=str, default=None,
                             help="Path to RiboNN state_dict.pth (auto-detected from checkpoints/ if omitted)")
    model_group.add_argument("--organism", type=str, default="human",
                             choices=["human", "mouse"],
                             help="Organism model to use (for auto-detection from checkpoints/)")
    model_group.add_argument("--checkpoints-dir", type=str, default=None,
                             help="Root directory for pretrained weights (default: checkpoints/)")
    model_group.add_argument("--predictor-checkpoint", type=str, default=None,
                             help="Path to trained predictor .pt file")
    model_group.add_argument("--use-mock", action="store_true",
                             help="Use MockRiboNN for testing without pretrained weights")
    model_group.add_argument("--ensemble-size", type=int, default=5,
                             help="Number of top models (by val_r2) to ensemble. "
                                  "0 = single model (legacy). Default: 5")

    # ── Sequence input ────────────────────────────────────────────────────
    seq_group = parser.add_argument_group(
        "Sequence input",
        "Provide the mRNA structure. The full transcript is assembled as "
        "5'UTR + CDS + 3'UTR. This initializes the CEM search and provides "
        "codon labels for gradient inversion."
    )
    seq_group.add_argument("--utr5", type=str, default=None,
                           help="5'UTR nucleotide sequence (e.g. ACUGGCUA...)")
    seq_group.add_argument("--cds", type=str, default=None,
                           help="CDS nucleotide sequence incl. start AUG and stop codon")
    seq_group.add_argument("--utr3", type=str, default=None,
                           help="3'UTR nucleotide sequence")
    seq_group.add_argument("--seed-fasta", type=str, default=None,
                           help="Full mRNA sequence from FASTA file (alternative to --utr5/--cds/--utr3)")
    seq_group.add_argument("--utr5-len", type=int, default=0,
                           help="5'UTR length (only needed with --seed-fasta)")
    seq_group.add_argument("--cds-len", type=int, default=0,
                           help="CDS length (only needed with --seed-fasta)")

    # ── Objective ─────────────────────────────────────────────────────────
    obj_group = parser.add_argument_group("Objective")
    obj_group.add_argument("--objective", choices=["predictor", "ribonn"], default="ribonn",
                           help="'ribonn' uses RiboNN's head directly; 'predictor' uses a trained MLP")
    obj_group.add_argument("--target-cell", type=str, default="HeLa",
                           help="Target cell type name or index (e.g. HeLa or 22)")
    obj_group.add_argument("--off-target-cells", type=str, nargs="+",
                           default=["HEK293T", "K562", "A549", "MCF7"],
                           help="Off-target cell types (names or indices)")
    obj_group.add_argument("--lam", type=float, default=1.0,
                           help="Off-target penalty weight. Higher = stronger off-target suppression")
    obj_group.add_argument("--obj-mode", choices=["linear", "ratio"], default="linear",
                           help="Specificity formula: 'linear' = TE_t - lam*TE_off; "
                                "'ratio' = log(TE_t) - lam*log(TE_off) (forces true ratio improvement)")

    # ── CEM optimizer ─────────────────────────────────────────────────────
    cem_group = parser.add_argument_group("CEM optimizer")
    cem_group.add_argument("--optimizer", choices=["flow", "vanilla", "diffusion", "direct", "combined"],
                           default="flow",
                           help="'flow' = FlowCEM; 'vanilla' = standard CEM; 'diffusion' = DDPM with guidance; "
                                "'direct' = gradient descent on codons through full CNN (no latent stage); "
                                "'combined' = diffusion + inversion + direct refinement")
    cem_group.add_argument("--pop-size", type=int, default=512,
                           help="Candidates per iteration. Larger = better exploration, slower")
    cem_group.add_argument("--elite-frac", type=float, default=0.05,
                           help="Fraction of top samples kept. Smaller = more selective, riskier")
    cem_group.add_argument("--n-iters", type=int, default=200,
                           help="CEM iterations. More = better convergence, diminishing returns >300")
    cem_group.add_argument("--schedule", choices=["linear", "cosine", "quadratic", "sqrt"],
                           default="cosine",
                           help="Flow time schedule. cosine is recommended for most cases")
    cem_group.add_argument("--momentum", type=float, default=0.0,
                           help="EMA smoothing for elite updates (0-1). >0 stabilizes noisy objectives")

    # ── Diffusion optimizer ──────────────────────────────────────────────
    diff_group = parser.add_argument_group("Diffusion optimizer (used when --optimizer diffusion)")
    diff_group.add_argument("--guidance-scale", type=float, default=10.0,
                            help="Classifier guidance weight. Higher = stronger gradient signal")
    diff_group.add_argument("--noise-schedule", choices=["cosine", "linear"], default="cosine",
                            help="Beta schedule for diffusion process")
    diff_group.add_argument("--diffusion-steps", type=int, default=None,
                            help="Diffusion timesteps (defaults to --n-iters if not set)")
    diff_group.add_argument("--clip-grad-norm", type=float, default=1.0,
                            help="Gradient clipping norm for diffusion guidance")
    diff_group.add_argument("--n-repeats", type=int, default=1,
                            help="Number of independent optimization runs (keep best). "
                                 "Used by diffusion and direct optimizers")
    diff_group.add_argument("--top-k", type=int, default=1,
                            help="Keep top K candidates across repeats (requires --n-repeats >= K). "
                                 "Candidates saved to optimized_candidates.fasta")
    diff_group.add_argument("--proximity-weight", type=float, default=0.1,
                            help="Penalty for ||z - seed||²/dim during guidance. "
                                 "Prevents drift into unrealistic latent regions. 0 = disabled")
    diff_group.add_argument("--max-radius", type=float, default=50.0,
                            help="Hard clamp on max distance from seed embedding. "
                                 "0 = disabled. Default: 50.0")

    # ── Gradient inversion ────────────────────────────────────────────────
    inv_group = parser.add_argument_group("Gradient inversion (z* -> sequence)")
    inv_group.add_argument("--seq-len", type=int, default=None,
                           help="Output sequence length. Auto-detected from input sequences if not set")
    inv_group.add_argument("--inversion-steps", type=int, default=500,
                           help="Gradient steps for inversion. More = lower latent distance, slower")
    inv_group.add_argument("--inversion-lr", type=float, default=0.05,
                           help="Learning rate for inversion. Too high = unstable, too low = slow")

    # ── Objective-aware inversion ────────────────────────────────────────
    obj_inv_group = parser.add_argument_group(
        "Objective-aware inversion",
        "During gradient inversion, also optimize the RiboNN prediction "
        "directly (not just latent distance). This guides codon choices "
        "toward sequences the model predicts are best for the target cell."
    )
    obj_inv_group.add_argument("--obj-weight", type=float, default=1.0,
                                help="Weight for the objective loss during inversion. "
                                     "Higher = codon choices follow model predictions more. "
                                     "0 = disable (pure latent reconstruction). Default: 1.0")

    # ── Output ────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default="results",
                           help="Directory for output files")
    out_group.add_argument("--device", type=str, default="cpu",
                           help="Torch device (cpu or cuda)")

    args = parser.parse_args()

    # ── --list-cell-types ─────────────────────────────────────────────────
    if args.list_cell_types:
        print(f"Available cell types ({len(HUMAN_CELL_TYPES)}):\n")
        for i, name in enumerate(HUMAN_CELL_TYPES):
            print(f"  {i:3d}  {name}")
        sys.exit(0)

    if args.config:
        cfg = load_config(args.config)
        for key, val in vars(cfg).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, val)

    # ── Resolve cell types ────────────────────────────────────────────────
    target_cell_idx = resolve_cell_type(args.target_cell)
    off_target_idxs = [resolve_cell_type(c) for c in args.off_target_cells]
    target_name = index_to_cell_type(target_cell_idx)
    off_target_names = [index_to_cell_type(i) for i in off_target_idxs]

    # ── Assemble input sequence ───────────────────────────────────────────
    utr5_seq = ""
    cds_seq = ""
    utr3_seq = ""
    seed_sequence = None

    if args.utr5 or args.cds or args.utr3:
        # Build mRNA from parts
        utr5_seq = clean_seq(args.utr5) if args.utr5 else ""
        cds_seq = clean_seq(args.cds) if args.cds else ""
        utr3_seq = clean_seq(args.utr3) if args.utr3 else ""
        seed_sequence = utr5_seq + cds_seq + utr3_seq

        if not seed_sequence:
            print("WARNING: All sequence parts are empty.")
        else:
            print(f"Input mRNA structure:")
            print(f"  5'UTR: {len(utr5_seq)} nt")
            print(f"  CDS:   {len(cds_seq)} nt")
            print(f"  3'UTR: {len(utr3_seq)} nt")
            print(f"  Total: {len(seed_sequence)} nt")

    elif args.seed_fasta:
        seed_sequence = clean_seq(read_fasta_sequence(args.seed_fasta))
        print(f"Loaded seed sequence from {args.seed_fasta}: {len(seed_sequence)} nt")
        # Use provided lengths for structure annotation
        if args.utr5_len > 0:
            utr5_seq = seed_sequence[:args.utr5_len]
        if args.cds_len > 0:
            cds_start = args.utr5_len
            cds_seq = seed_sequence[cds_start:cds_start + args.cds_len]
            utr3_seq = seed_sequence[cds_start + args.cds_len:]

    utr5_size = len(utr5_seq)
    cds_size = len(cds_seq)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────
    if args.use_mock:
        print("Using MockRiboNN (no pretrained weights)...")
        mock_model = MockRiboNN(seq_len=args.seq_len, num_targets=len(off_target_idxs) + 1)
        wrapper = RiboNNWrapper.from_model(mock_model, device=args.device)
    elif args.ribonn_state_dict:
        # Explicit single model
        print(f"Loading RiboNN from state_dict: {args.ribonn_state_dict}...")
        wrapper = RiboNNWrapper.from_state_dict(args.ribonn_state_dict, device=args.device)
    elif args.ribonn_checkpoint:
        print(f"Loading RiboNN from checkpoint: {args.ribonn_checkpoint}...")
        wrapper = RiboNNWrapper.from_checkpoint(args.ribonn_checkpoint, device=args.device)
    else:
        # Auto-detect: try ensemble from checkpoints directory
        ckpt_root = Path(args.checkpoints_dir) if args.checkpoints_dir else PROJECT_ROOT / "checkpoints"
        search_dir = ckpt_root / args.organism
        runs_csv = search_dir / "runs.csv"

        if args.ensemble_size > 0 and search_dir.exists():
            if runs_csv.exists():
                print(f"Loading ensemble (top {args.ensemble_size} by val_r2) "
                      f"from {search_dir}...")
                wrapper = EnsembleRiboNNWrapper.from_runs_csv(
                    runs_csv, search_dir,
                    top_k=args.ensemble_size, device=args.device,
                )
            else:
                candidates = sorted(search_dir.glob("*/state_dict.pth"))
                if candidates:
                    print(f"No runs.csv found; loading ensemble of first "
                          f"{args.ensemble_size} models from {search_dir}...")
                    wrapper = EnsembleRiboNNWrapper.from_directory(
                        search_dir,
                        max_models=args.ensemble_size, device=args.device,
                    )
                else:
                    print("ERROR: No pretrained weights found. Either:")
                    print("  1. Download from https://zenodo.org/records/17258709 into checkpoints/")
                    print("  2. Provide --ribonn-state-dict or --ribonn-checkpoint explicitly")
                    print("  3. Use --use-mock for testing without pretrained weights")
                    sys.exit(1)
        elif search_dir.exists():
            # ensemble_size == 0: single-model fallback
            candidates = sorted(search_dir.glob("*/state_dict.pth"))
            if candidates:
                args.ribonn_state_dict = str(candidates[0])
                print(f"Auto-detected single {args.organism} model: {args.ribonn_state_dict}")
                wrapper = RiboNNWrapper.from_state_dict(args.ribonn_state_dict, device=args.device)
            else:
                print("ERROR: No pretrained weights found.")
                sys.exit(1)
        else:
            print("ERROR: No pretrained weights found. Either:")
            print("  1. Download from https://zenodo.org/records/17258709 into checkpoints/")
            print("  2. Provide --ribonn-state-dict or --ribonn-checkpoint explicitly")
            print("  3. Use --use-mock for testing without pretrained weights")
            sys.exit(1)

    latent_dim = wrapper.latent_dim
    print(f"  Latent dim: {latent_dim}, Targets: {wrapper.num_targets}")

    # Auto-detect seq_len: use the model's trained max_seq_len
    # RiboNN's conv architecture requires sequences padded to a specific length
    if args.seq_len is None:
        args.seq_len = wrapper.max_seq_len
        print(f"  Using model's max_seq_len: {args.seq_len}")

    if seed_sequence and len(seed_sequence) > args.seq_len:
        print(f"WARNING: Seed sequence ({len(seed_sequence)} nt) exceeds seq_len "
              f"({args.seq_len}). It will be truncated.")

    predictor = None
    if args.objective == "predictor":
        if args.predictor_checkpoint:
            print(f"Loading predictor from {args.predictor_checkpoint}...")
            predictor = TranslationPredictor.load(args.predictor_checkpoint, device=args.device)
        else:
            print("No predictor checkpoint. Training a quick synthetic predictor...")
            from scripts.train_predictor import create_synthetic_dataset, train_predictor
            from torch.utils.data import DataLoader, TensorDataset

            z, ct, eff = create_synthetic_dataset(latent_dim=latent_dim,
                                                   n_cell_types=len(off_target_idxs) + 1)
            dataset = TensorDataset(z, ct, eff)
            loader = DataLoader(dataset, batch_size=128, shuffle=True)
            predictor = TranslationPredictor(latent_dim=latent_dim,
                                              n_cell_types=len(off_target_idxs) + 1)
            train_predictor(predictor, loader, loader, epochs=20, device=args.device)

    # ── Build objective ──────────────────────────────────────────────────
    obj_mode = getattr(args, "obj_mode", "linear")
    if args.objective == "predictor" and predictor is not None:
        print(f"Objective: Predictor-based | target={target_name} ({target_cell_idx}) | "
              f"off-target={off_target_names} | lam={args.lam} | mode={obj_mode}")
        objective = PredictorSpecificityObjective(
            predictor=predictor,
            target_cell=target_cell_idx,
            off_target_cells=off_target_idxs,
            lam=args.lam,
            device=args.device,
            obj_mode=obj_mode,
        )
    else:
        print(f"Objective: Direct RiboNN (latent) | target={target_name} ({target_cell_idx}) | "
              f"off-target={off_target_names} | lam={args.lam} | mode={obj_mode}")
        objective = LatentRiboNNObjective(
            wrapper=wrapper,
            target_col=target_cell_idx,
            off_target_cols=off_target_idxs,
            lam=args.lam,
            obj_mode=obj_mode,
        )

    # ── Run optimization ─────────────────────────────────────────────────
    utr3_len = len(utr3_seq) if utr3_seq else 0

    if args.optimizer == "direct":
        # Direct codon optimization: skip latent stage entirely
        print(f"\nOptimizer: DIRECT | steps={args.inversion_steps} | "
              f"lr={args.inversion_lr} | repeats={args.n_repeats}")

        print("\n" + "=" * 60)
        print("Starting direct codon optimization...")
        print("=" * 60 + "\n")

        direct_opt = DirectOptimizer(
            wrapper=wrapper,
            seq_len=args.seq_len,
            utr5_size=utr5_size,
            cds_size=cds_size,
            utr3_size=utr3_len,
            cds_seq=cds_seq if cds_seq else None,
            utr5_seq=utr5_seq if utr5_seq else None,
            utr3_seq=utr3_seq if utr3_seq else None,
            target_col=target_cell_idx,
            off_target_cols=off_target_idxs,
            lam=args.lam,
            obj_mode=obj_mode,
            n_steps=args.inversion_steps,
            n_repeats=args.n_repeats,
            top_k=args.top_k,
            lr=args.inversion_lr,
            device=args.device,
        )
        direct_result = direct_opt.optimize(verbose=True)

        # Wrap into InversionResult for the rest of the pipeline
        inv_result = InversionResult(
            sequence=direct_result.sequence,
            logits=direct_result.logits,
            final_loss=-direct_result.best_score,
            latent_distance=0.0,
            loss_history=[],
        )
        # Create a compatible result object for score reporting
        class _ResultCompat:
            def __init__(self, dr):
                self.best_score = dr.best_score
                self.best_z = dr.best_z
                self.history = dr.history
                self.candidates = dr.candidates
        result = _ResultCompat(direct_result)

        print(f"\nBest specificity: {direct_result.best_score:.4f}")

    else:
        # Latent space optimization (flow/vanilla/diffusion)
        init_mu = None
        if seed_sequence:
            print(f"\nEncoding seed sequence as CEM starting point...")
            init_mu = wrapper.encode_sequence(
                seed_sequence,
                max_len=args.seq_len,
                utr5_size=utr5_size,
                cds_size=cds_size,
            )
            print(f"  Seed embedding norm: {init_mu.norm():.4f}")

        if args.optimizer in ("diffusion", "combined"):
            diff_steps = args.diffusion_steps or args.n_iters
            label = "COMBINED (diffusion → inversion → direct)" if args.optimizer == "combined" else "DIFFUSION"
            print(f"\nOptimizer: {label} | batch={args.pop_size} | "
                  f"steps={diff_steps} | guidance={args.guidance_scale}")
        else:
            print(f"\nOptimizer: {args.optimizer.upper()} CEM | "
                  f"pop={args.pop_size} | elite={args.elite_frac} | iters={args.n_iters}")

        if args.optimizer in ("diffusion", "combined"):
            optimizer = DiffusionOptimizer(
                dim=latent_dim,
                batch_size=args.pop_size,
                n_steps=args.diffusion_steps or args.n_iters,
                guidance_scale=args.guidance_scale,
                noise_schedule=args.noise_schedule,
                init_mu=init_mu,
                n_repeats=args.n_repeats,
                clip_grad_norm=args.clip_grad_norm,
                proximity_weight=args.proximity_weight,
                max_radius=args.max_radius,
                device=args.device,
            )
        elif args.optimizer == "flow":
            optimizer = FlowCEM(
                dim=latent_dim,
                pop_size=args.pop_size,
                elite_frac=args.elite_frac,
                n_iters=args.n_iters,
                init_mu=init_mu,
                schedule=args.schedule,
                momentum=args.momentum,
                device=args.device,
            )
        else:
            optimizer = VanillaCEM(
                dim=latent_dim,
                pop_size=args.pop_size,
                elite_frac=args.elite_frac,
                n_iters=args.n_iters,
                init_mu=init_mu,
                device=args.device,
            )

        print("\n" + "=" * 60)
        print("Starting optimization...")
        print("=" * 60 + "\n")

        result = optimizer.optimize(objective, verbose=True)

        print(f"\nBest objective score: {result.best_score:.4f}")

        # ── Gradient-based inversion ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("Inverting best latent vector to mRNA sequence...")
        print("=" * 60 + "\n")

        decoder = GradientDecoder(
            wrapper=wrapper,
            seq_len=args.seq_len,
            n_steps=args.inversion_steps,
            lr=args.inversion_lr,
            utr5_size=utr5_size,
            cds_size=cds_size,
            utr3_size=utr3_len,
            cds_seq=cds_seq if cds_seq else None,
            target_col=target_cell_idx,
            off_target_cols=off_target_idxs,
            obj_weight=args.obj_weight,
            lam=args.lam,
            obj_mode=obj_mode,
            device=args.device,
        )

        inv_result = decoder.invert(result.best_z, verbose=True)

        # ── Combined: direct refinement after inversion ──────────
        if args.optimizer == "combined":
            print("\n" + "=" * 60)
            print("Stage 3: Direct codon refinement...")
            print("=" * 60 + "\n")

            # Use original CDS (protein preservation) + inverted UTRs
            inv_seq = inv_result.sequence
            inv_utr5 = inv_seq[:utr5_size] if utr5_size > 0 else ""
            cds_for_direct = cds_seq if cds_seq else ""
            inv_utr3 = inv_seq[utr5_size + cds_size:utr5_size + cds_size + utr3_len] if utr3_len > 0 else ""

            # Build seed for direct optimizer from inverted UTRs + original CDS
            direct_seed = inv_utr5 + cds_for_direct + inv_utr3
            if direct_seed:
                print(f"  Direct seed: {len(direct_seed)} nt "
                      f"(UTR5={len(inv_utr5)}, CDS={len(cds_for_direct)}, UTR3={len(inv_utr3)})")

            direct_opt = DirectOptimizer(
                wrapper=wrapper,
                seq_len=args.seq_len,
                utr5_size=utr5_size,
                cds_size=cds_size,
                utr3_size=utr3_len,
                cds_seq=cds_for_direct if cds_for_direct else None,
                utr5_seq=inv_utr5 if inv_utr5 else None,
                utr3_seq=inv_utr3 if inv_utr3 else None,
                target_col=target_cell_idx,
                off_target_cols=off_target_idxs,
                lam=args.lam,
                obj_mode=obj_mode,
                n_steps=args.inversion_steps,
                n_repeats=args.n_repeats,
                top_k=args.top_k,
                lr=args.inversion_lr,
                device=args.device,
            )
            direct_result = direct_opt.optimize(verbose=True)

            # Replace inversion result with direct result
            inv_result = InversionResult(
                sequence=direct_result.sequence,
                logits=direct_result.logits,
                final_loss=-direct_result.best_score,
                latent_distance=0.0,
                loss_history=[],
            )

            # Update result to reflect the combined score
            class _CombinedResultCompat:
                def __init__(self, latent_result, dr):
                    self.best_score = dr.best_score
                    self.best_z = dr.best_z
                    self.history = latent_result.history + dr.history
                    self.candidates = dr.candidates
            result = _CombinedResultCompat(result, direct_result)

            print(f"\nCombined best specificity: {direct_result.best_score:.4f}")

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequence length: {len(inv_result.sequence)}")
    print(f"Latent distance: {inv_result.latent_distance:.4f}")
    print(f"Objective score: {result.best_score:.4f}")

    seq = inv_result.sequence

    # Verify protein preservation and fix CDS if needed
    cds_seq_clean = ""
    if cds_seq:
        cds_seq_clean = clean_seq(cds_seq).replace("T", "U")
        original_protein = translate(cds_seq_clean)
        optimized_cds = seq[utr5_size:utr5_size + len(cds_seq_clean)]
        optimized_protein = translate(optimized_cds)

        print(f"\n--- Protein Verification ---")
        print(f"Original: {original_protein[:70]}...")
        print(f"Output:   {optimized_protein[:70]}...")

        if original_protein == optimized_protein:
            print(f"Protein preserved: YES ({len(original_protein)} aa)")
            n_codons = len(cds_seq_clean) // 3
            changed = sum(
                cds_seq_clean[i*3:(i+1)*3] != optimized_cds[i*3:(i+1)*3]
                for i in range(n_codons)
            )
            print(f"Synonymous codon changes: {changed}/{n_codons}")
        else:
            mismatches = [
                (i, a, b) for i, (a, b)
                in enumerate(zip(original_protein, optimized_protein)) if a != b
            ]
            print(f"WARNING: Protein CHANGED — {len(mismatches)}/{len(original_protein)} "
                  f"amino acids differ!")
            for pos, orig, opt in mismatches[:10]:
                print(f"  Position {pos}: {orig} -> {opt} "
                      f"(codon: {cds_seq_clean[pos*3:(pos+1)*3]} -> "
                      f"{optimized_cds[pos*3:(pos+1)*3]})")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")

            print(f"\nForcing original CDS back into output sequence...")
            seq = seq[:utr5_size] + cds_seq_clean + seq[utr5_size + len(cds_seq_clean):]
            inv_result.sequence = seq

            verify_protein = translate(seq[utr5_size:utr5_size + len(cds_seq_clean)])
            assert verify_protein == original_protein, "CDS restoration failed!"
            print(f"CDS restored and verified.")

    # ── Region breakdown ──────────────────────────────────────────────
    seq = inv_result.sequence
    cds_len = len(cds_seq_clean) if cds_seq_clean else cds_size
    opt_utr5 = seq[:utr5_size] if utr5_size > 0 else ""
    opt_cds = seq[utr5_size:utr5_size + cds_len] if cds_len > 0 else ""
    opt_utr3_plus_pad = seq[utr5_size + cds_len:] if (utr5_size + cds_len) < len(seq) else ""

    # utr3_len already defined above for GradientDecoder

    def _nuc_comp(s):
        """Return composition string for a region."""
        n = len(s)
        if n == 0:
            return "(empty)"
        c = {nuc: s.count(nuc) for nuc in "AUCG"}
        gc = (c["G"] + c["C"]) / n
        return (f"A={c['A']/n:.0%} U={c['U']/n:.0%} "
                f"C={c['C']/n:.0%} G={c['G']/n:.0%} | GC={gc:.0%}")

    def _max_run(s):
        """Return max homopolymer run per nucleotide."""
        runs = {}
        for nuc in "AUCG":
            mr, cur = 0, 0
            for c in s:
                if c == nuc:
                    cur += 1
                    mr = max(mr, cur)
                else:
                    cur = 0
            runs[nuc] = mr
        return " ".join(f"{n}={r}" for n, r in runs.items())

    print(f"\n--- Sequence Regions ---")

    # 5'UTR
    if utr5_size > 0:
        print(f"\n5'UTR ({utr5_size} nt):")
        if utr5_seq:
            print(f"  Original: {utr5_seq[:80]}{'...' if len(utr5_seq) > 80 else ''}")
        print(f"  Optimized: {opt_utr5[:80]}{'...' if len(opt_utr5) > 80 else ''}")
        if utr5_seq:
            identity = sum(a == b for a, b in zip(utr5_seq.upper().replace('T','U'), opt_utr5)) / max(utr5_size, 1)
            print(f"  Identity: {identity:.0%} | {_nuc_comp(opt_utr5)}")
        else:
            print(f"  Composition: {_nuc_comp(opt_utr5)}")

    # CDS
    if cds_len > 0:
        print(f"\nCDS ({cds_len} nt, {cds_len//3} codons):")
        print(f"  Optimized: {opt_cds[:80]}{'...' if len(opt_cds) > 80 else ''}")
        print(f"  Composition: {_nuc_comp(opt_cds)}")

    # 3'UTR
    if utr3_len > 0:
        opt_utr3 = opt_utr3_plus_pad[:utr3_len]
        print(f"\n3'UTR ({utr3_len} nt):")
        print(f"  Original:  {utr3_seq[:80]}{'...' if len(utr3_seq) > 80 else ''}")
        print(f"  Optimized: {opt_utr3[:80]}{'...' if len(opt_utr3) > 80 else ''}")
        utr3_orig_clean = utr3_seq.upper().replace('T', 'U')
        identity = sum(a == b for a, b in zip(utr3_orig_clean, opt_utr3)) / max(utr3_len, 1)
        print(f"  Identity: {identity:.0%} | {_nuc_comp(opt_utr3)}")

    # Biological sequence = 5'UTR + CDS + 3'UTR (no padding)
    bio_len = utr5_size + cds_len + utr3_len
    bio_seq = seq[:bio_len] if bio_len > 0 else seq
    pad_len = len(seq) - bio_len if bio_len > 0 else 0
    if pad_len > 0:
        print(f"\n(Model padding: {pad_len} nt — not included in output)")

    # Overall stats on biological sequence only
    n_total = len(bio_seq)
    counts = {nuc: bio_seq.count(nuc) for nuc in "AUCG"}
    gc_frac = (counts["G"] + counts["C"]) / n_total if n_total > 0 else 0
    u_frac = counts["U"] / n_total if n_total > 0 else 0
    print(f"\nOverall ({n_total} nt): {_nuc_comp(bio_seq)}")
    print(f"Max homopolymer runs: {_max_run(bio_seq)}")

    # ── RiboNN Predictions ─────────────────────────────────────────────
    print(f"\n--- RiboNN Predictions ---")
    all_cell_idxs = [target_cell_idx] + off_target_idxs
    all_cell_names = [target_name] + off_target_names

    # Predict for optimized sequence
    opt_te = wrapper.predict_sequence(
        inv_result.sequence, max_len=args.seq_len,
        utr5_size=utr5_size, cds_size=cds_len,
    )

    # Predict for original sequence (if available)
    orig_te = None
    if seed_sequence:
        orig_te = wrapper.predict_sequence(
            seed_sequence, max_len=args.seq_len,
            utr5_size=utr5_size, cds_size=cds_size,
        )

    print(f"\n  {'Cell Type':<16} {'Original':>10} {'Optimized':>10} {'Delta':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names):
        opt_val = opt_te[ct_idx].item()
        label = "TARGET" if ct_idx == target_cell_idx else "off-tgt"
        if orig_te is not None:
            orig_val = orig_te[ct_idx].item()
            delta = opt_val - orig_val
            print(f"  {ct_name:<16} {orig_val:>10.4f} {opt_val:>10.4f} {delta:>+10.4f}  [{label}]")
        else:
            print(f"  {ct_name:<16} {'N/A':>10} {opt_val:>10.4f}  [{label}]")

    # Specificity score from predictions
    opt_target_te = opt_te[target_cell_idx].item()
    opt_offtarget_te = sum(opt_te[i].item() for i in off_target_idxs) / max(len(off_target_idxs), 1)
    opt_specificity = opt_target_te - args.lam * opt_offtarget_te
    print(f"\n  Optimized specificity: {opt_specificity:.4f} "
          f"(target={opt_target_te:.4f} - {args.lam}*off_target_avg={opt_offtarget_te:.4f})")

    if orig_te is not None:
        orig_target_te = orig_te[target_cell_idx].item()
        orig_offtarget_te = sum(orig_te[i].item() for i in off_target_idxs) / max(len(off_target_idxs), 1)
        orig_specificity = orig_target_te - args.lam * orig_offtarget_te
        print(f"  Original specificity:  {orig_specificity:.4f} "
              f"(target={orig_target_te:.4f} - {args.lam}*off_target_avg={orig_offtarget_te:.4f})")
        print(f"  Specificity delta:     {opt_specificity - orig_specificity:+.4f}")

    # Custom predictor evaluation (if used)
    if args.objective == "predictor" and predictor is not None:
        z_decoded = wrapper.encode_sequence(inv_result.sequence, max_len=args.seq_len)
        print(f"\n--- Custom Predictor Predictions ---")
        with torch.no_grad():
            for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names):
                eff = predictor(
                    z_decoded.unsqueeze(0).to(args.device),
                    torch.tensor([ct_idx], device=args.device),
                ).item()
                label = "TARGET" if ct_idx == target_cell_idx else "off-tgt"
                print(f"  {ct_name:<16} {eff:>10.4f}  [{label}]")

    # ── Save outputs ─────────────────────────────────────────────────────
    seq = inv_result.sequence

    def _write_fasta_entry(f, header, sequence):
        f.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")

    # Save full biological sequence (no padding)
    seq_path = output_dir / "optimized_sequence.fasta"
    with open(seq_path, "w") as f:
        _write_fasta_entry(f, f"optimized_mRNA target={target_name} "
                           f"score={result.best_score:.4f} len={len(bio_seq)}", bio_seq)
    print(f"\nSaved sequence to {seq_path}")

    # Save regions as separate FASTA entries
    regions_path = output_dir / "optimized_regions.fasta"
    with open(regions_path, "w") as f:
        if utr5_size > 0:
            _write_fasta_entry(f, f"5UTR len={utr5_size}", opt_utr5)
        if cds_len > 0:
            _write_fasta_entry(f, f"CDS len={cds_len} codons={cds_len//3}", opt_cds)
        if utr3_len > 0:
            _write_fasta_entry(f, f"3UTR len={utr3_len}", opt_utr3_plus_pad[:utr3_len])
    print(f"Saved regions to {regions_path}")

    # Save top-K candidates (if available)
    candidates = getattr(result, "candidates", [])
    if len(candidates) > 1:
        cands_path = output_dir / "optimized_candidates.fasta"
        with open(cands_path, "w") as f:
            for rank, cand in enumerate(candidates, 1):
                cand_bio = cand.sequence[:bio_len] if bio_len > 0 else cand.sequence
                _write_fasta_entry(
                    f,
                    f"candidate_{rank} target={target_name} score={cand.score:.4f} "
                    f"len={len(cand_bio)}",
                    cand_bio,
                )
        print(f"Saved {len(candidates)} candidates to {cands_path}")

        # Print candidate summary table
        print(f"\n--- Top {len(candidates)} Candidates ---")
        print(f"  {'Rank':<6} {'Score':>10}  Predictions")
        print(f"  {'-'*6} {'-'*10}  {'-'*40}")
        for rank, cand in enumerate(candidates, 1):
            cand_te = wrapper.predict_sequence(
                cand.sequence, max_len=args.seq_len,
                utr5_size=utr5_size, cds_size=cds_len,
            )
            te_str = "  ".join(
                f"{ct_name}={cand_te[ct_idx].item():.4f}"
                for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names)
            )
            marker = " *" if rank == 1 else ""
            print(f"  {rank:<6} {cand.score:>10.4f}  {te_str}{marker}")

    # Save optimization results
    results_dict = {
        "best_score": result.best_score,
        "sequence_length": len(inv_result.sequence),
        "latent_distance": inv_result.latent_distance,
        "optimizer": args.optimizer,
        "schedule": args.schedule if args.optimizer == "flow" else None,
        "pop_size": args.pop_size,
        "n_iters": args.n_iters,
        "target_cell": target_name,
        "target_cell_index": target_cell_idx,
        "off_target_cells": off_target_names,
        "off_target_cell_indices": off_target_idxs,
        "lam": args.lam,
        "utr5_size": utr5_size,
        "cds_size": cds_size,
        "input_sequence_length": len(seed_sequence) if seed_sequence else 0,
        "gc_content": gc_frac,
        "u_content": u_frac,
        "composition": counts,
        "obj_weight": args.obj_weight,
        "obj_mode": obj_mode,
        "predictions": {
            "optimized": {ct_name: opt_te[ct_idx].item()
                          for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names)},
            "optimized_specificity": opt_specificity,
        },
    }
    if orig_te is not None:
        results_dict["predictions"]["original"] = {
            ct_name: orig_te[ct_idx].item()
            for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names)
        }
        results_dict["predictions"]["original_specificity"] = orig_specificity

    # Add candidates to results
    if len(candidates) > 1:
        cand_list = []
        for rank, cand in enumerate(candidates, 1):
            cand_bio = cand.sequence[:bio_len] if bio_len > 0 else cand.sequence
            cand_te = wrapper.predict_sequence(
                cand.sequence, max_len=args.seq_len,
                utr5_size=utr5_size, cds_size=cds_len,
            )
            cand_list.append({
                "rank": rank,
                "score": cand.score,
                "sequence": cand_bio,
                "predictions": {
                    ct_name: cand_te[ct_idx].item()
                    for ct_idx, ct_name in zip(all_cell_idxs, all_cell_names)
                },
            })
        results_dict["candidates"] = cand_list

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save latent vector(s)
    save_dict = {
        "best_z": result.best_z.cpu(),
        "history": result.history,
    }
    if len(candidates) > 1:
        save_dict["top_k_z"] = [c.z.cpu() for c in candidates]
        save_dict["top_k_scores"] = [c.score for c in candidates]
    torch.save(save_dict, output_dir / "optimization.pt")

    # Plot
    plot_optimization(result, str(output_dir / "optimization_plot.png"))

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
