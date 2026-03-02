"""Extract RiboNN latent embeddings from mRNA sequences.

Usage:
    python scripts/extract_embeddings.py --config configs/optimize.yaml
    python scripts/extract_embeddings.py --checkpoint path/to/ribonn.ckpt --input data/sequences.tsv --output data/embeddings.pt

Outputs a .pt file containing:
    - embeddings: (N, latent_dim) tensor
    - predictions: (N, num_targets) tensor
    - sequences: list of raw sequence strings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnaflow.data.encoding import one_hot_encode_ribonn
from rnaflow.embeddings.ribonn_wrapper import RiboNNWrapper


def load_sequences_from_tsv(path: str) -> list[dict]:
    """Load sequences from a RiboNN-format TSV file.

    Supports both formats:
        - tx_id, utr5_sequence, cds_sequence, utr3_sequence
        - tx_id, tx_sequence, utr5_size, cds_size
    """
    df = pd.read_csv(path, sep="\t")

    sequences = []
    if "tx_sequence" in df.columns:
        for _, row in df.iterrows():
            sequences.append({
                "tx_id": row["tx_id"],
                "sequence": row["tx_sequence"],
            })
    elif "utr5_sequence" in df.columns:
        for _, row in df.iterrows():
            seq = row["utr5_sequence"] + row["cds_sequence"] + row["utr3_sequence"]
            sequences.append({
                "tx_id": row["tx_id"],
                "sequence": seq,
            })
    else:
        raise ValueError(f"Unrecognized TSV format. Columns: {list(df.columns)}")

    return sequences


def extract_embeddings(
    wrapper: RiboNNWrapper,
    sequences: list[dict],
    max_seq_len: int = 12288,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings and predictions for a list of sequences.

    Returns:
        embeddings: (N, latent_dim) tensor
        predictions: (N, num_targets) tensor
    """
    all_embeddings = []
    all_predictions = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch = sequences[i : i + batch_size]
        tensors = torch.stack([
            one_hot_encode_ribonn(s["sequence"], max_seq_len) for s in batch
        ])

        z, te = wrapper.encode_and_predict(tensors)
        all_embeddings.append(z.cpu())
        all_predictions.append(te.cpu())

    return torch.cat(all_embeddings), torch.cat(all_predictions)


def main():
    parser = argparse.ArgumentParser(description="Extract RiboNN embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RiboNN .ckpt")
    parser.add_argument("--input", type=str, required=True, help="Path to input TSV")
    parser.add_argument("--output", type=str, default="data/embeddings.pt", help="Output .pt path")
    parser.add_argument("--max-seq-len", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading RiboNN from {args.checkpoint}...")
    wrapper = RiboNNWrapper.from_checkpoint(args.checkpoint, device=args.device)
    print(f"  Latent dim: {wrapper.latent_dim}, Targets: {wrapper.num_targets}")

    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences_from_tsv(args.input)
    print(f"  Loaded {len(sequences)} sequences")

    embeddings, predictions = extract_embeddings(
        wrapper, sequences, max_seq_len=args.max_seq_len, batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings,
        "predictions": predictions,
        "sequences": [s["sequence"] for s in sequences],
        "tx_ids": [s["tx_id"] for s in sequences],
        "latent_dim": wrapper.latent_dim,
        "num_targets": wrapper.num_targets,
    }, output_path)

    print(f"Saved embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")


if __name__ == "__main__":
    main()
