"""Train a cell-type-conditioned translation efficiency predictor on RiboNN embeddings.

Usage:
    python scripts/train_predictor.py --config configs/predictor.yaml

Or with explicit arguments:
    python scripts/train_predictor.py \
        --embeddings data/embeddings.pt \
        --n-cell-types 5 \
        --epochs 50 \
        --output checkpoints/predictor.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnaflow.models.predictor import TranslationPredictor
from rnaflow.utils.config import load_config


def create_dataset_from_embeddings(
    embeddings_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load precomputed embeddings and create (z, cell_type, efficiency) dataset.

    The embeddings .pt file should contain:
        - embeddings: (N, latent_dim)
        - predictions: (N, num_targets) — used as efficiency labels

    We expand each sequence into N_targets rows, one per cell type.

    Returns:
        z: (N * num_targets, latent_dim)
        cell_types: (N * num_targets,)
        efficiencies: (N * num_targets,)
    """
    data = torch.load(embeddings_path, weights_only=False)
    emb = data["embeddings"]       # (N, latent_dim)
    preds = data["predictions"]    # (N, num_targets)

    N, num_targets = preds.shape
    latent_dim = emb.shape[1]

    # Expand: each (sequence, cell_type) pair becomes a training example
    z = emb.unsqueeze(1).expand(N, num_targets, latent_dim).reshape(-1, latent_dim)
    cell_types = torch.arange(num_targets).unsqueeze(0).expand(N, num_targets).reshape(-1)
    efficiencies = preds.reshape(-1)

    return z, cell_types, efficiencies


def create_synthetic_dataset(
    n_sequences: int = 5000,
    latent_dim: int = 64,
    n_cell_types: int = 5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a synthetic dataset for testing the predictor pipeline.

    Generates random embeddings with cell-type-dependent efficiency patterns.
    """
    rng = np.random.RandomState(seed)

    # Random embeddings
    z = torch.randn(n_sequences, latent_dim)

    # Cell-type-specific "receptive" directions in latent space
    directions = torch.randn(n_cell_types, latent_dim)
    directions = directions / directions.norm(dim=1, keepdim=True)

    all_z, all_ct, all_eff = [], [], []
    for ct in range(n_cell_types):
        all_z.append(z)
        all_ct.append(torch.full((n_sequences,), ct, dtype=torch.long))
        # Efficiency = sigmoid(dot product with cell-type direction + noise)
        scores = (z @ directions[ct]) + torch.randn(n_sequences) * 0.3
        eff = torch.sigmoid(scores)
        all_eff.append(eff)

    return torch.cat(all_z), torch.cat(all_ct), torch.cat(all_eff)


def train_predictor(
    model: TranslationPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict]:
    """Train the predictor model.

    Returns:
        List of dicts with train_loss and val_loss per epoch.
    """
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for z, ct, eff in train_loader:
            z, ct, eff = z.to(device), ct.to(device), eff.to(device)
            pred = model(z, ct)
            loss = criterion(pred, eff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for z, ct, eff in val_loader:
                z, ct, eff = z.to(device), ct.to(device), eff.to(device)
                pred = model(z, ct)
                loss = criterion(pred, eff)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        history.append({"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train cell-type predictor")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--embeddings", type=str, default=None, help="Embeddings .pt path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-cell-types", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="checkpoints/predictor.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)
        for key, val in vars(cfg).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, val)

    # Load or generate data
    if args.synthetic or args.embeddings is None:
        print("Using synthetic dataset for training...")
        z, cell_types, efficiencies = create_synthetic_dataset(
            latent_dim=args.latent_dim, n_cell_types=args.n_cell_types,
        )
    else:
        print(f"Loading embeddings from {args.embeddings}...")
        z, cell_types, efficiencies = create_dataset_from_embeddings(args.embeddings)

    print(f"Dataset: {z.shape[0]} samples, latent_dim={z.shape[1]}")

    # Create data loaders
    dataset = TensorDataset(z, cell_types, efficiencies)
    n_val = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Create model
    model = TranslationPredictor(
        latent_dim=args.latent_dim,
        n_cell_types=args.n_cell_types,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("Training predictor...")
    history = train_predictor(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=args.device,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved predictor to {output_path}")
    print(f"Final val_loss: {history[-1]['val_loss']:.6f}")


if __name__ == "__main__":
    main()
