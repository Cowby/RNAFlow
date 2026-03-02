# RNAFlow

**Flow-inspired latent space optimization of mRNA sequences for cell-type specificity.**

RNAFlow uses [RiboNN](https://github.com/Sanofi-Public/RiboNN) pretrained embeddings and a flow-inspired Cross-Entropy Method (FlowCEM) to optimize mRNA sequences in a 64-dimensional latent space, maximizing translation efficiency in a target cell type while suppressing off-target expression.

## Overview

```
Your mRNA (5'UTR + CDS + 3'UTR)
         │
         ▼
    RiboNN encoder ──> 64-dim latent z  (= CEM starting point)
                             │
                        FlowCEM optimizer
                        (CEM + flow schedule)
                             │
                        optimized z*
                             │
                    gradient-based inversion
                    (with codon labels from your CDS)
                             │
                    optimized mRNA sequence
```

**Key idea:** Instead of optimizing over discrete nucleotide sequences directly, RNAFlow:
1. Embeds your input mRNA into RiboNN's 64-dim bottleneck space
2. Runs CEM optimization with flow-inspired distribution interpolation to find a latent vector z* that maximizes target-cell specificity
3. Inverts z* back to a nucleotide sequence via gradient descent, using your CDS structure for proper codon labeling

## Installation

### 1. Clone and install RNAFlow

```bash
git clone <repo-url> RNAFlow
cd RNAFlow
pip install -e ".[dev]"
```

### 2. Set up RiboNN

```bash
# Clone RiboNN into external/
bash setup_ribonn.sh

# Or pip-based install (recommended if conda solver fails):
bash setup_ribonn_pip.sh
```

### 3. Download pretrained weights

Download the pretrained RiboNN models from [Zenodo](https://zenodo.org/records/17258709):

```bash
mkdir -p checkpoints
cd checkpoints
# Download and extract human and/or mouse models
# Each model directory contains a state_dict.pth file (~997KB)
```

The human model predicts translation efficiency across **78 cell types** using **5 input channels** (4 nucleotide one-hot + 1 codon label).

## Typical Workflow

You have an mRNA of interest (e.g., a therapeutic mRNA) with known 5'UTR, CDS, and 3'UTR. You want to find sequence variants with high translation in HeLa cells but low translation in off-target cell types.

### 1. List available cell types

```bash
python scripts/optimize.py --list-cell-types
```

### 2. Optimize with your mRNA as starting point

Pretrained weights are **auto-detected** from `checkpoints/human/` — no need to specify a path. Just provide your sequences and cell types:

```bash
python scripts/optimize.py \
  --utr5 "ACUGGCUAGCUA..." \
  --cds  "AUGAAAGGGCCC...UAA" \
  --utr3 "AAUAAACCCUUU..." \
  --target-cell HeLa \
  --off-target-cells HEK293T K562 A549 MCF7 \
  --n-iters 200 \
  --output-dir results/my_optimization
```

The script will:
- Auto-detect the first available pretrained model from `checkpoints/human/`
- Concatenate your UTR/CDS sequences into a full mRNA
- Encode it through RiboNN to get an initial latent embedding (CEM starting point)
- Build proper codon labels from your CDS during gradient inversion
- Auto-detect `seq_len` from the model (typically 13,312 nt)

**About the pretrained models:** The Zenodo download contains 91 independently trained model runs (cross-validation folds). Auto-detection picks the first one alphabetically. To use a specific run: `--ribonn-state-dict checkpoints/human/<run_id>/state_dict.pth`

### 3. Or use a FASTA file

If you have the full mRNA in a FASTA file with known region boundaries:

```bash
python scripts/optimize.py \
  --seed-fasta my_mrna.fasta \
  --utr5-len 150 \
  --cds-len 900 \
  --target-cell HeLa \
  --off-target-cells HEK293T K562 \
  --output-dir results/from_fasta
```

### 4. Use a mouse model

```bash
python scripts/optimize.py \
  --organism mouse \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --target-cell 0 \
  --off-target-cells 1 2 3 \
  --output-dir results/mouse_run
```

### 5. Quick test without pretrained weights

```bash
python scripts/optimize.py \
  --use-mock \
  --target-cell 0 \
  --off-target-cells 1 2 3 4 \
  --n-iters 50 \
  --seq-len 512 \
  --output-dir results/mock_test
```

## Parameter Guide

### Model Selection

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--ribonn-state-dict PATH` | auto-detected | Path to a `state_dict.pth` file. **If omitted**, the script scans `checkpoints/<organism>/` and picks the first available model. You only need this if you want a specific model run. |
| `--organism` | `human` | Which organism's pretrained model to use (`human` or `mouse`). Only matters for auto-detection. |
| `--checkpoints-dir PATH` | `checkpoints/` | Root directory containing `human/` and `mouse/` subdirectories with pretrained weights. |
| `--ribonn-checkpoint PATH` | — | Alternative: path to a PyTorch Lightning `.ckpt` file (if you trained RiboNN yourself). |
| `--predictor-checkpoint PATH` | — | Path to a trained custom predictor `.pt` file (only for `--objective predictor`). |
| `--use-mock` | — | Use a random-weight MockRiboNN for testing. No pretrained weights needed. |

### Sequence Input

You provide your mRNA structure so that RNAFlow can (a) start optimization near your sequence and (b) build correct codon labels for gradient inversion.

| Parameter | Description |
|-----------|-------------|
| `--utr5 SEQ` | 5'UTR nucleotide sequence. Used to compute CDS offset for codon labeling. |
| `--cds SEQ` | CDS nucleotide sequence including start AUG and stop codon. Codon labels are derived from this: every 3rd position starting from position 0 of the CDS is marked. |
| `--utr3 SEQ` | 3'UTR nucleotide sequence. |
| `--seed-fasta PATH` | Alternative: load full mRNA from a FASTA file. Requires `--utr5-len` and `--cds-len` to annotate structure. |
| `--utr5-len N` | 5'UTR length in nt (only with `--seed-fasta`). |
| `--cds-len N` | CDS length in nt (only with `--seed-fasta`). |

**Why this matters:** RiboNN was trained with a 5th input channel marking codon start positions in the CDS. If you omit the CDS, the codon channel is all zeros during inversion, which degrades embedding quality because the model has never seen input like that during training. Always provide your CDS if you have it.

**What happens without sequence input:** CEM starts from a random point in latent space (zero mean, unit variance). This still works but explores more broadly and takes longer to converge.

### Objective

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--objective` | `ribonn` | **`ribonn`**: Uses RiboNN's own output head to predict TE for each cell type. Fast, no extra training needed. Works directly on 64-dim latent vectors via the head tail (BatchNorm -> Dropout -> Linear). **`predictor`**: Uses a separately trained MLP that takes (z, cell_type) as input. More flexible but requires training first. |
| `--target-cell` | `HeLa` | The cell type to maximize translation efficiency in. Accepts names (`HeLa`, `HEK293T`, `neurons`) or indices (`22`, `18`, `73`). Case-insensitive. |
| `--off-target-cells` | `HEK293T K562 A549 MCF7` | Cell types to suppress. The objective penalizes high TE in these. More off-targets = stronger constraint on the optimization. |
| `--lam` | `1.0` | Controls the strength of off-target suppression. The objective is `TE[target] - lam * mean(TE[off_targets])`. **lam=0**: Maximize target only, ignore off-targets. **lam=1**: Equal weight to target and off-target suppression. **lam=5-10**: Aggressively suppress off-targets, even at some cost to target TE. Start with 1.0 and increase if off-target suppression is insufficient. |

### CEM Optimizer

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--optimizer` | `flow` | **`flow`** (FlowCEM): Interpolates the sampling distribution from a broad prior toward the elite-fitted distribution using a time schedule. Better exploration-exploitation balance. **`vanilla`**: Standard CEM that immediately fits to elites each iteration. Faster convergence but more prone to getting stuck in local optima. Use `vanilla` as a baseline for comparison. |
| `--pop-size` | `512` | Number of candidate latent vectors sampled per iteration. **Larger (1024+)**: Better coverage of latent space, less likely to miss good solutions. Costs linearly more compute per iteration. **Smaller (128)**: Faster iterations but noisier elite estimates. Good for quick exploration or when compute is limited. For 64-dim latent space, 256-512 is a good balance. |
| `--elite-frac` | `0.05` | Fraction of top-scoring samples used to update the distribution. With pop_size=512 and elite_frac=0.05, the top 25 samples are kept. **Smaller (0.01-0.02)**: More selective, faster convergence, but risks losing diversity. **Larger (0.1-0.2)**: More conservative, maintains diversity, slower convergence. If optimization plateaus early, try increasing this. |
| `--n-iters` | `200` | Number of CEM iterations. Each iteration samples, evaluates, and updates. **50-100**: Quick exploration, good for testing. **200-300**: Standard runs, usually sufficient for convergence. **500+**: Diminishing returns unless the objective landscape is very rugged. Watch the optimization plot: if the score plateaus, more iterations won't help. |
| `--schedule` | `cosine` | Controls how quickly FlowCEM transitions from exploration to exploitation. **`cosine`** (recommended): Slow start, fast middle, slow end. Gives time to explore early, then converges smoothly. **`linear`**: Uniform transition. Simple and predictable. **`quadratic`**: Spends more time exploring (t grows slowly early). Good if your starting point is far from the optimum. **`sqrt`**: Exploits quickly (t grows fast early). Good if your starting point is already good (e.g., seeded from a known sequence). |
| `--momentum` | `0.0` | Exponential moving average smoothing for elite mean/sigma updates (0-1). **0.0**: No smoothing, each iteration fully replaces the elite statistics. **0.1-0.3**: Mild smoothing, reduces noise from small elite sets. Useful when pop_size is small or objective is noisy. **0.5+**: Heavy smoothing, very slow adaptation. Rarely needed. |

### Gradient Inversion (z* -> sequence)

After CEM finds the optimal latent vector z*, gradient inversion recovers the corresponding nucleotide sequence by optimizing soft logits through the RiboNN encoder.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `--seq-len` | auto | Length of the output sequence. **Auto-detected from the model** (typically 13,312 for the pretrained human model). Only override if you need a specific length. Must match or exceed the model's expected input size. |
| `--inversion-steps` | `500` | Number of gradient descent steps. Each step updates the soft sequence logits to match z*. **100-200**: Fast but higher latent distance (the decoded sequence's embedding won't closely match z*). Good for quick exploration. **500**: Standard. Usually sufficient for reasonable reconstruction. **1000+**: Lower latent distance but slow, especially for long sequences (13k nt). Diminishing returns after the loss plateaus. |
| `--inversion-lr` | `0.05` | Learning rate for the Adam optimizer on sequence logits. **0.01**: Conservative, stable but slow convergence. **0.05** (default): Good balance. **0.1+**: Faster but may oscillate or diverge. If inversion loss is noisy, reduce this. |

### Composition Constraints

Without composition penalties, the gradient inversion produces **skewed nucleotide compositions** — the optimizer finds trivial solutions (e.g., poly-U or poly-A) that satisfy the latent objective but are biologically useless. These parameters enforce balanced composition:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nuc-targets A U C G` | `0.25 0.25 0.25 0.25` | Target fraction for each nucleotide. The loss penalizes `sum((actual_i - target_i)²)` across all four nucleotides, preventing any single base from dominating. Example: `--nuc-targets 0.20 0.20 0.30 0.30` for GC-rich mRNAs. |
| `--composition-weight` | `5.0` | Penalty strength for composition deviation. **0**: No composition constraint (produces degenerate sequences). **2.0**: Light guidance. **5.0** (default): Good balance — composition stays near targets while still allowing the optimizer room to match z*. **10.0+**: Very strict composition, may increase latent distance. |

**Why this matters:**
- **High U content** → TLR7/8 recognition → strong innate immune response (reactogenicity)
- **Low GC content** → poor mRNA stability, faster degradation by RNases
- **Extreme A-richness** → poly-A mimic → nonsense polyadenylation signals, ribosome stalling
- Modern mRNA therapeutics (e.g., BNT162b2) use N1-methylpseudouridine; for pseudouridine designs, you can shift targets: `--nuc-targets 0.20 0.30 0.25 0.25`

### Synonymous Codon Constraints

When you provide `--cds`, the CDS region is automatically constrained to **synonymous codon substitutions only**. This guarantees the encoded protein is preserved while allowing codon optimization for the target cell type. UTR regions are optimized freely. The output reports `Protein preserved: YES` to confirm.

**What is "latent distance"?** After inversion, the discrete sequence is re-encoded through RiboNN and compared to z*. The latent distance measures how well the inversion preserved the target embedding. Lower is better, but even distances of 20-40 can produce useful sequences because the objective operates in latent space, not sequence space.

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | `results` | Directory for all output files |
| `--device` | `cpu` | `cpu` or `cuda`. GPU dramatically speeds up both CEM (batched objective evaluation) and inversion (gradient computation through the CNN). |
| `--config` | — | Path to YAML config file (overrides defaults, CLI args take precedence) |
| `--list-cell-types` | — | Print all 78 cell type names with indices and exit |

## Architecture

### Project Structure

```
RNAFlow/
├── rnaflow/
│   ├── data/
│   │   ├── encoding.py          # One-hot encoding matching RiboNN's format
│   │   ├── cell_types.py        # 78 human cell type name <-> index mapping
│   │   └── synthetic.py         # Synthetic data for testing
│   ├── embeddings/
│   │   └── ribonn_wrapper.py    # RiboNN wrapper: embedding extraction + prediction
│   ├── inversion/
│   │   └── gradient_decoder.py  # Gradient-based latent -> sequence inversion
│   ├── models/
│   │   └── predictor.py         # Cell-type-conditioned predictor MLP
│   ├── optim/
│   │   ├── cem.py               # Vanilla CEM baseline
│   │   ├── flow_cem.py          # Flow-inspired CEM (core novelty)
│   │   └── objective.py         # Specificity objective functions
│   └── utils/
│       └── config.py            # YAML config loader
├── scripts/
│   ├── optimize.py              # Main optimization pipeline
│   ├── train_predictor.py       # Train custom cell-type predictor
│   └── extract_embeddings.py    # Batch-extract RiboNN embeddings
├── configs/
│   ├── optimize.yaml            # Optimization config
│   └── predictor.yaml           # Predictor training config
└── tests/                       # 33 unit tests
```

### How RiboNN Embeddings Work

RiboNN is a CNN that predicts translation efficiency (TE) from mRNA sequences across 78 human cell types. Its head contains a 64-dim bottleneck:

```
input (5, L) -> conv backbone (10 layers + MaxPool) -> head:
  [0] ReLU
  [1] Flatten
  [2] Dropout
  [3] Linear(filters*L', 64)    <-- bottleneck projection
  [4] ReLU                       <-- EMBEDDING EXTRACTED HERE (64-dim)
  [5] BatchNorm1d(64)
  [6] Dropout
  [7] Linear(64, 78)            <-- TE predictions per cell type
```

- **Embedding extraction:** A forward hook on layer [4] captures the 64-dim vector after the bottleneck activation.
- **Latent objective:** During CEM, latent vectors are passed through layers [5-7] only (BatchNorm -> Dropout -> Linear) to get TE predictions. This is fast because it skips the expensive conv backbone.
- **Gradient inversion:** Soft sequence logits are passed through the full model (conv backbone + layers [0-4]) with gradients enabled to optimize for the target z*.

### FlowCEM Algorithm

Standard CEM iteratively fits a Gaussian to elite samples. FlowCEM adds **time-dependent distribution interpolation**:

```
For iteration k = 0, ..., N-1:
    t_k = schedule(k / (N-1))                # 0 -> 1
    mu_t  = (1 - t_k) * mu_0  + t_k * mu_elite
    sig_t = (1 - t_k) * sig_0 + t_k * sig_elite
    samples ~ N(mu_t, diag(sig_t^2))
    scores = objective(samples)
    elites = top-k% by score
    mu_elite, sig_elite = fit(elites)
```

At t=0, sampling comes from the broad prior (wide exploration). At t=1, sampling comes from the elite-fitted distribution (focused exploitation). The schedule controls the transition speed.

### Codon Labels

RiboNN's 5th input channel marks the first nucleotide of each codon in the CDS. This is critical biological context: the model learned that codon position affects translation efficiency.

When you provide `--cds`, the gradient decoder builds this channel automatically:
- Positions `utr5_len, utr5_len+3, utr5_len+6, ...` up to `utr5_len + cds_len - 3` are set to 1.0
- All other positions are 0.0
- The stop codon (last 3 nt of CDS) is excluded

Without CDS information, the channel is all zeros. The model still runs but produces embeddings from a distribution it never saw during training.

## Examples

### Optimize a therapeutic mRNA for liver-specific expression

```bash
python scripts/optimize.py \
  --utr5 "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC" \
  --cds  "AUGGUUAGCAAAGGGGA...UGAUAA" \
  --utr3 "UGAUAAUAGGCUGGAGCCUCGGUGGC...AAAAAAAAAA" \
  --target-cell HepG2 \
  --off-target-cells HeLa HEK293T K562 A549 MCF7 neurons primary_macrophages \
  --lam 3.0 \
  --objective ribonn \
  --optimizer flow \
  --schedule cosine \
  --pop-size 512 \
  --n-iters 200 \
  --inversion-steps 500 \
  --output-dir results/liver_specific
```

### Compare time schedules

```bash
for schedule in linear cosine quadratic sqrt; do
  python scripts/optimize.py \
    --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
    --target-cell HeLa \
    --off-target-cells HEK293T K562 \
    --optimizer flow \
    --schedule $schedule \
    --n-iters 200 \
    --output-dir results/schedule_${schedule}
done
```

- **cosine**: Best general-purpose choice. Smooth exploration-to-exploitation transition.
- **quadratic**: Better when starting from scratch (no seed sequence). More exploration early.
- **sqrt**: Better when seeded from a known good sequence. Exploits quickly.
- **linear**: Simple baseline. Uniform transition.

### FlowCEM vs Vanilla CEM

```bash
# FlowCEM
python scripts/optimize.py \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --target-cell HeLa --off-target-cells HEK293T K562 \
  --optimizer flow --schedule cosine \
  --n-iters 200 --output-dir results/flow

# Vanilla CEM
python scripts/optimize.py \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --target-cell HeLa --off-target-cells HEK293T K562 \
  --optimizer vanilla \
  --n-iters 200 --output-dir results/vanilla
```

FlowCEM typically achieves higher final scores because the gradual transition avoids premature convergence to local optima.

### Aggressive off-target suppression

```bash
python scripts/optimize.py \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --target-cell neurons \
  --off-target-cells HeLa HEK293T K562 A549 MCF7 HepG2 HCT116 PANC1 \
  --lam 5.0 \
  --pop-size 1024 \
  --n-iters 300 \
  --output-dir results/neuron_strict
```

With `lam=5.0` and 8 off-target cell types, the optimizer will strongly favor sequences that are translationally silent in most cell types but active in neurons. This is a harder optimization problem, so larger pop_size and more iterations help.

### GPU acceleration

```bash
python scripts/optimize.py \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --target-cell HeLa --off-target-cells HEK293T K562 A549 MCF7 \
  --device cuda \
  --pop-size 1024 \
  --n-iters 200 \
  --inversion-steps 500 \
  --output-dir results/gpu_run
```

GPU speeds up both CEM (batched head-tail evaluation) and inversion (backprop through the CNN backbone).

### Custom predictor objective

Train a cell-type-conditioned predictor MLP, then use it:

```bash
# 1. Extract embeddings (auto-detects model from checkpoints/)
python scripts/extract_embeddings.py \
  --input data/sequences.tsv \
  --output data/embeddings.pt

# 2. Train predictor
python scripts/train_predictor.py --config configs/predictor.yaml

# 3. Optimize with predictor
python scripts/optimize.py \
  --predictor-checkpoint checkpoints/predictor.pt \
  --utr5 "..." --cds "AUG...UAA" --utr3 "..." \
  --objective predictor \
  --target-cell 0 \
  --off-target-cells 1 2 3 4 \
  --output-dir results/predictor_based
```

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `optimized_sequence.fasta` | Optimized mRNA in FASTA format (80-char wrapped) |
| `results.json` | Full metadata: scores, cell types, parameters, sequence structure |
| `optimization.pt` | PyTorch checkpoint: best latent vector z*, score history |
| `optimization_plot.png` | Three-panel plot: objective score, flow schedule (t), distribution width (sigma) |

Example `results.json`:
```json
{
  "best_score": 928.75,
  "sequence_length": 13312,
  "latent_distance": 20.76,
  "optimizer": "flow",
  "schedule": "cosine",
  "pop_size": 128,
  "n_iters": 50,
  "target_cell": "HeLa",
  "target_cell_index": 22,
  "off_target_cells": ["HEK293T", "K562", "A549", "MCF7"],
  "off_target_cell_indices": [18, 27, 3, 33],
  "lam": 1.0,
  "utr5_size": 103,
  "cds_size": 513,
  "input_sequence_length": 661
}
```

## Available Cell Types (Human)

78 cell types from the pretrained RiboNN model. Use `--list-cell-types` or refer to the table below. Names are case-insensitive on the command line.

| Index | Cell Type | Index | Cell Type |
|-------|-----------|-------|-----------|
| 0 | 108T | 40 | OSCC |
| 1 | 12T | 41 | PANC1 |
| 2 | A2780 | 42 | PATU-8902 |
| 3 | A549 | 43 | PC3 |
| 4 | BJ | 44 | PC9 |
| 5 | BRx-142 | 45 | Primary_CD4+_T-cells |
| 6 | C643 | 46 | Primary_human_bronchial_epithelial_cells |
| 7 | CRL-1634 | 47 | RD-CCL-136 |
| 8 | Calu-3 | 48 | RPE-1 |
| 9 | Cybrid_Cells | 49 | SH-SY5Y |
| 10 | H1-hESC | 50 | SUM159PT |
| 11 | H1933 | 51 | SW480TetOnAPC |
| 12 | H9-hESC | 52 | T47D |
| 13 | HAP-1 | 53 | THP-1 |
| 14 | HCC_tumor | 54 | U-251 |
| 15 | HCC_adjancent_normal | 55 | U-343 |
| 16 | HCT116 | 56 | U2392 |
| 17 | HEK293 | 57 | U2OS |
| 18 | HEK293T | 58 | Vero_6 |
| 19 | HMECs | 59 | WI38 |
| 20 | HSB2 | 60 | WM902B |
| 21 | HSPCs | 61 | WTC-11 |
| 22 | HeLa | 62 | ZR75-1 |
| 23 | HeLa_S3 | 63 | cardiac_fibroblasts |
| 24 | HepG2 | 64 | ccRCC |
| 25 | Huh-7.5 | 65 | early_neurons |
| 26 | Huh7 | 66 | fibroblast |
| 27 | K562 | 67 | hESC |
| 28 | Kidney_normal_tissue | 68 | human_brain_tumor |
| 29 | LCL | 69 | iPSC-differentiated_dopamine_neurons |
| 30 | LuCaP-PDX | 70 | megakaryocytes |
| 31 | MCF10A | 71 | muscle_tissue |
| 32 | MCF10A-ER-Src | 72 | neuronal_precursor_cells |
| 33 | MCF7 | 73 | neurons |
| 34 | MD55A3 | 74 | normal_brain_tissue |
| 35 | MDA-MB-231 | 75 | normal_prostate |
| 36 | MM1.S | 76 | primary_macrophages |
| 37 | MOLM-13 | 77 | skeletal_muscle |
| 38 | Molt-3 | | |
| 39 | Mutu | | |

## Testing

```bash
pytest tests/ -v                                    # all 33 tests
pytest tests/test_flow_cem.py -v                    # optimizer tests only
pytest tests/ --cov=rnaflow --cov-report=term-missing  # with coverage
```

## Configuration Files

### configs/optimize.yaml

```yaml
ribonn_checkpoint: null
predictor_checkpoint: null
use_mock: true

objective: ribonn
target_cell: HeLa
off_target_cells: [HEK293T, K562, A549, MCF7]
lam: 1.0

optimizer: flow
pop_size: 512
elite_frac: 0.05
n_iters: 200
schedule: cosine
momentum: 0.0

inversion_steps: 500
inversion_lr: 0.05

output_dir: results
device: cpu
```

### configs/predictor.yaml

```yaml
embeddings: data/embeddings.pt
latent_dim: 64
n_cell_types: 5
cell_embed_dim: 16
hidden_dims: [128, 64]
dropout: 0.2

epochs: 50
batch_size: 128
lr: 0.001

output: checkpoints/predictor.pt
device: cpu
```

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- NumPy, pandas, PyYAML, tqdm, matplotlib
- [RiboNN](https://github.com/Sanofi-Public/RiboNN) (cloned into `external/`)
