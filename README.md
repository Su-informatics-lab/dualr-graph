# Graph Transformer Autoencoder for ICI→AKI

**Target:** npj Digital Medicine
**Status:** Pilot implementation

## Idea

A graph autoencoder on a **feature-interaction graph** where nodes are clinical
features (demographics, NCI-CCI comorbidities, ICI class) and the AKI outcome
is just another node. Reconstructing the AKI node = predicting AKI risk.
Attention weights between nodes = learned feature-interaction heatmap.

Three edge construction methods:
- **Spearman/MI** — data-derived (standard)
- **LLM-prior** — LLM-estimated pairwise associations (DualR bridge,
  no patient data needed → portable)

## Architecture: Quartz (ETL) → SCP → Tempest (GPU)

Zero data dependencies between projects. The INPC registry OMOP dump
lives on Quartz; GPUs live on Tempest. ETL runs on Quartz and produces
portable tensors that get SCP'd to Tempest for all GPU work.

```
┌─────────────────────────────────────────────────────────────────────┐
│  QUARTZ  (ETL — CPU, reads raw INPC OMOP dump)                      │
├─────────────────────────────────────────────────────────────────────┤
│  01_etl_quartz.py    ICI cohort → Cr phenotyping → NCI-CCI →        │
│                      demographics → [F, N] feature tensor            │
│  Source: /N/project/depot/hw56/irAKI_data/structured_data/           │
│  Output: data/{feature_matrix.pt, feature_names.json, ...}          │
│                                                                      │
│  ─── scp -r data/ tempest:~/graphai_ici_aki/data/ ───               │
├─────────────────────────────────────────────────────────────────────┤
│  TEMPEST  (GPU — H100, training + LLM edges)                        │
├─────────────────────────────────────────────────────────────────────┤
│  02_graph.py         Build feature graph (Spearman / MI / LLM)       │
│  03_models.py        TransformerConv / GATv2 / GCN autoencoders      │
│  05_train.py         Training loop + CV + W&B logging                │
│  06_tune.py          Optuna Phase 1 (random) + Phase 2 (TPE)         │
│  09_heatmap.py       Attention extraction → heatmap figures           │
│  10_baselines.py     XGBoost / LogReg / RF on same folds             │
│  12_llm_edges.py     LLM pairwise queries → llm_adj.npy (GPU)       │
├─────────────────────────────────────────────────────────────────────┤
│  SLURM                                                               │
├─────────────────────────────────────────────────────────────────────┤
│  slurm/run_train.sh      Training / tuning (PHASE=0/1/2)            │
│  slurm/run_llm_edges.sh  LLM pairwise edge estimation               │
└─────────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# ── QUARTZ ─────────────────────────────────────────────────
python 01_etl_quartz.py
# Inspect output
ls -la data/
# Transfer to Tempest
scp -r data/ tempest:~/graphai_ici_aki/data/

# ── TEMPEST ────────────────────────────────────────────────
# Setup (one time)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric scikit-learn xgboost optuna wandb vllm
wandb login

# Build graphs (data-derived)
python 02_graph.py --method spearman --k 8
python 02_graph.py --method mi --k 8

# Debug run
python 05_train.py --arch transformer --edge_method spearman --k 8 --wandb

# Submit GPU jobs
sbatch --export=PHASE=1,ARCH=transformer,EDGE=spearman slurm/run_train.sh
sbatch --export=PHASE=2,ARCH=transformer,EDGE=spearman slurm/run_train.sh

# LLM-prior edges (parallel — no data dependency on training)
sbatch slurm/run_llm_edges.sh
# After completion:
python 02_graph.py --method llm --k 8 --llm_prior data/llm_adj.npy

# Baselines (same folds)
python 10_baselines.py --wandb

# Heatmap from best model
python 09_heatmap.py --model_path results/best_model.pt --arch transformer
```

## Data

| Source | Patients | AKI Events | Notes |
|--------|----------|------------|-------|
| INPC   | ~2,607   | ~680       | Primary (larger N) |
| AoU    | 605      | 128 (1.5×), 45 (2.0×) | External validation |

## Feature Nodes (v1.1)

- **Demographics (4):** age, sex, race_black, ethnicity_hispanic
- **NCI-CCI (14–16):** 14 scored conditions, 16 binary flags
- **ICI class (3):** ici_pd1, ici_pdl1, ici_ctla4
- **Outcome (1):** aki_event
- **Optional (1–2):** nephrotoxin, nci_cci_score
- **Total: 23–26 nodes**

## Key Dependencies

Quartz: `pandas numpy torch` (CPU only, for ETL)

Tempest (.venv):
```
torch >= 2.0  (cu121)
torch-geometric >= 2.4
optuna >= 3.0
scikit-learn, xgboost
wandb
vllm  (for 12_llm_edges.py only)
```
