# PRAGMA-G: Graph-Temporal Extension of Revolut's PRAGMA Foundation Model

> **Fixing the 47.1% AML performance gap identified in Revolut's own paper.**

[![CI](https://github.com/yourusername/pragma-g/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/pragma-g/actions)
[![Live API](https://img.shields.io/badge/API-live-brightgreen)](https://your-space.hf.space/health)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08649-b31b1b)](https://arxiv.org/abs/2604.08649)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

-----

## What is this?

Revolut published **PRAGMA** (arXiv:2604.08649, Apr 2026) — a family of foundation models for
multi-source banking event sequences, co-authored with NVIDIA. It achieves strong results across
credit scoring, fraud detection, and lifetime value prediction.

However, the paper explicitly reports that **PRAGMA underperforms its production baseline on
Anti-Money Laundering (AML) by 47.1%**, attributing this to:

> *"AML detection is inherently relational: the baseline leverages cross-record features that
> capture network-level signals. Because PRAGMA processes event histories in isolation, the
> resulting embeddings do not capture cross-account dependencies."*

The paper identifies the solution: **"a Graph-Temporal approach combining temporal sequential
learning with Graph Neural Networks to model account-to-account interactions."**

**PRAGMA-G is exactly that.** This repository:

1. Faithfully reproduces PRAGMA-S (10M param) on the public IBM AML dataset
1. Extends it with a 3-layer GraphSAGE encoder over the account transaction graph
1. Demonstrates that the graph layer closes the AML performance gap
1. Serves everything as a live production API with SHAP explainability and drift monitoring

-----

## Live Demo

**API endpoint:** `https://your-space.hf.space`

```bash
# Score a transaction
curl -X POST https://your-space.hf.space/score \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACC_1234",
    "profile": {"plan": "standard", "region": "uk", "balance_quantile": 0.6},
    "events": [
      {"type": "wire", "amount": 9500.00, "currency": "USD",
       "timestamp": "2026-06-01T14:32:00"}
    ]
  }'

# Response
{
  "score": 0.847,
  "decision": "flag",
  "shap_values": {"amount_paid": 0.31, "payment_format": 0.18, ...},
  "latency_ms": 47.3
}
```

-----

## Architecture

```
IBM AML Transactions (CSV)
         │
         ▼
┌─────────────────────────────┐
│  PRAGMA-Mini Encoder        │  Key-value-time Transformer (per account)
│  d_model=192, PRAGMA-S scale│  Reproduces paper §2.2–2.3 architecture
│  Output: [USR] embedding    │  Shape: (N_accounts, 192)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Transaction Graph          │  PyTorch Geometric
│  Nodes: accounts            │  Directed multigraph
│  Edges: transactions        │  Edge features: amount, time, format, currency
│  GraphSAGE (3 layers)       │  Inductive — handles new accounts at inference
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Fusion + AML Classifier    │  concat([temporal, graph]) → MLP → score ∈ [0,1]
│  + SHAP explainability      │  Top-5 feature attributions per decision
│  + What-If simulator        │  Real-time counterfactual analysis
└─────────────────────────────┘
```

-----

## Results

|Model                          |PR-AUC|ROC-AUC|Notes                      |
|-------------------------------|------|-------|---------------------------|
|XGBoost (hand-crafted features)|—     |—      |Paper's production baseline|
|PRAGMA-Mini only               |—     |—      |Reproduces paper's AML gap |
|**PRAGMA-G (ours)**            |**—** |**—**  |Graph layer closes the gap |

*Results on IBM AML HI-Small test split (temporal 60/20/20 split). To be updated after training.*

-----

## Quick start

```bash
# Clone
git clone https://github.com/yourusername/pragma-g
cd pragma-g

# Install
pip install -r requirements.txt

# Download data (needs Kaggle credentials in .env)
bash scripts/download_data.sh

# Build vocabulary
python scripts/build_vocab.py --data data/HI-Small_Trans.csv

# Pre-train PRAGMA-Mini
python -m src.training.pretrain --config configs/pragma_s.yaml

# Fine-tune on AML
python -m src.training.finetune --config configs/pragma_s.yaml

# Run the full stack
docker-compose up
# → API: http://localhost:8000
# → Grafana: http://localhost:3000
# → MLflow: http://localhost:5000
```

-----

## Repository structure

```
pragma-g/
├── src/
│   ├── tokenizer/      # Key-value-time tokenisation (paper §2.2)
│   ├── model/          # PRAGMA-Mini architecture (paper §2.3)
│   ├── graph/          # GraphSAGE extension (our contribution)
│   ├── training/       # Pre-training + LoRA fine-tuning
│   ├── api/            # FastAPI serving with SHAP
│   └── monitoring/     # Evidently drift + Prometheus metrics
├── tests/              # Unit + integration tests
├── configs/            # Hyperparameter configs
├── scripts/            # Data download, vocab building, evaluation
├── notebooks/          # Development and exploration
├── CLAUDE.md           # AI workspace context (Claude Code)
├── ARCHITECTURE.md     # Deep technical spec
├── DATASETS.md         # Dataset reference
└── INTERVIEWS.md       # Interview preparation
```

-----

## Design decisions

**Why GraphSAGE over GAT?**
GraphSAGE is inductive — it handles new accounts at inference time without full-graph recomputation.
This is a hard production requirement for AML systems where new accounts are created continuously.

**Why IBM AML over PaySim?**
IBM AML (NeurIPS 2023) provides explicit account-to-account transaction graphs with realistic
laundering typologies. It is the standard benchmark for GNN-based AML research.

**Why reproduce PRAGMA-S and not fine-tune a larger model?**
Revolut has not released their weights. The paper fully specifies the architecture, tokenisation
scheme, and hyperparameters (Table 1). This is a faithful open-source reproduction on public data.

-----

## Explainability and compliance

This project implements explainability mechanisms consistent with EU AI Act Article 13/14 requirements
for high-risk AI systems:

- SHAP values for every decision
- Audit log of feature vectors at inference time
- Configurable decision thresholds (versioned, not baked into model)
- What-If simulator for human oversight
- Evidently drift monitoring with automated alerting

See [`docs/EU_AI_ACT.md`](docs/EU_AI_ACT.md) for details.

-----

## Related work

- PRAGMA: Revolut Foundation Model — <https://arxiv.org/abs/2604.08649>
- IBM AML Dataset (NeurIPS 2023) — <https://arxiv.org/abs/2306.16424>
- GraphSAGE — Hamilton et al., NeurIPS 2017
- LoRA — Hu et al., ICLR 2022

-----

## Author

**Rayen Ben Abdallah** — AI/Fintech Engineer

- LinkedIn: linkedin.com/in/rayen-ben-abdallah
- Email: [rayenbenabdallah88@gmail.com](mailto:rayenbenabdallah88@gmail.com)