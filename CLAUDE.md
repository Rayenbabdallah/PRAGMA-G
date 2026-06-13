# PRAGMA-G — Claude Code Workspace

## Who I am

Rayen Ben Abdallah — final-year IT + Finance student at Tunis Business School (graduating 2027).
Currently a junior consultant at PwC TAC (Microsoft/Dynamics 365 Finance team).
Building this project to land an AI/fintech engineering role at Revolut, Klarna, Wise, or similar.

## What this project is

**PRAGMA-G** is an open-source Graph-Temporal extension of Revolut’s published PRAGMA foundation model
(arXiv:2604.08649, Apr 2026 — co-authored by Revolut Research + NVIDIA).

The PRAGMA paper explicitly states:

> “AML detection is inherently relational… PRAGMA processes event histories in isolation,
> resulting in a 47.1% performance degradation vs the production baseline.”
> “The next frontier is a Graph-Temporal approach combining temporal sequential learning with GNNs.”

**PRAGMA-G is exactly that.** We reproduce PRAGMA-Mini (the key-value-time Transformer encoder)
on the public IBM AML dataset, then graft a GraphSAGE layer on top to capture account-to-account
relational signals — the specific gap Revolut’s paper identifies.

## Research paper

- **Title:** PRAGMA: Revolut Foundation Model
- **arXiv:** <https://arxiv.org/abs/2604.08649>
- **Authors:** Ostroukhov et al. (Revolut Research + NVIDIA), Apr 2026
- **Key architecture:** Encoder-only Transformer, key-value-time tokenisation, two-branch design
  (Profile State Encoder + Event Encoder → History Encoder), MLM pre-training
- **PRAGMA-S (10M params):** d_model=192, d_ffn=768, Profile layers=1, Event layers=5, History layers=2, Heads=3
- **PRAGMA-M (100M):** d_model=512, d_ffn=2048, Profile=3, Event=16, History=6, Heads=8
- **Known gap:** PRAGMA underperforms on AML by 47.1% because it has no cross-account graph signal

## Dataset

**IBM AML (Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml)**

- Synthetic, agent-based, calibrated to real financial behaviour (NeurIPS 2023 paper)
- HI-Small: high illicit ratio, ~5K accounts, ~5M transactions — use this for dev
- HI-Large: use for final training runs
- Files per split: transactions CSV + account graph edges CSV
- Labels are at transaction level (is_laundering: 0/1)
- Graph construction: node = account, edge = transaction (directed, weighted by amount)
- License: CDLA-Sharing-1.0 (free for research/non-commercial)
- Download: `scripts/download_data.sh`

## My existing skills that transfer here

- **Olea Intelligence XAI engine** → reuse What-If simulator + SHAP explainability layer
- **Tunax** → Redis session handling + Nginx reverse proxy pattern
- **Sagemcom** → CI/CD pipeline thinking (GitLab → GitHub Actions translation)
- **Markowitz portfolio** → finance domain depth for interview conversations
- **Flask/FastAPI** → API serving layer

## Architecture: PRAGMA-G

```
IBM AML Transactions (CSV)
         │
         ▼
┌─────────────────────────────┐
│  PRAGMA-Mini Encoder        │  ← Key-value-time Transformer (per account)
│  - Profile State Encoder    │    Reproduces paper §2.2, §2.3 on public data
│  - Event Encoder            │    d_model=192 (PRAGMA-S scale)
│  - History Encoder          │
│  Output: [USR] embedding    │    shape: (N_accounts, 192)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Transaction Graph          │  ← PyTorch Geometric
│  Nodes: accounts            │    Directed multigraph
│  Edges: transactions        │    Edge features: amount, time delta, tx type
│  GraphSAGE (3 layers)       │
│  Output: graph embedding    │    shape: (N_accounts, 192)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Fusion + Classification    │  ← concat([pragma_emb, graph_emb]) → MLP head
│  AML risk score ∈ [0,1]    │    + SHAP explainability
│  Decision: flag / clear     │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  FastAPI serving            │  ← /score endpoint, <100ms p95
│  Evidently drift monitor    │
│  MLflow experiment tracker  │
│  Prometheus + Grafana       │
└─────────────────────────────┘
```

## Tech stack

|Layer              |Tool                            |Version|
|-------------------|--------------------------------|-------|
|Language           |Python                          |3.11   |
|Deep learning      |PyTorch                         |2.3+   |
|Graph ML           |PyTorch Geometric               |2.5+   |
|Transformer        |Custom (paper-faithful impl)    |—      |
|Explainability     |SHAP                            |0.45+  |
|Experiment tracking|MLflow                          |2.13+  |
|Serving            |FastAPI + Uvicorn               |0.111+ |
|Drift monitoring   |Evidently                       |0.4+   |
|Metrics            |Prometheus + Grafana            |—      |
|CI/CD              |GitHub Actions                  |—      |
|Containerisation   |Docker + docker-compose         |—      |
|Deployment         |Hugging Face Spaces (Docker SDK)|—      |
|Data download      |kaggle CLI                      |—      |

## Repo structure

```
pragma-g/
├── CLAUDE.md                    ← THIS FILE — read first every session
├── PLAN.md                      ← 12-week build plan with weekly milestones
├── ARCHITECTURE.md              ← Deep architecture spec from the paper
├── DATASETS.md                  ← Dataset details, download instructions, schema
├── INTERVIEWS.md                ← Interview Q&A prep tied to this project
├── README.md                    ← Public-facing project README
├── .github/workflows/
│   ├── ci.yml                   ← Lint, test, data-validation gate on every PR
│   └── deploy.yml               ← Build + push Docker to HF Spaces on main merge
├── src/
│   ├── tokenizer/               ← Key-value-time tokenisation (paper §2.2)
│   │   ├── __init__.py
│   │   ├── vocab.py             ← Vocabulary builder (~60 key tokens, ~28K value tokens)
│   │   └── tokenizer.py         ← KVT tokeniser: numerical buckets, categorical, BPE text
│   ├── model/                   ← PRAGMA-Mini Transformer (paper §2.3)
│   │   ├── __init__.py
│   │   ├── embeddings.py        ← Key embed + value embed + temporal embed (RoPE)
│   │   ├── profile_encoder.py   ← Profile State Encoder branch
│   │   ├── event_encoder.py     ← Event Encoder branch
│   │   ├── history_encoder.py   ← History Encoder (fusion)
│   │   └── pragma_mini.py       ← Full PRAGMA-Mini model (PRAGMA-S scale)
│   ├── graph/                   ← Graph layer (PRAGMA-G extension)
│   │   ├── __init__.py
│   │   ├── graph_builder.py     ← Build PyG graph from IBM AML transactions
│   │   └── graphsage.py         ← GraphSAGE encoder (3 layers, edge features)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py          ← MLM pre-training loop (paper §2.3.5)
│   │   ├── finetune.py          ← AML downstream fine-tuning (LoRA)
│   │   └── losses.py            ← MLM loss + AML classification loss
│   ├── api/                     ← FastAPI serving
│   │   ├── __init__.py
│   │   ├── main.py              ← FastAPI app, /score, /health, /explain endpoints
│   │   ├── schemas.py           ← Pydantic request/response models
│   │   └── model_loader.py      ← Singleton model loader with warm-up
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py             ← Evidently drift reports on live traffic
│       └── metrics.py           ← Prometheus counters: latency, score dist, drift flags
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_graph.py
│   ├── test_api.py
│   └── conftest.py
├── scripts/
│   ├── download_data.sh         ← kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml
│   ├── build_vocab.py           ← Run once to build vocabulary from IBM AML data
│   └── evaluate.py              ← Evaluation script: PR-AUC, ROC-AUC, cost-based metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenizer_dev.ipynb
│   ├── 03_pragma_mini_dev.ipynb
│   └── 04_graph_layer_dev.ipynb
├── data/                        ← gitignored, populated by download_data.sh
├── configs/
│   ├── pragma_s.yaml            ← PRAGMA-S hyperparams (d_model=192, per paper Table 1)
│   └── training.yaml            ← Training hyperparams
├── docker-compose.yml           ← Wires: API + Prometheus + Grafana + MLflow
├── Dockerfile                   ← API container
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Key design decisions (explain in interviews)

1. **Why reproduce PRAGMA-Mini instead of using their weights?**
   Their weights are proprietary and not released. The paper fully specifies the architecture
   (tokenisation scheme, two-branch design, hyperparams in Table 1). We faithfully implement
   PRAGMA-S scale on public data — this is the correct scientific approach.
1. **Why IBM AML and not PaySim?**
   IBM AML (NeurIPS 2023) provides explicit account-to-account transaction graphs with
   realistic laundering typologies calibrated to real financial behaviour. PaySim simulates
   mobile money (M-Pesa style) with different network structure. IBM AML is the standard
   benchmark for graph-based AML research (referenced by 20+ papers).
1. **Why GraphSAGE over GAT or GCN?**
   GraphSAGE is inductive — it generalises to unseen accounts at inference time, which is
   critical for a production AML system where new accounts appear continuously. GAT requires
   the full graph at inference; GCN does too. GraphSAGE scales to millions of nodes.
1. **Why the 47.1% AML gap matters**
   The paper reports PRAGMA underperforms on AML specifically because it lacks cross-record
   signals. PRAGMA-G adds exactly those signals. This is a direct, paper-cited gap we are
   filling — not an arbitrary extension.
1. **Latency target: <100ms p95**
   PRAGMA-Mini (PRAGMA-S scale, 10M params) runs in ~15ms on CPU for a single record.
   GraphSAGE neighbourhood sampling adds ~20ms for 2-hop neighbourhood. Full pipeline
   including FastAPI overhead targets <100ms p95.

## What NOT to do (anti-patterns)

- Do NOT hardcode hyperparams in model files — use configs/pragma_s.yaml
- Do NOT skip type hints — every function needs them
- Do NOT commit data files — data/ is gitignored
- Do NOT use `float("inf")` for temporal gaps — use the log transform from the paper (§2.2)
- Do NOT use standard positional encoding — use RoPE (Rotary Position Embedding) per the paper
- Do NOT build a monolith — each src/ submodule should be independently testable
- Do NOT skip the MLM pre-training step — embedding probe performance depends on it

## Running the project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
bash scripts/download_data.sh   # needs KAGGLE_USERNAME and KAGGLE_KEY in .env

# 3. Build vocabulary
python scripts/build_vocab.py --data data/HI-Small_Trans.csv

# 4. Pre-train PRAGMA-Mini
python -m src.training.pretrain --config configs/pragma_s.yaml

# 5. Fine-tune on AML
python -m src.training.finetune --config configs/pragma_s.yaml

# 6. Serve API
docker-compose up

# 7. Run tests
pytest tests/ -v

# 8. Evaluate
python scripts/evaluate.py --split test
```

## Session startup checklist (read before every Claude Code session)

- [ ] What phase of PLAN.md are we in?
- [ ] What was the last completed task?
- [ ] What is the specific file/function we’re building today?
- [ ] Are tests passing? (`pytest tests/ -v`)
- [ ] Is the API still live? (check HF Spaces URL)
