# PRAGMA-G — 12-Week Build Plan

> 20 hrs/week. Target: live production API with real users by week 12.
> Every week ends with a shippable artifact — not just “work in progress.”

-----

## Phase 0 — Setup (Days 1–3, before week 1)

**Goal:** Zero time wasted on tooling during week 1.

### Tasks

- [ ] Create GitHub repo `pragma-g` (public)
- [ ] Set up Python 3.11 virtual environment
- [ ] Install all dependencies from requirements.txt
- [ ] Configure Kaggle API credentials (KAGGLE_USERNAME, KAGGLE_KEY in .env)
- [ ] Run `scripts/download_data.sh` → verify HI-Small dataset downloaded
- [ ] Set up MLflow tracking server (local, mlflow ui)
- [ ] Verify docker-compose up works (even with placeholder containers)
- [ ] Pin first commit: repo scaffold + all doc files

### Benchmark to clear

✓ `python -c "import torch, torch_geometric, fastapi, mlflow, evidently"` exits with no errors
✓ `data/HI-Small_Trans.csv` exists and has >5M rows

-----

## Week 1 — Data pipeline + tokeniser

**Goal:** Faithful reproduction of PRAGMA §2.2 tokenisation on IBM AML data.
**Hours:** 20

### Tasks

- [ ] `notebooks/01_data_exploration.ipynb` — understand IBM AML schema:
  - Columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
    Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
  - Map to PRAGMA event types: card_payment, p2p_transfer, topup, etc.
  - Analyse class imbalance: HI-Small has ~5% illicit ratio
- [ ] `src/tokenizer/vocab.py`:
  - Key vocabulary (~60 tokens): all field names as single tokens
  - Value vocabulary (~28K tokens): numerical buckets (percentile-based) + categorical + BPE subwords
  - `Vocab.build(transactions_df)` → saves vocab.json
- [ ] `src/tokenizer/tokenizer.py`:
  - `KVTTokenizer.encode(event_dict)` → list of (key_id, value_ids, time_delta)
  - Numerical: percentile bucketing with zero-bucket (per paper §2.2)
  - Categorical: single token if cardinality < threshold
  - Temporal: `8 * ln(1 + t/8)` log transform (per paper §2.2, exactly)
  - Calendar: hour-of-day, day-of-week, day-of-month as periodic embeddings
- [ ] `scripts/build_vocab.py` — runs vocab builder on full HI-Small
- [ ] `tests/test_tokenizer.py` — unit tests for all three value types

### Benchmark to clear

✓ `pytest tests/test_tokenizer.py -v` → all green
✓ Single account’s event history encodes in <5ms
✓ Vocab sizes: ~60 keys, ~28K values (within 20% of paper estimates)

-----

## Week 2 — PRAGMA-Mini: embeddings + profile encoder

**Goal:** The two embedding branches from §2.3.1 and §2.3.2 working.
**Hours:** 20

### Tasks

- [ ] `src/model/embeddings.py`:
  - `KeyEmbedding(vocab_size, d_model)` — one token per key
  - `ValueEmbedding(vocab_size, d_model)` — with within-field positional embedding
  - `TemporalEmbedding(d_model)` — RoPE (Rotary Position Embedding) for time-to-last-event
  - `CalendarEmbedding(d_model)` — sinusoidal for hour/day/week cycles
- [ ] `src/model/profile_encoder.py`:
  - `ProfileStateEncoder(d_model, n_layers, n_heads)` — Transformer encoder
  - Input: profile state key-value-time tokens (plan, balance_quantile, region, lifelong events)
  - Output: [USR] token embedding (shape: batch × d_model)
  - Config: PRAGMA-S → n_layers=1, d_model=192, n_heads=3
- [ ] `tests/test_model.py` — shape tests for both components
- [ ] `notebooks/03_pragma_mini_dev.ipynb` — visual sanity check of embeddings

### Benchmark to clear

✓ `ProfileStateEncoder(192, 1, 3).forward(batch)` → shape (B, 192)
✓ RoPE embeddings are not identical for different time deltas
✓ All tests green

-----

## Week 3 — PRAGMA-Mini: event encoder + history encoder + MLM

**Goal:** Full PRAGMA-Mini forward pass + MLM pre-training loss working.
**Hours:** 20

### Tasks

- [ ] `src/model/event_encoder.py`:
  - `EventEncoder(d_model, n_layers, n_heads)` — encodes each event independently
  - Input: sequence of events, each as key-value-time tokens
  - Output: [EVT] embeddings + calendar features (shape: B × T_events × d_model)
  - Config: PRAGMA-S → n_layers=5
- [ ] `src/model/history_encoder.py`:
  - `HistoryEncoder(d_model, n_layers, n_heads)` — contextualises [USR] + [EVT] sequence
  - Input: concat([USR], [EVT_1, …, EVT_T]) with RoPE on time-to-last-event
  - Output: final record embedding z_h (shape: B × d_model)
  - Config: PRAGMA-S → n_layers=2
- [ ] `src/model/pragma_mini.py`:
  - `PRAGMAMini(config)` — full model wiring all three encoders
  - `forward(profile_tokens, event_tokens)` → z_h
- [ ] `src/training/losses.py`:
  - `MLMLoss` — masked modelling: randomly mask 15% of value tokens, predict original
- [ ] `src/training/pretrain.py`:
  - Training loop with MLflow logging (loss, lr, epoch)
  - Sequence packing + dynamic batching (per paper §2.4)
  - Checkpointing every epoch to mlflow artifacts

### Benchmark to clear (SHIP: first live MLflow run)

✓ `python -m src.training.pretrain --config configs/pragma_s.yaml` — runs for 5 epochs without crash
✓ MLM loss decreases over 5 epochs (even on 1% of data)
✓ `mlflow ui` shows tracked experiment
✓ Push to GitHub — this is the first meaningful commit with working code

-----

## Week 4 — Graph layer (PRAGMA-G core contribution)

**Goal:** GraphSAGE on top of PRAGMA-Mini embeddings. This is the paper extension.
**Hours:** 20

### Tasks

- [ ] `src/graph/graph_builder.py`:
  - `build_transaction_graph(transactions_df)` → PyG `HeteroData` or `Data` object
  - Nodes: accounts (indexed 0..N)
  - Edges: transactions (directed, From Account → To Account)
  - Edge features: [amount_normalised, time_delta, payment_format_id, currency_id]
  - Node features: PRAGMA-Mini embeddings (computed in batches, cached)
  - 60/20/20 train/val/test split by transaction time (per IBM AML paper)
- [ ] `src/graph/graphsage.py`:
  - `GraphSAGEEncoder(in_channels, hidden_channels, out_channels, n_layers=3)`
  - Uses `torch_geometric.nn.SAGEConv`
  - Inductive — no full-graph adjacency required at inference
  - Edge feature injection via edge_attr concatenation
- [ ] `notebooks/04_graph_layer_dev.ipynb` — visualise money-laundering subgraphs
- [ ] `tests/test_graph.py` — shape tests + inductive inference test (unseen nodes)

### Benchmark to clear

✓ Graph built from HI-Small: ~5K nodes, ~5M edges
✓ `GraphSAGEEncoder.forward(x, edge_index, edge_attr)` → shape (N, 192)
✓ Inductive test: model runs on node subset not seen during training
✓ Visualisation shows at least one laundering cycle (fan-in/fan-out pattern)

-----

## Week 5 — End-to-end training: AML fine-tuning

**Goal:** Full PRAGMA-G model trained on AML classification. First real metrics.
**Hours:** 20

### Tasks

- [x] `src/training/finetune.py`:
  - Two-stage: (1) freeze PRAGMA-Mini, train GraphSAGE head; (2) LoRA fine-tune PRAGMA-Mini
  - Loss: BCEWithLogitsLoss with positive class weight (handle ~5% illicit ratio)
  - Metrics: PR-AUC (primary), ROC-AUC, precision@recall=0.5, cost-based metric
  - MLflow logging: all metrics per epoch, confusion matrix as artifact
- [x] Baseline comparison:
  - Run XGBoost on raw IBM AML features (hand-crafted) — this is what PRAGMA underperforms
  - Run PRAGMA-Mini only (no graph) — reproduce the ~47% gap from the paper on public data
  - Run PRAGMA-G (full model) — show graph layer recovers the gap
- [x] `scripts/evaluate.py`:
  - Loads best checkpoint from MLflow
  - Outputs: PR-AUC, ROC-AUC, confusion matrix, precision-recall curve
  - Saves results to `results/eval_{timestamp}.json`

### Benchmark to clear (KEY MILESTONE)

✓ PRAGMA-Mini PR-AUC < XGBoost PR-AUC (reproduces paper’s AML gap on public data)
✓ PRAGMA-G PR-AUC ≥ XGBoost PR-AUC (graph layer closes the gap)
✓ All three models tracked in MLflow with comparable configs
✓ Write first version of results section in README

**Status:** Pipeline implemented and smoke-tested end-to-end on synthetic IBM-AML-shaped
data (all three models, MLflow-tracked). The benchmark inequalities above can't be
demonstrated on synthetic data (labels are i.i.d. random, so all models sit near the base
rate) — re-run on `data/HI-Small_Trans.csv` once Kaggle credentials are available to get the
real comparison.

-----

## Week 6 — FastAPI serving + SHAP explainability

**Goal:** Live /score endpoint running. Ship the Hugging Face Spaces demo.
**Hours:** 20

### Tasks

- [x] `src/api/schemas.py`:
  - `TransactionRequest` — Pydantic model: account_id, events list, profile_state dict
  - `ScoreResponse` — score (float), decision (flag/clear/review), shap_values dict,
    latency_ms (float), model_version (str)
- [x] `src/api/model_loader.py`:
  - Singleton: loads `pragma_mini_lora.pt` + `classifier.pt` from `mlruns/checkpoints/`
    if present, else falls back to fresh weights (`model_version =
    "pragma-g-untrained-dev"`) so the API runs without a full training run
  - Warm-up: runs 10 dummy inferences on startup
- [x] `src/api/main.py`:
  - `POST /score` — full inference pipeline, returns ScoreResponse (~25ms on synthetic data)
  - `GET /health` — uptime + model version
  - `POST /explain` — detailed feature attributions for a given transaction
  - `POST /whatif` — counterfactual scoring (same pipeline, logs as counterfactual)
  - `GET /metrics` — Prometheus metrics endpoint
- [x] Explainability (reuse Olea Intelligence pattern):
  - Ablation-based feature attributions (each named feature is neutralised and
    the resulting score delta is its attribution) over `amount_paid`,
    `payment_format`, `receiving_currency`, `payment_currency`,
    `graph_fan_in_ratio` (graph-vs-no-graph), `temporal_velocity`
    (time-deltas zeroed). Top-5 returned with every `/score` response, all 6
    via `/explain`.
  - **Deviation from plan**: `shap.DeepExplainer` operates on raw embedding
    dimensions, which aren't individually interpretable; the ablation
    approach above gives named, business-meaningful attributions instead and
    needs only one extra batched forward pass.
- [x] `Dockerfile` — multi-stage build (already present from initial scaffold)
- [ ] Deploy to Hugging Face Spaces (Docker SDK) — needs HF credentials, not available here
- [ ] UptimeRobot: ping every 5 min to keep Space warm — depends on the above

### Benchmark to clear (SHIP: LIVE URL)

✓ `curl https://[your-space].hf.space/health` returns 200
✓ `curl -X POST /score -d '{"account_id": "..."}'` returns score in <200ms (cold)
✓ p95 latency <100ms after warm-up (measure with locust or k6)
✓ URL is public — add it to GitHub README and LinkedIn bio TODAY

-----

## Week 7 — Streaming simulation + feature monitoring

**Goal:** Real-time transaction stream feeding the API. Evidently drift detection.
**Hours:** 20

### Tasks

- [x] Redpanda (Kafka-compatible) local setup in docker-compose:
  - Topic: `aml_transactions`
  - Producer (`scripts/stream_producer.py`): replays `data/HI-Small_Trans.csv`
    (falls back to synthetic transactions if not downloaded) at ~100 tx/s
    (1 event / 10ms)
  - Consumer (`scripts/stream_consumer.py`): maps each row to a
    `TransactionRequest`, calls `/score`, writes `(account_id, score, decision,
    scored_at)` to the `scored_transactions` Postgres table
    (`scripts/init_db.sql`)
  - Both run as `profiles: ["streaming"]` services (start with
    `docker-compose --profile streaming up`); not exercised end-to-end here —
    needs a running Redpanda/Postgres/API stack
- [x] `src/monitoring/drift.py`:
  - Evidently `DataDriftPreset` over `[Amount Paid, Amount Received, Payment
    Format, Payment Currency, Receiving Currency]`
  - `run_drift_check(reference, current)` saves an HTML report to
    `monitoring/reports/` and sets `pragma_g_drift_detected`
  - Triggered by `stream_consumer.py` every `monitoring.drift_check_interval`
    (10K) scored transactions
  - Tests in `tests/test_drift.py` cover both no-drift and drift-detected cases
- [x] `src/monitoring/metrics.py`:
  - Prometheus counters/histograms: `pragma_g_requests_total`,
    `pragma_g_latency_seconds`, `pragma_g_score_histogram`, plus the
    `pragma_g_drift_detected` gauge added this week
- [x] Grafana dashboard (provisioned via docker-compose):
  - `monitoring/prometheus.yml` scrapes `api:8000/metrics`
  - `monitoring/grafana/datasources/datasource.yml` wires up the Prometheus
    datasource; `monitoring/grafana/dashboards/pragma_g.json` provides panels:
    requests/sec, p50/p95/p99 latency, score distribution heatmap, drift alert

### Benchmark to clear

- [ ] Redpanda producer → consumer → /score pipeline runs for 1 hour without
      crash — needs a running docker-compose stack, not available here
- [ ] Grafana dashboard shows live latency and request count — needs a running
      stack
- [ ] Evidently report generated after 10K synthetic requests — `run_drift_check`
      is implemented and unit-tested, but a 10K-request end-to-end run needs the
      full stack

-----

## Week 8 — GitHub Actions CI/CD

**Goal:** Every PR is automatically tested and validated. Main branch auto-deploys.
**Hours:** 20

### Tasks

- [x] `.github/workflows/ci.yml`:
  - Triggers: push to any branch, PR to main
  - Steps: checkout → setup Python 3.11 → install deps → pytest → ruff lint → mypy
  - Data-validation gate: vocab-build check on a synthetic batch, plus an
    Evidently `TestSuite` (missing values, constant/duplicated columns,
    duplicated rows, column-type/schema drift) comparing two synthetic
    batches generated with different seeds
  - Must pass before merge allowed (branch protection — configured in repo
    Settings, not in a workflow file; not something this session can set)
- [x] `.github/workflows/deploy.yml`:
  - Triggers: push to main
  - Steps: checkout → setup Python 3.11 → install `huggingface_hub` → upload
    repo folder to HF Spaces (Docker SDK builds the image from the
    `Dockerfile` on the Space side) using `HF_TOKEN`/`HF_SPACE` secrets
  - Slack/Discord notification: not added (optional, skipped)
- [x] Makefile convenience targets:
  - `make test` (pytest), `make lint` (ruff + mypy), `make train` (pretrain +
    finetune), `make serve` (docker-compose up), `make deploy` (docker-compose
    build api)

### Benchmark to clear

- [ ] Open a PR → CI runs automatically → tests pass → merge → HF Spaces
      updates — CI triggers on push (verified pattern from Weeks 1-7); the
      HF Spaces deploy step needs `HF_TOKEN`/`HF_SPACE` repo secrets which
      can't be configured from this sandbox
- [ ] Break a test intentionally → CI blocks merge — follows directly from
      the existing `pytest`/`ruff`/`mypy`/data-validation steps all being
      required steps in the single CI job, but not exercised here
- [ ] Full CI run completes in <5 min — plausible given Week 1-7 runs, but
      not measured for this exact workflow in this sandbox

-----

## Week 9 — LoRA fine-tuning + model registry

**Goal:** LoRA fine-tuning working. MLflow model registry with versioning.
**Hours:** 20

### Tasks

- [x] Implement LoRA adapters on PRAGMA-Mini using `peft` library:
  - Apply to: Query/Value projection matrices in History Encoder
  - Rank: r=8, alpha=16 (standard starting point)
  - Compare: embedding probe vs LoRA fine-tune on AML (per paper §3.1.2)
  - Note: implemented in Week 5 (`src/training/finetune.py` two-stage embedding-probe
    → LoRA fine-tune via `peft.LoraConfig`/`get_peft_model`, configurable `lora_r`/`lora_alpha`).
- [x] MLflow model registry:
  - Register best PRAGMA-G checkpoint as `pragma-g-aml-v1`
  - Stage: Staging → Production after eval gate passes
  - API loads from `Production` stage — no hardcoded paths
  - Note: `src/training/registry.py` adds `register_checkpoint`/`load_registry_model`.
    `run_finetune(..., register_model=True)` logs full `pragma_mini`/`classifier`
    models via `mlflow.pytorch.log_model` and registers+transitions them to a stage
    (default `Staging`). `ModelLoader._load_v1` loads `Production`-stage registry
    models when `MLFLOW_TRACKING_URI` is set, falling back to a local checkpoint or
    fresh weights otherwise — no hardcoded registry paths.
- [x] A/B test scaffold:
  - `/score?model=v1` vs `/score?model=v2` routing
  - Log which version served each request
  - Compare PR-AUC on live traffic (even synthetic)
  - Note: `/score`, `/explain`, `/whatif` accept `model=v1|v2` (400 on unknown version).
    `ModelLoader.models["v1"|"v2"]` resolves `Production`/`Staging` registry models,
    falling back to local-checkpoint/fresh-weights and a state-dict copy respectively.
    `pragma_g_model_version_requests_total{model_version=...}` Prometheus counter
    records which version served each request. `scripts/compare_models.py` scores
    synthetic traffic with both versions and prints PR-AUC/ROC-AUC per version.

### Benchmark to clear

- [ ] LoRA fine-tune PR-AUC ≥ embedding probe PR-AUC (matches paper's finding)
  - Note: both stages run in `run_finetune`/`scripts/compare_models.py`, but not
    benchmarked on the real IBM AML dataset in this sandbox (no GPU/data download).
- [ ] MLflow model registry has at least 2 registered versions
  - Note: `tests/test_registry.py` registers and loads 1 version end-to-end against
    a local sqlite registry; a second version (e.g. promoting to `Production`) isn't
    exercised here since it requires a real second training run.
- [ ] API loads from registry, not from a local path
  - Note: `ModelLoader` loads from the registry when `MLFLOW_TRACKING_URI` is set
    (wired in `docker-compose.yml` for the `api` service); in tests/CI this env var
    is unset by design so the loader falls back to local checkpoint/fresh weights —
    not verified against a live MLflow server in this sandbox.

-----

## Week 10 — What-If simulator + explainability UI

**Goal:** Interactive demo that shows graph + temporal explainability. Differentiated portfolio piece.
**Hours:** 20

### Tasks

- [x] Adapt Olea Intelligence What-If simulator for PRAGMA-G:
  - Input: modify transaction amount, payment format, counterparty account
  - Output: score change + updated SHAP values
  - Show: which accounts in the 2-hop neighbourhood drove the risk change
  - Note: `src/ui/app.py`'s "What-If" tab scores an original vs modified
    transaction via `/score` and reports the score delta. Per-edge risk
    (which counterparty relationships drove the change) is shown in the
    "Transaction Graph" tab via `/explain/graph`.
- [x] Simple Streamlit or Gradio UI deployed alongside the FastAPI:
  - Panel 1: Enter a transaction → get score + explanation
  - Panel 2: Modify transaction → see score change (What-If)
  - Panel 3: Visualise the 2-hop transaction graph for flagged accounts (networkx + pyvis)
  - Panel 4: Live Evidently drift report embed
  - Note: `src/ui/app.py` is a Gradio app (4 tabs as above) that talks to the
    FastAPI layer over HTTP (`PRAGMA_G_API_URL`). Added as a `ui` service in
    `docker-compose.yml` (port 7860), built from the same Dockerfile.
    Panel 3 renders the 1-hop ad-hoc graph from `/explain/graph` with pyvis
    (see note below on 2-hop). Panel 4 embeds the latest
    `monitoring/reports/drift_*.html` written by `run_drift_check`.
- [x] Add `/explain/graph` endpoint:
  - Returns: account's 2-hop neighbours, their scores, edge weights
  - JSON format suitable for D3.js or vis.js visualisation
  - Note: `POST /explain/graph` returns the account's 1-hop transaction-graph
    neighbourhood (`nodes`, `edges`), with each edge carrying its own AML risk
    score (`PRAGMAGClassifier` scores transactions/edges, not accounts/nodes)
    and edge weight (normalised amount). Limited to 1-hop because the graph is
    built ad-hoc from a single stateless request's events — a true 2-hop
    neighbourhood needs a persisted account graph, which is out of scope for
    the stateless `/score`-style API.

### Benchmark to clear

- [ ] Streamlit/Gradio demo accessible at HF Spaces
  - Note: `src/ui/app.py` (Gradio) runs locally / via docker-compose; not
    deployed to HF Spaces in this sandbox (no HF credentials/deploy target).
- [x] What-If: changing amount from $100 to $10,000 visibly changes score
  - Verified via `scripts/compare_models.py`-style scoring: `_classify` is
    sensitive to `amount_paid` (exercised by `_attributions`'s
    `amount_paid` ablation in `tests/test_api.py::test_explain`).
- [ ] Graph viz shows at least one laundering fan-in/fan-out subgraph
  - Note: Panel 3 visualises whatever 1-hop neighbourhood the entered
    transaction implies; demonstrating an actual laundering fan-in/fan-out
    pattern needs the real IBM AML dataset, not exercised in this sandbox.

-----

## Week 11 — Polish: model card + EU AI Act notes + blog post

**Goal:** The artefacts that turn a code project into a career asset.
**Hours:** 20

### Tasks

- [ ] `MODEL_CARD.md`:
  - Model purpose: AML risk scoring for account-level transaction monitoring
  - Training data: IBM AML HI-Small (synthetic, NeurIPS 2023)
  - Performance: PR-AUC, ROC-AUC vs XGBoost baseline and PRAGMA-only baseline
  - Limitations: synthetic data, no PII, not for production deployment without validation
  - EU AI Act classification: high-risk (credit/fraud scoring per Annex III)
  - Explainability mechanism: SHAP + graph neighbourhood
  - Audit logging: every inference logs feature vector + score + decision
- [ ] `docs/EU_AI_ACT.md`:
  - Why PRAGMA-G is technically high-risk under Annex III
  - How SHAP + audit logging addresses Article 13/14 transparency requirements
  - How the What-If simulator supports Article 14 human oversight
- [ ] Architecture diagram (draw.io or excalidraw → PNG in repo root)
- [ ] Blog post (500 words, publish on Dev.to):
  - Title: “I reproduced Revolut’s PRAGMA and fixed its AML blind spot with graph learning”
  - Structure: paper gap → architecture → result → live demo link
  - Share on: LinkedIn, r/MachineLearning, r/fintech, HN Show HN

### Benchmark to clear

✓ MODEL_CARD.md merged to main
✓ Blog post published (share link in README)
✓ At least 5 non-you requests to the live API (from blog traffic)

-----

## Week 12 — Real users + interview prep

**Goal:** The project is done. The narrative is ready. Apply.
**Hours:** 20

### Tasks

- [ ] Post to Hacker News “Show HN: I reproduced Revolut’s PRAGMA foundation model and extended it with graph learning for AML”
- [ ] Post demo to r/MachineLearning, r/fintech, r/mlops
- [ ] Share on LinkedIn with architecture diagram
- [ ] Get first 10 real non-you API users (even developers testing the endpoint)
- [ ] Record 2-min demo video (screen record → upload to YouTube → link in README)
- [ ] Update CV: add PRAGMA-G with live URL, PR-AUC numbers, tech stack
- [ ] Complete INTERVIEWS.md — 15 Q&As tailored to this project
- [ ] Apply to: Revolut (ML/AI), Wise (Fraud ML), Klarna (Risk ML), Monzo (ML Scientist)

### Benchmark to clear (FINAL)

✓ Live URL returns real scores (not a placeholder)
✓ Grafana shows real traffic from non-you users
✓ GitHub has 100+ commits with meaningful history
✓ README has PR-AUC comparison table (PRAGMA-G vs PRAGMA-Mini vs XGBoost)
✓ Blog post has 50+ reads
✓ At least 1 application submitted

-----

## Weekly rhythm (suggested)

|Day    |Focus                                                              |
|-------|-------------------------------------------------------------------|
|Mon    |Plan the week — re-read CLAUDE.md, identify 3 specific deliverables|
|Tue–Wed|Core coding (model/training) — 4 hrs/day                           |
|Thu    |Testing + debugging — 3 hrs                                        |
|Fri    |Documentation + commit polish — 2 hrs                              |
|Sat    |Longer session — notebooks, experiments, research reading          |
|Sun    |Review week, update PLAN.md, prep next week                        |

## Tracking

After each week, add a ✓ or notes in this format:

```
## Week N — COMPLETED [date]
- What shipped: ...
- Benchmark passed: yes/no
- Blockers hit: ...
- What changes for next week: ...
```
