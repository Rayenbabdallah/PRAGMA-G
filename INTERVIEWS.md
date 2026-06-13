# PRAGMA-G — Interview Preparation

> Answers tailored to Revolut, Wise, Klarna, and Monzo ML/AI engineering interviews.
> Every answer references the actual project. Never give a generic answer.

-----

## Part 1 — “Tell me about your project” questions

### Q1: Walk me through a project you’re most proud of.

**Answer framework (2 minutes):**
“I read Revolut’s PRAGMA paper — their foundation model for banking event sequences, published April 2026
in collaboration with NVIDIA. The paper explicitly states that PRAGMA fails on AML detection by 47.1%
compared to their production baseline, because it processes user histories in isolation and can’t capture
cross-account network signals. The paper then says the next frontier is a graph-temporal approach.

So I built exactly that. PRAGMA-G faithfully reproduces the PRAGMA-S encoder architecture on the IBM AML
public dataset — the same key-value-time tokenisation, two-branch design, masked modelling pre-training —
and then grafts a three-layer GraphSAGE encoder on top of the temporal embeddings to capture
account-to-account transaction graph signals.

The result: the graph layer recovers the AML performance gap. PRAGMA-Mini alone underperforms XGBoost
by [X]%, matching what the paper reports. PRAGMA-G with graph embeddings matches or exceeds the XGBoost
baseline. I’ve deployed the full pipeline as a live API on Hugging Face Spaces — you can call it right
now at [URL] — with SHAP explainability, Evidently drift monitoring, and a What-If simulator.”

**What this shows:** Paper literacy, end-to-end ownership, production thinking, fintech domain depth.

-----

### Q2: Why GraphSAGE and not GAT or GCN?

**Answer:**
“The production constraint. GraphSAGE is inductive — it generalises to nodes unseen during training
using a neighbourhood aggregation function that doesn’t require the full graph at inference time.
For a real AML system, new accounts are created continuously. GAT and standard GCN are transductive —
they need the full adjacency matrix at inference, which means retraining or full-graph inference
every time a new account appears. That’s not deployable.

GraphSAGE with mean aggregation also scales naturally to high-degree hub nodes — which is exactly
what fan-in laundering patterns create — without the attention score instability that GAT can show
under extreme degree imbalance.”

-----

### Q3: Why does PRAGMA fail on AML specifically?

**Answer (cite the paper directly):**
“The paper identifies two factors. First, the AML dataset is large enough for task-specific models
to learn robust representations without needing foundation-level pre-training — so the transfer
advantage disappears. Second, and more critically, AML is inherently relational. A transaction that
looks innocent in isolation — say, $500 from account A to account B — becomes suspicious when
account B is sending $490 to C, $490 to D, and $490 to E in the next 10 minutes.

PRAGMA processes each account’s event history independently. It has no mechanism to see that B is a
structuring node in a fan-in/fan-out pattern. That’s exactly what the GNN layer captures: the 2-3 hop
neighbourhood around an account surfaces these multi-hop laundering cycles.”

-----

## Part 2 — Technical depth questions

### Q4: How does the key-value-time tokenisation work, and why not just use text?

**Answer:**
“Standard text serialisation has two problems for financial data. First, it inflates sequence length —
every field name and delimiter becomes several subword tokens, so a transaction record with 10 fields
might expand to 80+ tokens instead of 20. Second, numerical values get split into digit fragments
that lose magnitude and ordering — ‘1000’ and ‘100’ share the token ‘1’, ‘0’, ‘0’, but mean
very different things.

PRAGMA’s KVT scheme addresses this directly. Each field becomes exactly one key token plus
one-to-few value tokens. Numerical values are percentile-bucketed — so $1000 maps to bucket 87
and $100 maps to bucket 42, preserving ordinal structure. The vocabulary is ~60 key tokens plus
~28K value tokens — much smaller than an LLM vocabulary while being richer for tabular financial data.”

-----

### Q5: How did you handle the MLM pre-training on a public dataset without Revolut’s 26M user records?

**Answer:**
“I used the IBM AML HI-Small dataset — 5K accounts, 5M transactions — which, while much smaller,
still gives the model enough signal to learn useful representations. The key insight from the paper
is that for AML specifically, the dataset size advantage of foundation pre-training is less important
than the graph structure. The paper itself shows the task-specific XGBoost baseline outperforms
PRAGMA-Mini on AML — so the 10M-scale pre-training is sufficient to produce useful initial embeddings
that the graph layer then enriches.

I also used sequence packing — batching multiple short account histories into a single sequence
to maximise GPU utilisation on limited compute — exactly as described in §2.4 of the paper.”

-----

### Q6: How do you ensure no temporal data leakage?

**Answer:**
“This is critical and easy to get wrong. The split is always by timestamp — never random shuffle.
I sort all transactions by timestamp, take the first 60% for training, next 20% for validation,
final 20% for test. For the graph, I only include edges from the training period when computing
training node features, validation period edges when computing validation features, and so on.

This prevents the model from seeing ‘future’ transaction patterns — like knowing that account B
eventually becomes part of a laundering ring — when making predictions about earlier transactions.”

-----

### Q7: Walk me through your inference pipeline. What is the p95 latency and how did you achieve it?

**Answer:**
“The pipeline has three stages. First, the input transaction is tokenised using the pre-built
vocabulary — this is just dictionary lookups, sub-millisecond. Second, PRAGMA-Mini runs the
three-encoder stack. At PRAGMA-S scale (10M params), a single record takes ~15ms on CPU with
PyTorch inference mode and torch.no_grad(). Third, GraphSAGE runs 2-hop neighbourhood sampling —
this adds ~20ms for average node degree in IBM AML.

Total: ~40ms model compute. FastAPI adds ~5ms overhead. I run a warm-up pass of 10 inferences
on startup to trigger PyTorch’s JIT compilation. Measured p95 latency is around 47ms cold-warm,
consistently under 100ms. I track this with Prometheus histograms and have a Grafana panel
showing latency percentiles in real time.”

-----

### Q8: How does your What-If simulator work?

**Answer:**
“I adapted the What-If simulator I built for Olea Intelligence — an insurance XAI engine I built
previously. The simulator accepts a modified version of the original transaction — change the amount,
payment format, or counterparty — and returns the new score plus the change in SHAP values.

For PRAGMA-G specifically, it also shows the graph neighbourhood impact: if you change the
counterparty account, the simulator recomputes the 2-hop GraphSAGE aggregation from the new
neighbourhood and shows how the risk changes. This is particularly useful for compliance analysts
who want to understand ‘what if this was a legitimate wire instead of Bitcoin?’”

-----

## Part 3 — Revolut-specific questions

### Q9: Revolut values “Think Deeper, Never Settle.” How does this project reflect that?

**Answer:**
“Most people who read the PRAGMA paper would implement it as described and call it a day.
I read the limitations section and built the extension the authors themselves said was needed.
That’s ‘Think Deeper’ — not just understanding the contribution, but understanding the gap
and working toward closing it.

‘Never Settle’ shows up in the production approach: I didn’t stop at a Jupyter notebook.
It’s a live API with drift monitoring, CI/CD, and SHAP explainability. Every week I
committed to making it more production-grade, not just more accurate.”

-----

### Q10: Revolut’s ML engineers are “end-to-end practitioners.” Demonstrate that.

**Answer:**
“I owned every layer of this project:

- Research: read and understood arXiv:2604.08649 in detail
- Data: downloaded IBM AML, built the temporal split, constructed the PyG graph
- Model: implemented PRAGMA-S tokenisation, three-encoder architecture, GraphSAGE fusion
- Training: MLM pre-training loop, LoRA fine-tuning, MLflow tracking
- Serving: FastAPI with SHAP explainability, sub-100ms p95 latency
- MLOps: GitHub Actions CI/CD, Evidently drift monitoring, Prometheus/Grafana
- Deployment: Docker, Hugging Face Spaces, UptimeRobot keep-alive
- Communication: blog post, architecture diagram, model card with EU AI Act notes

That’s the full lifecycle, owned by one person.”

-----

## Part 4 — System design questions (common at Revolut)

### Q11: Design a real-time AML system for 10M daily transactions.

**Answer framework:**
“At 10M transactions/day, that’s ~116 transactions/second sustained. The architecture I’d propose:

**Ingestion:** Kafka/Redpanda cluster (3 brokers for HA). Transactions published as events by payment processing service.

**Feature computation:** Apache Flink or Spark Streaming for real-time velocity features (tx count in last 1hr, 24hr, 7d per account). Features written to Redis (online store) with TTL.

**Scoring:** PRAGMA-G behind a FastAPI service, horizontally scaled (5 replicas for 116 tx/s at 50ms each = 580 tx/s capacity, 5× headroom). Load balanced by account_id hash for cache locality.

**Decision:** Rules engine wraps the score. Configurable thresholds in a config service, not hardcoded.

**Output:** Score + decision written to alert topic. Compliance team UI consumes flagged events.

**Monitoring:** Evidently on feature distributions (drift = model performance degrades). Prometheus for latency/throughput. Daily model performance check against known-good labelled set.

**Retraining trigger:** PR-AUC drops >5% from baseline → automated retraining job kicks off, new model goes through staging → production registry stages before replacing production.”

-----

### Q12: How would you handle a sudden spike in false positives?

**Answer:**
“First, distinguish: is it a model problem or a data problem?

I’d check Evidently drift reports — if feature distributions have shifted (e.g., a new payment partner that creates unusual Amount patterns), that’s a data drift issue, not a model bug.

If it’s a model problem (no drift detected, but FP rate spiked), check if the threshold version changed, or if a new laundering pattern appeared that the model hasn’t seen.

Short-term: raise the flag threshold from 0.70 to 0.85 via the config service — no redeployment needed. This reduces FPs at the cost of some FN coverage.

Medium-term: collect the false-positive cases, label them as negatives, retrain or fine-tune.

Long-term: add a human-review tier at 0.5–0.7 rather than a binary flag/clear decision.”

-----

## Part 5 — Behavioural questions

### Q13: Tell me about a time you had to learn something deeply technical in a short time.

**Answer:**
“Reading the PRAGMA paper and implementing it faithfully. I had no prior experience with
Transformer architectures for tabular financial data. I spent week 1 just on the tokenisation
scheme — understanding why percentile bucketing preserves ordinal structure better than
standard normalisation, and why RoPE is better than sinusoidal for irregular time deltas.

I built unit tests for every component before moving to the next one. When the MLM loss
wasn’t decreasing, I found a bug in my within-field positional embedding — I was adding
absolute positions instead of relative ones, breaking the paper’s design. The test suite
caught it within 20 minutes of adding the history encoder.”

-----

### Q14: How do you balance speed and code quality?

**Answer:**
“For this project: type hints on every function, test for every module, no committed code
without a passing CI run. But I’m not perfect — in week 3 I shipped the event encoder
without edge case tests for accounts with zero events, and the API crashed on the first
empty-history request. I added the edge case test that week and it’s been in CI since.

In a production context: I’d follow the principle that correctness gates always block merge,
but style gates (format, lint) are warnings-only until the PR is mature. Speed without
quality creates on-call incidents. I’ve seen that at Sagemcom — a pipeline with no tests
that ran nightly. When it broke at 2am, no one knew where to start debugging.”

-----

### Q15: What would you do differently if you built PRAGMA-G again?

**Answer:**
“Three things. First, I’d start with the graph construction before the model — the IBM AML
graph structure dictates a lot of design choices I had to backtrack on in week 4.

Second, I’d use LoRA from the start rather than adding it in week 9. Pre-training and
fine-tuning the full PRAGMA-Mini takes 3× longer than LoRA fine-tuning, and the paper
shows LoRA matches or beats full fine-tuning consistently.

Third, I’d write the model card and EU AI Act compliance notes in week 1 as a design
constraint, not in week 11 as documentation. Writing ‘this model must return SHAP values
for every decision’ as a requirement up front would have shaped the API design earlier.”

-----

## Quick reference — numbers to memorise for interviews

|Metric          |Value                         |Context                    |
|----------------|------------------------------|---------------------------|
|PRAGMA paper    |arXiv:2604.08649, Apr 2026    |Revolut Research + NVIDIA  |
|PRAGMA AML gap  |47.1% degradation vs baseline |The gap we fix             |
|PRAGMA-S params |10M                           |d_model=192, 5 event layers|
|IBM AML HI-Small|~5K accounts, ~5M transactions|~5% illicit ratio          |
|API latency     |<100ms p95                    |After warm-up on HF Spaces |
|GraphSAGE layers|3                             |2-hop neighbourhood        |
|Training split  |60/20/20 by timestamp         |Temporal, no leakage       |
|Primary metric  |PR-AUC                        |Handles class imbalance    |
|SHAP values     |Top-5 per decision            |EU AI Act Article 13       |
|Weekly hours    |20+ hrs/week                  |12-week build              |
