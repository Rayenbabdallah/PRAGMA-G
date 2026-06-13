# PRAGMA-G — Architecture Specification

> Source: arXiv:2604.08649 (Revolut Research + NVIDIA, Apr 2026)
> This document is the implementation reference. Every design choice traces back to the paper.

-----

## 1. Overview

PRAGMA-G = PRAGMA-Mini (faithful reproduction of §2.2–2.3) + GraphSAGE extension (our contribution).

The paper’s core claim: “AML detection is inherently relational. Because PRAGMA processes event
histories in isolation, it cannot capture network-level signals — resulting in 47.1% degradation.”

Our extension adds exactly the missing component: a GraphSAGE encoder that operates on the
account-to-account transaction graph, using PRAGMA-Mini embeddings as node features.

-----

## 2. Tokenisation (PRAGMA paper §2.2)

### 2.1 Key-Value-Time representation

Every event field is represented as a triple: (key, value, temporal_coordinate).

Example from the paper:

```
Channel: email at 24-04-07 19:20:18
→ key_id: 7 (index of "Channel" in key vocab)
→ value_id: 142 (index of "email" in value vocab)
→ temporal: [log_seconds=..., hour=19, dow=0, dom=7]
```

### 2.2 Key vocabulary

- All field names as single tokens
- ~60 tokens total (per paper §2.2)
- Same vocab for both event fields and profile state fields
- IBM AML field mapping:
  
  ```
  "From Bank"          → key: "from_bank"
  "Account" (sender)   → key: "sender_account"
  "To Bank"            → key: "to_bank"
  "Account.1" (recv)   → key: "receiver_account"
  "Amount Received"    → key: "amount_received"
  "Receiving Currency" → key: "receiving_currency"
  "Amount Paid"        → key: "amount_paid"
  "Payment Currency"   → key: "payment_currency"
  "Payment Format"     → key: "payment_format"
  "Timestamp"          → key: "created" (maps to temporal coordinate)
  ```

### 2.3 Value vocabulary (~28K tokens)

Three value types, determined by field type:

**Numerical values** — percentile bucketing:

```python
# Build during vocabulary construction on training data
boundaries = np.percentile(values, np.linspace(0, 100, N_BUCKETS))
# Extra bucket for zero (zero-inflated financial amounts)
bucket_id = np.searchsorted(boundaries, value)
# Special token for zero: ZERO_BUCKET_ID
```

- N_BUCKETS: 100 (one token per percentile)
- Zero gets its own dedicated bucket
- One token per numerical field value

**Categorical values** — single token if cardinality < threshold:

```python
CATEGORICAL_THRESHOLD = 1000  # fields with < 1000 unique values
# Payment Format: wire, check, credit_card, ACH, cash → 5 tokens
# Currency: USD, EUR, GBP, etc. → ~50 tokens
# Bank IDs: treated as categorical (capped at threshold)
```

**Textual values** — BPE subword tokeniser:

```python
# Fields with cardinality >= threshold
# Uses sentencepiece or tokenizers library
# Reserved [UNK] for rare fragments
# Vocab size: ~28K - 60 (keys) - 100*n_numerical (buckets) - n_categorical
```

### 2.4 Temporal encoding (paper §2.2, exactly)

**Log-seconds transform:**

```python
def log_time_transform(seconds_since_last_event: float) -> float:
    """Soft log transform from PRAGMA paper §2.2.
    Compresses dynamic range for life-long events while preserving
    linear granularity for recent events."""
    return 8.0 * math.log(1.0 + seconds_since_last_event / 8.0)
```

**Calendar features** (for event-history entries only, not life-long events):

```python
# Periodic functions with fixed periods (not learned)
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
dow_sin = sin(2π * day_of_week / 7)
dow_cos = cos(2π * day_of_week / 7)
dom_sin = sin(2π * day_of_month / 31)
dom_cos = cos(2π * day_of_month / 31)
# → 6-dim calendar embedding, projected to d_model
```

-----

## 3. PRAGMA-Mini Architecture (§2.3)

### 3.1 Model family — use PRAGMA-S scale

|Model   |Params|d_model|d_ffn|Profile layers|Event layers|History layers|Heads|
|--------|------|-------|-----|--------------|------------|--------------|-----|
|PRAGMA-S|10M   |192    |768  |1             |5           |2             |3    |
|PRAGMA-M|100M  |512    |2048 |3             |16          |6             |8    |
|PRAGMA-L|1B    |1024   |4096 |9             |45          |18            |16   |

**Use PRAGMA-S (10M params) for this project.** Trainable on a single GPU or even CPU.

All variants use:

- GELU activations (Hendrycks & Gimpel, 2016)
- Pre-norm layer normalisation (Xiong et al., 2020)
- Dropout: 0.1
- RoPE (Rotary Position Embedding) for temporal positions

### 3.2 Token Embedding (§2.3.1)

```python
class TokenEmbedding(nn.Module):
    def __init__(self, key_vocab_size, val_vocab_size, d_model):
        self.key_embed = nn.Embedding(key_vocab_size, d_model)
        self.val_embed = nn.Embedding(val_vocab_size, d_model)
        # Within-field positional embedding (for multi-token value fields)
        self.within_field_pos = nn.Embedding(MAX_VALUE_TOKENS_PER_FIELD, d_model)

    def forward(self, key_ids, value_ids, within_field_positions):
        # key_ids: (B, T) — one key per token
        # value_ids: (B, T) — one value per token (multi-token fields expand T)
        # Returns: (B, T, d_model) combined embedding
        return self.key_embed(key_ids) + self.val_embed(value_ids) + \
               self.within_field_pos(within_field_positions)
```

### 3.3 Profile State Encoder (§2.3.2)

Encodes static/slow-changing account attributes into a single [USR] embedding.

```python
class ProfileStateEncoder(nn.Module):
    """
    Input: profile_tokens (B, T_profile, d_model) — key-value tokens for profile fields
           time_deltas (B, T_profile) — time since life-long events (log-transformed)
    Output: usr_embedding (B, d_model) — [USR] token representation
    """
    def __init__(self, d_model=192, n_layers=1, n_heads=3, d_ffn=768, dropout=0.1):
        self.rope = RotaryPositionEmbedding(d_model)  # RoPE for life-long event timing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout,
                                       activation='gelu', norm_first=True),  # pre-norm
            num_layers=n_layers
        )
        self.usr_token = nn.Parameter(torch.zeros(1, 1, d_model))  # learnable [USR]

    def forward(self, profile_tokens, time_deltas):
        # Prepend learnable [USR] token
        B = profile_tokens.shape[0]
        usr = self.usr_token.expand(B, -1, -1)
        x = torch.cat([usr, profile_tokens], dim=1)
        # Apply RoPE using time_deltas (prepend 0 for [USR])
        x = self.rope(x, torch.cat([torch.zeros(B, 1), time_deltas], dim=1))
        x = self.transformer(x)
        return x[:, 0, :]  # return [USR] position output: (B, d_model)
```

### 3.4 Event Encoder (§2.3.3)

Encodes each event independently into an [EVT] embedding, then adds calendar features.

```python
class EventEncoder(nn.Module):
    """
    Processes each event record independently (no cross-event attention here).
    Input: event_tokens (B, T_events, T_fields, d_model)
    Output: evt_embeddings (B, T_events, d_model) + calendar_features
    """
    def __init__(self, d_model=192, n_layers=5, n_heads=3, d_ffn=768, dropout=0.1):
        self.evt_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.within_event_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout,
                                       activation='gelu', norm_first=True),
            num_layers=n_layers
        )
        self.calendar_proj = nn.Linear(6, d_model)  # project 6-dim calendar to d_model

    def forward(self, event_tokens, calendar_features):
        # event_tokens: (B, T_events, T_fields, d_model)
        B, T_events, T_fields, D = event_tokens.shape
        # Process each event independently
        x = event_tokens.view(B * T_events, T_fields, D)
        evt = self.evt_token.expand(B * T_events, -1, -1)
        x = torch.cat([evt, x], dim=1)
        x = self.within_event_transformer(x)
        evt_emb = x[:, 0, :].view(B, T_events, D)  # [EVT] embeddings
        # Add calendar features
        cal = self.calendar_proj(calendar_features)  # (B, T_events, d_model)
        return evt_emb + cal
```

### 3.5 History Encoder (§2.3.4)

Contextualises the full sequence [USR, EVT_1, …, EVT_T].

```python
class HistoryEncoder(nn.Module):
    """
    Input: usr_emb (B, d_model), evt_embs (B, T_events, d_model)
           time_to_last_event (B, T_events) — time delta between consecutive events
    Output: z_h (B, d_model) — final record-level embedding
    """
    def __init__(self, d_model=192, n_layers=2, n_heads=3, d_ffn=768, dropout=0.1):
        self.rope = RotaryPositionEmbedding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout,
                                       activation='gelu', norm_first=True),
            num_layers=n_layers
        )

    def forward(self, usr_emb, evt_embs, time_to_last_event):
        # Concat [USR] with event sequence
        x = torch.cat([usr_emb.unsqueeze(1), evt_embs], dim=1)  # (B, 1+T, d_model)
        # RoPE: prepend 0 time delta for [USR]
        B = x.shape[0]
        times = torch.cat([torch.zeros(B, 1, device=x.device), time_to_last_event], dim=1)
        x = self.rope(x, times)
        z = self.transformer(x)
        return z[:, 0, :]  # return [USR] position = record-level embedding z_h
```

### 3.6 MLM Pre-training objective (§2.3.5)

```python
# Randomly mask 15% of value tokens → predict original value_id
# Do NOT mask key tokens (per standard MLM practice)
# Do NOT mask the temporal coordinate
# Loss: cross-entropy over value vocabulary (~28K classes)
mask_prob = 0.15
masked_value_ids = value_ids.clone()
mask = torch.rand_like(value_ids.float()) < mask_prob
masked_value_ids[mask] = MASK_TOKEN_ID
# Forward pass → reconstruct original value_ids at masked positions
logits = mlm_head(pragma_mini(profile_tokens, masked_event_tokens))
loss = F.cross_entropy(logits[mask], value_ids[mask])
```

### 3.7 Sequence packing + dynamic batching (§2.4)

```python
# High-activity users may have 10K+ events — truncate to MAX_SEQ_LEN
MAX_SEQ_LEN = 512  # events per user (conservative for PRAGMA-S on single GPU)
# Pack multiple short user histories into a single batch element to maximise GPU utilisation
# Use attention masks to prevent cross-user attention within packed sequences
```

-----

## 4. Graph Extension — PRAGMA-G (our contribution)

### 4.1 Graph construction from IBM AML

```python
# Nodes: unique accounts (From + To accounts)
# Edges: transactions (directed: From Account → To Account)
# Node features: PRAGMA-Mini embeddings (pre-computed, cached)
# Edge features: [amount_norm, time_delta_log, payment_format_id, currency_id]

# Train/val/test split: by transaction timestamp (60/20/20)
# This prevents data leakage — future transactions never inform past predictions
```

### 4.2 GraphSAGE encoder

```python
class GraphSAGEEncoder(nn.Module):
    """
    Inductive GNN — generalises to unseen accounts at inference time.
    This is critical for production AML (new accounts appear continuously).

    Uses mean aggregation (most stable for financial graphs with high-degree hubs).
    3 layers → 2-hop neighbourhood (captures fan-in/fan-out laundering patterns).
    """
    def __init__(self, in_channels=192, hidden_channels=256, out_channels=192, n_layers=3):
        from torch_geometric.nn import SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.edge_encoder = nn.Linear(4, hidden_channels)  # 4 edge features

    def forward(self, x, edge_index, edge_attr):
        # x: (N, 192) — PRAGMA-Mini embeddings
        # edge_index: (2, E) — directed transaction edges
        # edge_attr: (E, 4) — edge features
        # Returns: (N, 192) — graph-enriched node embeddings
        edge_emb = self.edge_encoder(edge_attr)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x
```

### 4.3 Fusion + classification head

```python
class PRAGMAGClassifier(nn.Module):
    """
    Fuses PRAGMA-Mini temporal embedding with GraphSAGE graph embedding.
    Outputs AML risk score ∈ [0, 1].
    """
    def __init__(self, d_model=192):
        self.pragma_mini = PRAGMAMini(config)
        self.graphsage = GraphSAGEEncoder(d_model, 256, d_model, n_layers=3)
        # Fusion: simple concatenation + MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, profile_tokens, event_tokens, edge_index, edge_attr):
        # Temporal embedding
        z_temporal = self.pragma_mini(profile_tokens, event_tokens)  # (N, 192)
        # Graph embedding (uses z_temporal as node features)
        z_graph = self.graphsage(z_temporal, edge_index, edge_attr)  # (N, 192)
        # Fuse
        z_fused = torch.cat([z_temporal, z_graph], dim=-1)  # (N, 384)
        return torch.sigmoid(self.fusion(z_fused)).squeeze(-1)  # (N,) ∈ [0,1]
```

-----

## 5. Evaluation metrics

### Primary: PR-AUC (Precision-Recall AUC)

- Class imbalance (~5% illicit in HI-Small) makes ROC-AUC misleading
- PR-AUC reflects real detection capability under imbalance

### Secondary metrics

- ROC-AUC
- Precision@Recall=0.5 (catch half of laundering, how many false alarms?)
- Cost-based metric: `cost = FP * cost_false_alert + FN * cost_missed_laundering`
  - Typical ratio: cost_FN = 100 × cost_FP (missing laundering is far worse)

### Baseline comparison table (target for README)

|Model                          |PR-AUC|ROC-AUC|Notes                      |
|-------------------------------|------|-------|---------------------------|
|XGBoost (hand-crafted features)|TBD   |TBD    |Paper’s production baseline|
|PRAGMA-Mini only (no graph)    |TBD   |TBD    |Reproduces paper’s AML gap |
|PRAGMA-G (ours)                |TBD   |TBD    |Graph layer closes the gap |

-----

## 6. Inference pipeline (API)

```
POST /score
{
  "account_id": "ACC_1234",
  "profile": {"plan": "standard", "region": "uk", "balance_quantile": 0.6},
  "events": [
    {"type": "card_payment", "amount": 14.99, "currency": "gbp",
     "mcc": "6012", "timestamp": "2026-06-01T14:32:00"},
    ...
  ]
}

Response:
{
  "account_id": "ACC_1234",
  "score": 0.847,
  "decision": "flag",          # flag (>0.7) / review (0.3–0.7) / clear (<0.3)
  "threshold_version": "v1.2",
  "shap_values": {
    "amount_received": 0.31,   # top contributor
    "payment_format": 0.18,
    "graph_fan_in_ratio": 0.15,
    "temporal_velocity": 0.12,
    "receiving_currency": 0.09
  },
  "graph_neighbours": 3,        # accounts in 1-hop neighbourhood
  "latency_ms": 47.3,
  "model_version": "pragma-g-aml-v1"
}
```

### Thresholds (versioned, not baked into model)

```yaml
# configs/thresholds.yaml
thresholds:
  v1.2:
    flag: 0.70      # high confidence — send to compliance team
    review: 0.30    # medium confidence — queue for analyst review
    clear: 0.30     # below this — no action
```

-----

## 7. RoPE implementation reference

```python
class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021) adapted for temporal positions."""
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        # Fixed frequency basis (not learned)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, positions):
        # x: (B, T, d_model)
        # positions: (B, T) — log-transformed time deltas
        freqs = torch.einsum('bt,d->btd', positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return x * cos_emb + self._rotate_half(x) * sin_emb

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
```
