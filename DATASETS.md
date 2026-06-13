# PRAGMA-G — Dataset Reference

## Primary dataset: IBM AML (Kaggle)

**Full name:** IBM Transactions for Anti Money Laundering (AML)
**Kaggle slug:** `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
**Paper:** Altman et al., “Realistic Synthetic Financial Transactions for AML Models”, NeurIPS 2023
**License:** CDLA-Sharing-1.0 (free for research, requires attribution)
**GitHub:** <https://github.com/IBM/AML-Data>

### Why this dataset

- Explicitly designed for GNN-based AML research (the paper tests GraphSAGE, GCN, GAT on it)
- Synthetic but calibrated to real financial transaction patterns
- Ground truth labels are complete (unlike real data where many launderings go undetected)
- Multiple size variants — use HI-Small for dev, HI-Large for final training
- Referenced by 20+ AML papers; results are comparable to published work
- Cited by Revolut-adjacent researchers for AML GNN benchmarking

### Dataset variants

|Variant  |Accounts|Transactions|Illicit ratio|Use for                      |
|---------|--------|------------|-------------|-----------------------------|
|HI-Small |~5K     |~5M         |~5%          |Development, fast iteration  |
|HI-Medium|~50K    |~50M        |~5%          |Validation runs              |
|HI-Large |~500K   |~500M       |~5%          |Final training               |
|LI-Small |~5K     |~5M         |~0.1%        |Low-illicit-ratio experiments|

Start with **HI-Small** for all of weeks 1–6. Switch to HI-Large only for final evaluation.

-----

## Download instructions

```bash
# Set credentials in .env (never commit these)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Run download script
bash scripts/download_data.sh

# Or manually:
pip install kaggle
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  --path data/ --unzip
```

### Expected files after download

```
data/
├── HI-Small_Trans.csv       # ~500MB — transactions with laundering labels
├── HI-Small_Patterns.csv    # laundering pattern metadata
├── HI-Medium_Trans.csv
├── HI-Large_Trans.csv
└── README.txt               # IBM dataset documentation
```

-----

## Schema: HI-Small_Trans.csv

|Column              |Type     |Description                      |PRAGMA mapping                                         |
|--------------------|---------|---------------------------------|-------------------------------------------------------|
|`Timestamp`         |datetime |Transaction datetime             |→ temporal coordinate                                  |
|`From Bank`         |str      |Sending bank identifier          |→ key: “from_bank”, value: categorical                 |
|`Account`           |str      |Sending account ID               |→ node ID in graph                                     |
|`To Bank`           |str      |Receiving bank identifier        |→ key: “to_bank”, value: categorical                   |
|`Account.1`         |str      |Receiving account ID             |→ node ID in graph                                     |
|`Amount Received`   |float    |Amount received (target currency)|→ key: “amount_received”, value: numerical (bucketised)|
|`Receiving Currency`|str      |Currency of received amount      |→ key: “receiving_currency”, value: categorical        |
|`Amount Paid`       |float    |Amount sent (source currency)    |→ key: “amount_paid”, value: numerical (bucketised)    |
|`Payment Currency`  |str      |Currency of paid amount          |→ key: “payment_currency”, value: categorical          |
|`Payment Format`    |str      |Transaction method               |→ key: “payment_format”, value: categorical            |
|`Is Laundering`     |int (0/1)|Ground truth AML label           |→ training target                                      |

### Payment Format values (categorical, ~6 unique)

- Reinvestment
- Wire
- Cheque
- Credit Card
- Cash
- Bitcoin

### Currency values (categorical, ~10+ unique)

- US Dollar, Euro, British Pounds, Bitcoin, Swiss Franc, etc.

-----

## Graph construction from IBM AML

### Node definition

```
Node = unique account ID (union of Account and Account.1 columns)
Node features = PRAGMA-Mini embedding (computed per account, cached)
Node label = 1 if any transaction involving this account is laundering, else 0
```

### Edge definition

```
Edge = (From Account → To Account) for each transaction
Edge features:
  - amount_norm: Amount Paid normalised to log-scale
  - time_delta_log: log(1 + seconds_since_account_first_seen / 8) * 8
  - payment_format_id: integer encoding of Payment Format
  - currency_id: integer encoding of Payment Currency

Multiple edges can exist between same pair of accounts (multigraph)
PyG supports multigraphs natively
```

### Temporal train/val/test split

```python
# Split by timestamp — NEVER by random shuffle (causes temporal leakage)
timestamps = sorted(df['Timestamp'].unique())
train_cutoff = timestamps[int(len(timestamps) * 0.6)]
val_cutoff   = timestamps[int(len(timestamps) * 0.8)]

train_df = df[df['Timestamp'] < train_cutoff]
val_df   = df[(df['Timestamp'] >= train_cutoff) & (df['Timestamp'] < val_cutoff)]
test_df  = df[df['Timestamp'] >= val_cutoff]
```

### Laundering typologies in the data

The IBM generator creates realistic laundering patterns:

- **Fan-out:** One account sends to many recipients (structuring)
- **Fan-in:** Many accounts send to one (aggregation)
- **Cycle:** Money flows A→B→C→A (layering)
- **Scatter-gather:** Fan-out followed by fan-in
  These patterns are exactly what GraphSAGE’s 2-3 hop neighbourhood captures.

-----

## Secondary dataset: PaySim (streaming simulation)

**Kaggle slug:** `ealaxi/paysim1`
**Use:** Streaming event producer only (Redpanda/Kafka feed in week 7)
**Do NOT use for model training** — balance columns leak the label

### Why PaySim for streaming only

- 6.3M transactions, 744 time steps (30-day simulation)
- Replay at 1 step/10ms = realistic transaction rate
- Different domain (mobile money / M-Pesa style) — IBM AML is primary
- Good for testing API throughput and drift detection logic

-----

## Data versioning

```bash
# Track data checksums (not the data itself) in git
sha256sum data/HI-Small_Trans.csv > data/checksums.sha256
git add data/checksums.sha256
```

-----

## Class imbalance handling

HI-Small has ~5% illicit transactions. Options:

1. **Positive class weight** in BCEWithLogitsLoss: `pos_weight = (N_neg / N_pos)`
   
   ```python
   pos_weight = torch.tensor([19.0])  # 95/5 ratio
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```
1. **Undersampling** the majority class during batch construction
1. **PR-AUC** as primary metric (not accuracy, not ROC-AUC)

Use option 1 + option 3 as the standard approach.

-----

## Data loading code skeleton

```python
import pandas as pd
import torch
from torch_geometric.data import Data

def load_ibm_aml(path: str, split: str = "train") -> Data:
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp')

    # Temporal split
    n = len(df)
    if split == "train":
        df = df.iloc[:int(n * 0.6)]
    elif split == "val":
        df = df.iloc[int(n * 0.6):int(n * 0.8)]
    else:
        df = df.iloc[int(n * 0.8):]

    # Build account index
    accounts = pd.unique(df[['Account', 'Account.1']].values.ravel())
    acc2idx = {a: i for i, a in enumerate(accounts)}

    # Edge index (directed)
    src = df['Account'].map(acc2idx).values
    dst = df['Account.1'].map(acc2idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge features
    edge_attr = torch.tensor([
        df['Amount Paid'].apply(lambda x: 8 * np.log(1 + x / 8)).values,
        # ... other edge features
    ], dtype=torch.float).T

    # Labels (transaction-level → account-level: flag if ANY tx is laundering)
    labels = df.groupby('Account')['Is Laundering'].max()
    y = torch.tensor([labels.get(a, 0) for a in accounts], dtype=torch.float)

    return Data(edge_index=edge_index, edge_attr=edge_attr, y=y,
                num_nodes=len(accounts))
```
