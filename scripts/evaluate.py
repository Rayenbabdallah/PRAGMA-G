"""CLI: evaluate a fine-tuned PRAGMA-G checkpoint on the test split (PLAN.md Week 5).

Loads `pragma_mini_lora.pt` + `classifier.pt` from `--checkpoint-dir` (written
by `src.training.finetune`), re-runs the model on the test-split accounts,
and writes PR-AUC, ROC-AUC, precision@recall=0.5, the cost metric, the
confusion matrix, and the precision-recall curve to
`results/eval_{timestamp}.json`.

Usage:
    python scripts/evaluate.py --config configs/pragma_s.yaml \
        --checkpoint-dir mlruns/checkpoints
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.graph.graph_builder import build_transaction_graph
from src.model.classifier import PRAGMAGClassifier
from src.model.pragma_mini import PRAGMAMini, PRAGMAMiniConfig
from src.tokenizer.vocab import Vocab
from src.training.dataset import compute_node_embeddings
from src.training.finetune import build_lora_target_modules
from src.training.metrics import compute_metrics
from src.training.pretrain import load_transactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PRAGMA-G checkpoint on the test split"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pragma_s.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/HI-Small_Trans.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mlruns/checkpoints"))
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    config = PRAGMAMiniConfig.from_yaml(args.config)
    graph_cfg = raw_config["graph"]

    transactions_df, is_synthetic = load_transactions(args.data)
    if is_synthetic:
        print(f"WARNING: {args.data} not found — using synthetic dev data.")

    vocab = Vocab().build(
        transactions_df,
        key_vocab_size=config.key_vocab_size,
        value_vocab_size=config.value_vocab_size,
    )
    data = build_transaction_graph(transactions_df, vocab=vocab, d_model=config.d_model)
    labels = data.y.numpy()
    test_mask = data.test_mask.numpy()

    pragma_mini = PRAGMAMini(config)
    lora_config = LoraConfig(
        r=graph_cfg["lora"]["r"],
        lora_alpha=graph_cfg["lora"]["alpha"],
        target_modules=build_lora_target_modules(config.history_layers),
    )
    # peft's type stubs expect a transformers `PreTrainedModel`, but
    # `get_peft_model` works with any `nn.Module` at runtime.
    peft_mini = get_peft_model(pragma_mini, lora_config)  # type: ignore[arg-type]

    classifier = PRAGMAGClassifier(
        d_model=config.d_model,
        graph_hidden_channels=graph_cfg["hidden_channels"],
        graph_n_layers=graph_cfg["n_layers"],
        graph_aggregation=graph_cfg["aggregation"],
        dropout=graph_cfg["dropout"],
    )

    peft_mini.load_state_dict(torch.load(args.checkpoint_dir / "pragma_mini_lora.pt"))
    classifier.load_state_dict(torch.load(args.checkpoint_dir / "classifier.pt"))

    peft_mini.eval()
    classifier.eval()
    with torch.no_grad():
        node_embeddings = compute_node_embeddings(
            peft_mini,
            transactions_df,
            vocab,
            data.account_to_idx,
            d_model=config.d_model,
            max_events=args.max_events,
        )
        logits = classifier(node_embeddings, data.edge_index, data.edge_attr)
        scores = torch.sigmoid(logits).numpy()

    test_scores = scores[test_mask]
    test_labels = labels[test_mask]

    metrics = compute_metrics(test_labels, test_scores)
    preds = (test_scores >= 0.5).astype(int)
    cm = confusion_matrix(test_labels, preds).tolist()
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)

    results = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.results_dir / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {out_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
