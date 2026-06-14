"""Two-stage AML fine-tuning for PRAGMA-G (ARCHITECTURE.md §4.3, PLAN.md Week 5).

Stage 1: freeze PRAGMA-Mini, train the GraphSAGE encoder + fusion head on
frozen per-account temporal embeddings.
Stage 2: LoRA fine-tune PRAGMA-Mini jointly with the GraphSAGE + fusion head.

Note on LoRA target modules: `configs/pragma_s.yaml` specifies
`target_modules: ["query", "value"]` per the paper's attention-projection
convention, but `nn.TransformerEncoderLayer`'s `nn.MultiheadAttention` uses a
single fused `in_proj_weight` with no `query`/`value` submodules to target.
We instead apply LoRA to the History Encoder's feed-forward `linear1`/
`linear2` layers, which are the closest available `nn.Linear` targets and
keep the adapter scoped to the History Encoder as intended.

Usage:
    python -m src.training.finetune --config configs/pragma_s.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from src.graph.graph_builder import build_transaction_graph
from src.model.classifier import PRAGMAGClassifier
from src.model.pragma_mini import PRAGMAMini, PRAGMAMiniConfig
from src.tokenizer.vocab import Vocab
from src.training.dataset import compute_node_embeddings
from src.training.metrics import compute_metrics
from src.training.pretrain import load_transactions
from src.training.registry import register_checkpoint


def build_lora_target_modules(history_layers: int) -> list[str]:
    """History Encoder feed-forward `linear1`/`linear2` for each layer."""
    modules = []
    for i in range(history_layers):
        modules.append(f"history_encoder.transformer.layers.{i}.linear1")
        modules.append(f"history_encoder.transformer.layers.{i}.linear2")
    return modules


def run_finetune(
    config: PRAGMAMiniConfig,
    transactions_df: pd.DataFrame,
    graph_config: dict[str, Any],
    stage_1_epochs: int,
    stage_2_epochs: int,
    learning_rate: float,
    pos_weight: float,
    lora_r: int,
    lora_alpha: int,
    max_events: int = 64,
    device: str = "cpu",
    tracking_uri: str | None = None,
    experiment_name: str = "pragma-g-aml",
    checkpoint_dir: Path | None = None,
    use_graph: bool = True,
    register_model: bool = False,
    model_registry_name: str = "pragma-g-aml",
    registry_stage: str = "Staging",
) -> dict[str, float]:
    """Runs the two-stage fine-tuning loop; returns test-split metrics."""
    vocab = Vocab().build(
        transactions_df,
        key_vocab_size=config.key_vocab_size,
        value_vocab_size=config.value_vocab_size,
    )
    data = build_transaction_graph(transactions_df, vocab=vocab, d_model=config.d_model)
    data = data.to(device)

    labels = data.y
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    pragma_mini = PRAGMAMini(config).to(device)
    classifier = PRAGMAGClassifier(
        d_model=config.d_model,
        graph_hidden_channels=graph_config["hidden_channels"],
        graph_n_layers=graph_config["n_layers"],
        graph_aggregation=graph_config["aggregation"],
        dropout=graph_config["dropout"],
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "d_model": config.d_model,
                "stage_1_epochs": stage_1_epochs,
                "stage_2_epochs": stage_2_epochs,
                "learning_rate": learning_rate,
                "pos_weight": pos_weight,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "use_graph": use_graph,
            }
        )

        # --- Stage 1: freeze PRAGMA-Mini, train GraphSAGE + fusion head ---
        for p in pragma_mini.parameters():
            p.requires_grad_(False)
        pragma_mini.eval()

        with torch.no_grad():
            frozen_embeddings = compute_node_embeddings(
                pragma_mini,
                transactions_df,
                vocab,
                data.account_to_idx,
                d_model=config.d_model,
                max_events=max_events,
                device=device,
            )

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
        for epoch in range(stage_1_epochs):
            classifier.train()
            logits = classifier(
                frozen_embeddings, data.edge_index, data.edge_attr, use_graph=use_graph
            )
            loss = loss_fn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(
                    frozen_embeddings, data.edge_index, data.edge_attr, use_graph=use_graph
                )
                val_scores = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_metrics = compute_metrics(labels[val_mask].cpu().numpy(), val_scores)

            mlflow.log_metric("stage1_train_loss", loss.item(), step=epoch)
            mlflow.log_metrics({f"stage1_val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            print(f"[stage 1] epoch {epoch + 1}/{stage_1_epochs}  loss={loss.item():.4f}")

        # --- Stage 2: LoRA fine-tune PRAGMA-Mini jointly with the head ---
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=build_lora_target_modules(config.history_layers),
        )
        # peft's type stubs expect a transformers `PreTrainedModel`, but
        # `get_peft_model` works with any `nn.Module` at runtime.
        peft_mini = get_peft_model(pragma_mini, lora_config)  # type: ignore[arg-type]

        trainable_params = [p for p in peft_mini.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params + list(classifier.parameters()), lr=learning_rate
        )

        for epoch in range(stage_2_epochs):
            peft_mini.train()
            classifier.train()

            node_embeddings = compute_node_embeddings(
                peft_mini,
                transactions_df,
                vocab,
                data.account_to_idx,
                d_model=config.d_model,
                max_events=max_events,
                device=device,
            )
            logits = classifier(
                node_embeddings, data.edge_index, data.edge_attr, use_graph=use_graph
            )
            loss = loss_fn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            peft_mini.eval()
            classifier.eval()
            with torch.no_grad():
                node_embeddings = compute_node_embeddings(
                    peft_mini,
                    transactions_df,
                    vocab,
                    data.account_to_idx,
                    d_model=config.d_model,
                    max_events=max_events,
                    device=device,
                )
                val_logits = classifier(
                    node_embeddings, data.edge_index, data.edge_attr, use_graph=use_graph
                )
                val_scores = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_metrics = compute_metrics(labels[val_mask].cpu().numpy(), val_scores)

            mlflow.log_metric("stage2_train_loss", loss.item(), step=epoch)
            mlflow.log_metrics({f"stage2_val_{k}": v for k, v in val_metrics.items()}, step=epoch)
            print(f"[stage 2] epoch {epoch + 1}/{stage_2_epochs}  loss={loss.item():.4f}")

        # --- Final test-split evaluation ---
        peft_mini.eval()
        classifier.eval()
        with torch.no_grad():
            node_embeddings = compute_node_embeddings(
                peft_mini,
                transactions_df,
                vocab,
                data.account_to_idx,
                d_model=config.d_model,
                max_events=max_events,
                device=device,
            )
            test_logits = classifier(
                node_embeddings, data.edge_index, data.edge_attr, use_graph=use_graph
            )
            test_scores = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
            test_metrics = compute_metrics(labels[test_mask].cpu().numpy(), test_scores)

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(peft_mini.state_dict(), checkpoint_dir / "pragma_mini_lora.pt")
            torch.save(classifier.state_dict(), checkpoint_dir / "classifier.pt")
            mlflow.log_artifact(str(checkpoint_dir / "pragma_mini_lora.pt"))
            mlflow.log_artifact(str(checkpoint_dir / "classifier.pt"))

        # Log full models (architecture + weights) so the registry can hand
        # back ready-to-use `nn.Module`s — see `src.training.registry`.
        mlflow.pytorch.log_model(peft_mini, artifact_path="pragma_mini")
        mlflow.pytorch.log_model(classifier, artifact_path="classifier")

        if register_model:
            run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
            version = register_checkpoint(
                run_id, model_registry_name, tracking_uri=tracking_uri, stage=registry_stage
            )
            print(f"Registered {model_registry_name} v{version} -> {registry_stage}")

    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PRAGMA-G for AML classification")
    parser.add_argument("--config", type=Path, default=Path("configs/pragma_s.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/HI-Small_Trans.csv"))
    parser.add_argument("--stage-1-epochs", type=int, default=None)
    parser.add_argument("--stage-2-epochs", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mlruns/checkpoints"))
    parser.add_argument("--no-graph", action="store_true", help="PRAGMA-Mini-only baseline")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local ./mlruns store)",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the trained checkpoint in the MLflow model registry",
    )
    parser.add_argument(
        "--registry-stage",
        type=str,
        default="Staging",
        help="Stage to transition the registered model version to (default: Staging)",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    config = PRAGMAMiniConfig.from_yaml(args.config)
    finetune_cfg = raw_config["finetune"]
    graph_cfg = raw_config["graph"]
    mlflow_cfg = raw_config["mlflow"]

    transactions_df, is_synthetic = load_transactions(args.data)
    if is_synthetic:
        print(f"WARNING: {args.data} not found — using synthetic dev data.")

    metrics = run_finetune(
        config,
        transactions_df,
        graph_config=graph_cfg,
        stage_1_epochs=args.stage_1_epochs or finetune_cfg["stage_1_epochs"],
        stage_2_epochs=args.stage_2_epochs or finetune_cfg["stage_2_epochs"],
        learning_rate=finetune_cfg["learning_rate"],
        pos_weight=finetune_cfg["pos_weight"],
        lora_r=graph_cfg["lora"]["r"],
        lora_alpha=graph_cfg["lora"]["alpha"],
        max_events=args.max_events,
        tracking_uri=args.tracking_uri,
        experiment_name=mlflow_cfg["experiment_name"],
        checkpoint_dir=args.checkpoint_dir,
        use_graph=not args.no_graph,
        register_model=args.register_model,
        model_registry_name=mlflow_cfg["model_registry_name"],
        registry_stage=args.registry_stage,
    )
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
