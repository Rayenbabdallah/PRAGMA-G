"""MLM pre-training loop for PRAGMA-Mini (paper §2.3.5).

Usage:
    python -m src.training.pretrain --config configs/pragma_s.yaml

If `--data` is missing (e.g. IBM AML data not downloaded yet), falls back to
a small synthetic IBM-AML-shaped dataset so the training loop, MLflow
logging, and checkpointing can be smoke-tested end to end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model.pragma_mini import PRAGMAMini, PRAGMAMiniConfig
from src.tokenizer.vocab import MASK, Vocab
from src.training.dataset import (
    AccountHistoryDataset,
    collate_histories,
    make_synthetic_transactions_df,
)
from src.training.losses import MLMLoss, apply_mlm_masking


def load_transactions(data_path: Path) -> tuple[pd.DataFrame, bool]:
    """Returns (transactions_df, is_synthetic)."""
    if data_path.exists():
        return pd.read_csv(data_path), False
    return make_synthetic_transactions_df(n=2000, n_accounts=50), True


def run_pretrain(
    config: PRAGMAMiniConfig,
    transactions_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    max_events: int,
    learning_rate: float,
    mask_prob: float,
    device: str = "cpu",
    tracking_uri: str | None = None,
    experiment_name: str = "pragma-g-aml",
    checkpoint_dir: Path | None = None,
) -> list[float]:
    """Run MLM pre-training; returns the per-epoch mean loss."""
    vocab = Vocab().build(
        transactions_df,
        key_vocab_size=config.key_vocab_size,
        value_vocab_size=config.value_vocab_size,
    )
    mask_token_id = vocab.special2id[MASK]

    dataset = AccountHistoryDataset(transactions_df, vocab, max_events=max_events)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_histories(batch, vocab, max_events),
    )

    model = PRAGMAMini(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mlm_loss_fn = MLMLoss()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    epoch_losses: list[float] = []
    with mlflow.start_run():
        mlflow.log_params(
            {
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "profile_layers": config.profile_layers,
                "event_layers": config.event_layers,
                "history_layers": config.history_layers,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_events": max_events,
                "learning_rate": learning_rate,
                "mask_prob": mask_prob,
            }
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                mlm = apply_mlm_masking(
                    batch["event_value_ids"],
                    mask_token_id=mask_token_id,
                    padding_mask=batch["event_padding_mask"],
                    mask_prob=mask_prob,
                )
                model_input = dict(batch)
                model_input["event_value_ids"] = mlm.masked_value_ids

                output = model(model_input)
                loss = mlm_loss_fn(output["mlm_logits"], batch["event_value_ids"], mlm.mask)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            mean_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(mean_loss)
            mlflow.log_metric("mlm_loss", mean_loss, step=epoch)
            print(f"epoch {epoch + 1}/{epochs}  mlm_loss={mean_loss:.4f}")

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / "pragma_mini.pt"
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifact(str(ckpt_path))

    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train PRAGMA-Mini with MLM")
    parser.add_argument("--config", type=Path, default=Path("configs/pragma_s.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/HI-Small_Trans.csv"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("mlruns/checkpoints"))
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local ./mlruns store)",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        raw_config = yaml.safe_load(f)

    config = PRAGMAMiniConfig.from_yaml(args.config)
    pretrain_cfg = raw_config["pretrain"]
    mlflow_cfg = raw_config["mlflow"]

    transactions_df, is_synthetic = load_transactions(args.data)
    if is_synthetic:
        print(f"WARNING: {args.data} not found — using synthetic dev data.")

    run_pretrain(
        config,
        transactions_df,
        epochs=args.epochs or pretrain_cfg["epochs"],
        batch_size=args.batch_size or pretrain_cfg["batch_size"],
        max_events=args.max_events,
        learning_rate=pretrain_cfg["learning_rate"],
        mask_prob=pretrain_cfg["mlm_mask_prob"],
        tracking_uri=args.tracking_uri,
        experiment_name=mlflow_cfg["experiment_name"],
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
