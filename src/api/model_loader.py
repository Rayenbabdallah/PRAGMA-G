"""Singleton model loader + inference pipeline for PRAGMA-G serving (PLAN.md Week 6).

Loads PRAGMA-Mini (+ LoRA adapter) and the `PRAGMAGClassifier`, builds an
ad-hoc 1-hop transaction graph from a request's events, and runs the full
PRAGMA-G pipeline to produce an AML risk score with feature attributions.

Checkpoint loading order:
  1. `mlruns/checkpoints/{pragma_mini_lora.pt,classifier.pt}` if both exist
     (written by `src.training.finetune`).
  2. Otherwise, freshly-initialised weights with `model_version =
     "pragma-g-untrained-dev"` — keeps the API runnable for demos/tests
     without a full training run.

Vocabulary loading order:
  1. `data/vocab.json` if present (written by `scripts/build_vocab.py`).
  2. Otherwise, built from synthetic IBM-AML-shaped data
     (`make_synthetic_transactions_df`) with a fixed seed for determinism.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn

from src.api.schemas import EventRecord, TransactionRequest
from src.graph.graph_builder import EDGE_FEATURE_DIM
from src.model.classifier import PRAGMAGClassifier
from src.model.pragma_mini import PRAGMAMini, PRAGMAMiniConfig
from src.tokenizer.tokenizer import EncodedEvent, KVTTokenizer, log_time_transform
from src.tokenizer.vocab import Vocab
from src.training.dataset import collate_histories, make_synthetic_transactions_df
from src.training.finetune import build_lora_target_modules

CONFIG_PATH = Path("configs/pragma_s.yaml")
VOCAB_PATH = Path("data/vocab.json")
CHECKPOINT_DIR = Path("mlruns/checkpoints")

# Per-request normalisation constants for the ad-hoc edge features. Unlike
# `build_edge_features` (graph_builder.py), a single request has no
# dataset-wide max to normalise against, so fixed scales are used instead —
# chosen so that "typical" amounts/time-gaps land roughly in [0, 1].
AMOUNT_NORM_SCALE = 15.0
TIME_DELTA_NORM_SCALE = 60.0

# Features perturbed for attribution — `/score` returns the top-5 by |value|,
# `/explain` returns all of them.
EXPLAIN_FEATURES = [
    "amount_paid",
    "payment_format",
    "receiving_currency",
    "payment_currency",
    "graph_fan_in_ratio",
    "temporal_velocity",
]


@dataclass
class ScoreResult:
    score: float
    decision: str
    threshold_version: str
    shap_values: dict[str, float]
    graph_neighbours: int


@dataclass
class ExplainResult:
    score: float
    all_shap_values: dict[str, float]
    graph_neighbourhood: list[dict[str, Any]]


def _event_dict(event: EventRecord, account_id: str) -> dict[str, Any]:
    """Map an `EventRecord` to a PRAGMA key-named event dict (DATASETS.md schema)."""
    return {
        "sender_account": account_id,
        "receiver_account": event.counterparty_account,
        "from_bank": None,
        "to_bank": event.counterparty_bank,
        "amount_received": event.amount_received,
        "receiving_currency": event.receiving_currency,
        "amount_paid": event.amount,
        "payment_currency": event.currency,
        "payment_format": event.payment_format,
        "created": event.timestamp,
    }


def _categorical_id(vocab: Vocab, field: str, value: Any) -> float:
    """Normalised categorical id in `[0, 1]`, mirroring `graph_builder._categorical_feature`."""
    fv = vocab.field_vocabs.get(field)
    if fv is None or fv.value2id is None or value is None:
        return 1.0  # UNK slot is always the last one -> normalised id 1.0
    unk_id = fv.size - 1
    return fv.value2id.get(str(value), unk_id) / fv.size


def _edge_attr_for_event(
    vocab: Vocab, event: EventRecord, seconds_since_start: float
) -> list[float]:
    """Builds a single `(4,)` edge feature `[amount_norm, time_delta_log, payment_format_id,
    currency_id]`, approximating `build_edge_features` for one out-of-sample event.
    """
    amount = event.amount or 0.0
    amount_norm = min(torch.log1p(torch.tensor(float(amount))).item() / AMOUNT_NORM_SCALE, 1.0)
    time_delta_log = log_time_transform(max(seconds_since_start, 0.0))
    time_delta_log = min(time_delta_log / TIME_DELTA_NORM_SCALE, 1.0)
    payment_format_id = _categorical_id(vocab, "payment_format", event.payment_format)
    currency_id = _categorical_id(vocab, "payment_currency", event.currency)
    return [amount_norm, time_delta_log, payment_format_id, currency_id]


def _ablate_amount(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ablated = copy.deepcopy(events)
    ablated[-1]["amount_paid"] = None
    ablated[-1]["amount_received"] = None
    return ablated


def _ablate_field(events: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    ablated = copy.deepcopy(events)
    ablated[-1][field] = None
    return ablated


def _zero_time_deltas(encoded: list[EncodedEvent]) -> list[EncodedEvent]:
    return [
        EncodedEvent(
            key_ids=e.key_ids, value_ids=e.value_ids, time_delta=0.0, calendar=e.calendar
        )
        for e in encoded
    ]


class ModelLoader:
    """Singleton-style loader: instantiated once in `src.api.main`'s lifespan."""

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        vocab_path: Path = VOCAB_PATH,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ):
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.checkpoint_dir = checkpoint_dir

        self.config: PRAGMAMiniConfig | None = None
        self.raw_config: dict[str, Any] | None = None
        self.vocab: Vocab | None = None
        self.pragma_mini: nn.Module | None = None
        self.classifier: PRAGMAGClassifier | None = None
        self.tokenizer: KVTTokenizer | None = None
        self.model_version: str = "not_loaded"

    def load(self) -> None:
        """Loads config, vocab, and model weights (or falls back to fresh weights)."""
        with open(self.config_path) as f:
            self.raw_config = yaml.safe_load(f)
        assert self.raw_config is not None
        self.config = PRAGMAMiniConfig.from_yaml(self.config_path)
        graph_cfg = self.raw_config["graph"]

        if self.vocab_path.exists():
            self.vocab = Vocab.load(self.vocab_path)
        else:
            self.vocab = Vocab().build(
                make_synthetic_transactions_df(),
                key_vocab_size=self.config.key_vocab_size,
                value_vocab_size=self.config.value_vocab_size,
            )
        self.tokenizer = KVTTokenizer(self.vocab)

        pragma_mini = PRAGMAMini(self.config)
        lora_config = LoraConfig(
            r=graph_cfg["lora"]["r"],
            lora_alpha=graph_cfg["lora"]["alpha"],
            target_modules=build_lora_target_modules(self.config.history_layers),
        )
        # peft's type stubs expect a transformers `PreTrainedModel`, but
        # `get_peft_model` works with any `nn.Module` at runtime.
        self.pragma_mini = get_peft_model(pragma_mini, lora_config)  # type: ignore[arg-type]

        self.classifier = PRAGMAGClassifier(
            d_model=self.config.d_model,
            graph_hidden_channels=graph_cfg["hidden_channels"],
            graph_n_layers=graph_cfg["n_layers"],
            graph_aggregation=graph_cfg["aggregation"],
            dropout=graph_cfg["dropout"],
        )

        mini_ckpt = self.checkpoint_dir / "pragma_mini_lora.pt"
        clf_ckpt = self.checkpoint_dir / "classifier.pt"
        if mini_ckpt.exists() and clf_ckpt.exists():
            self.pragma_mini.load_state_dict(torch.load(mini_ckpt, map_location="cpu"))
            self.classifier.load_state_dict(torch.load(clf_ckpt, map_location="cpu"))
            self.model_version = "pragma-g-aml-v1"
        else:
            self.model_version = "pragma-g-untrained-dev"

        self.pragma_mini.eval()
        self.classifier.eval()

    def warmup(self, n: int = 10) -> None:
        """Runs `n` dummy inferences to warm up CPU kernel caches."""
        dummy = TransactionRequest(
            account_id="WARMUP",
            events=[
                EventRecord(
                    type="wire",
                    amount=100.0,
                    currency="US Dollar",
                    payment_format="Wire",
                    timestamp="2026-01-01T00:00:00",
                )
            ],
        )
        for _ in range(n):
            self.score(dummy)

    def _encode_events(
        self, request: TransactionRequest
    ) -> tuple[list[dict[str, Any]], list[EncodedEvent]]:
        assert self.tokenizer is not None
        events = [_event_dict(e, request.account_id) for e in request.events]
        events = sorted(events, key=lambda e: e["created"])
        encoded = self.tokenizer.encode_history(events)
        return events, encoded

    def _ad_hoc_graph(self, request: TransactionRequest) -> tuple[Tensor, Tensor, list[str]]:
        """Builds a 1-hop ad-hoc graph: node 0 = `request.account_id`, nodes 1..k =
        unique counterparty accounts (cold-start, zero embeddings).
        """
        assert self.vocab is not None
        counterparties: list[str] = []
        edges_src: list[int] = []
        edges_dst: list[int] = []
        edge_attrs: list[list[float]] = []

        events_sorted = sorted(request.events, key=lambda e: e.timestamp)
        if events_sorted:
            t0 = events_sorted[0].timestamp
        for event in events_sorted:
            if event.counterparty_account is None:
                continue
            if event.counterparty_account not in counterparties:
                counterparties.append(event.counterparty_account)
            dst_idx = 1 + counterparties.index(event.counterparty_account)
            seconds_since_start = (event.timestamp - t0).total_seconds()
            edges_src.append(0)
            edges_dst.append(dst_idx)
            edge_attrs.append(_edge_attr_for_event(self.vocab, event, seconds_since_start))

        if not edge_attrs:
            edges_src, edges_dst = [0], [0]
            edge_attrs = [[0.0] * EDGE_FEATURE_DIM]

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        return edge_index, edge_attr, counterparties

    def _classify(
        self,
        z_account: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        n_counterparties: int,
        use_graph: bool,
    ) -> float:
        assert self.classifier is not None
        z_temporal = torch.cat(
            [z_account.unsqueeze(0), torch.zeros(n_counterparties, z_account.shape[-1])], dim=0
        )
        with torch.no_grad():
            logits = self.classifier(z_temporal, edge_index, edge_attr, use_graph=use_graph)
        return torch.sigmoid(logits).mean().item()

    def _decision(self, score: float) -> tuple[str, str]:
        assert self.raw_config is not None
        thresholds = self.raw_config["api"]["thresholds"]
        if score >= thresholds["flag"]:
            decision = "flag"
        elif score >= thresholds["review"]:
            decision = "review"
        else:
            decision = "clear"
        return decision, thresholds["version"]

    def _attributions(self, request: TransactionRequest) -> dict[str, float]:
        """Ablation-based feature attributions: for each feature, the drop in score
        when that feature is removed/neutralised (positive => increases risk).
        """
        assert self.pragma_mini is not None
        assert self.vocab is not None
        assert self.tokenizer is not None

        events, encoded_baseline = self._encode_events(request)
        edge_index, edge_attr, counterparties = self._ad_hoc_graph(request)
        n_cp = len(counterparties)

        variants: dict[str, list[EncodedEvent]] = {
            "baseline": encoded_baseline,
            "amount_paid": self.tokenizer.encode_history(_ablate_amount(events)),
            "payment_format": self.tokenizer.encode_history(
                _ablate_field(events, "payment_format")
            ),
            "receiving_currency": self.tokenizer.encode_history(
                _ablate_field(events, "receiving_currency")
            ),
            "payment_currency": self.tokenizer.encode_history(
                _ablate_field(events, "payment_currency")
            ),
            "temporal_velocity": _zero_time_deltas(encoded_baseline),
        }
        names = list(variants.keys())
        batch = collate_histories([variants[n] for n in names], self.vocab, max_events=64)
        with torch.no_grad():
            z_h = self.pragma_mini(batch)["z_h"]

        z = dict(zip(names, z_h))
        base_score = self._classify(z["baseline"], edge_index, edge_attr, n_cp, use_graph=True)

        scores = {
            "amount_paid": self._classify(
                z["amount_paid"], edge_index, edge_attr, n_cp, use_graph=True
            ),
            "payment_format": self._classify(
                z["payment_format"], edge_index, edge_attr, n_cp, use_graph=True
            ),
            "receiving_currency": self._classify(
                z["receiving_currency"], edge_index, edge_attr, n_cp, use_graph=True
            ),
            "payment_currency": self._classify(
                z["payment_currency"], edge_index, edge_attr, n_cp, use_graph=True
            ),
            "graph_fan_in_ratio": self._classify(
                z["baseline"], edge_index, edge_attr, n_cp, use_graph=False
            ),
            "temporal_velocity": self._classify(
                z["temporal_velocity"], edge_index, edge_attr, n_cp, use_graph=True
            ),
        }
        return {feature: round(base_score - scores[feature], 4) for feature in EXPLAIN_FEATURES}

    def score(self, request: TransactionRequest, counterfactual: bool = False) -> ScoreResult:
        assert self.pragma_mini is not None
        assert self.vocab is not None

        _, encoded = self._encode_events(request)
        batch = collate_histories([encoded], self.vocab, max_events=64)
        with torch.no_grad():
            z_h = self.pragma_mini(batch)["z_h"][0]

        edge_index, edge_attr, counterparties = self._ad_hoc_graph(request)
        score = self._classify(z_h, edge_index, edge_attr, len(counterparties), use_graph=True)
        decision, threshold_version = self._decision(score)

        attributions = self._attributions(request)
        top5 = dict(
            sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        )

        return ScoreResult(
            score=score,
            decision=decision,
            threshold_version=threshold_version,
            shap_values=top5,
            graph_neighbours=len(counterparties),
        )

    def explain(self, request: TransactionRequest) -> ExplainResult:
        _, encoded = self._encode_events(request)
        assert self.pragma_mini is not None
        assert self.vocab is not None
        batch = collate_histories([encoded], self.vocab, max_events=64)
        with torch.no_grad():
            z_h = self.pragma_mini(batch)["z_h"][0]

        edge_index, edge_attr, counterparties = self._ad_hoc_graph(request)
        score = self._classify(z_h, edge_index, edge_attr, len(counterparties), use_graph=True)

        attributions = self._attributions(request)
        neighbourhood = [
            {"account_id": acc, "relation": "counterparty"} for acc in counterparties
        ]

        return ExplainResult(
            score=score,
            all_shap_values=attributions,
            graph_neighbourhood=neighbourhood,
        )
