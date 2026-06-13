"""CLI: build the PRAGMA-G key/value vocabulary from IBM AML transactions.

Usage:
    python scripts/build_vocab.py --data data/HI-Small_Trans.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.tokenizer.vocab import Vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PRAGMA-G vocabulary")
    parser.add_argument("--data", required=True, type=Path, help="Path to HI-Small_Trans.csv")
    parser.add_argument("--out", type=Path, default=Path("configs/vocab.json"))
    parser.add_argument("--key-vocab-size", type=int, default=64)
    parser.add_argument("--value-vocab-size", type=int, default=28672)
    parser.add_argument("--n-numerical-buckets", type=int, default=100)
    parser.add_argument("--categorical-threshold", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    vocab = Vocab().build(
        df,
        key_vocab_size=args.key_vocab_size,
        value_vocab_size=args.value_vocab_size,
        n_numerical_buckets=args.n_numerical_buckets,
        categorical_threshold=args.categorical_threshold,
    )
    vocab.save(args.out)

    print(f"Vocab built: {len(vocab.id2key)} keys, {vocab.value_vocab_size} values -> {args.out}")
    for key, fv in vocab.field_vocabs.items():
        print(f"  {key:24s} kind={fv.kind:11s} size={fv.size}")


if __name__ == "__main__":
    main()
