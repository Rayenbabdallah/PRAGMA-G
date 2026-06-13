#!/usr/bin/env bash
# Download IBM AML dataset from Kaggle
# Requires: KAGGLE_USERNAME and KAGGLE_KEY set in .env or environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data"

# Load .env if it exists
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

# Validate credentials
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    echo "ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env or environment"
    echo "Get your credentials at: https://www.kaggle.com/settings/account"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Downloading IBM AML dataset (HI-Small + HI-Medium)..."
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    --path "$DATA_DIR" \
    --unzip

echo ""
echo "Dataset downloaded to $DATA_DIR"
echo "Files:"
ls -lh "$DATA_DIR"/*.csv 2>/dev/null || echo "No CSV files found — check download"

echo ""
echo "Generating checksums..."
sha256sum "$DATA_DIR"/HI-Small_Trans.csv > "$DATA_DIR/checksums.sha256" 2>/dev/null || true
echo "Done."
