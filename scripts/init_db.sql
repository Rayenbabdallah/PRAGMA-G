-- PRAGMA-G Postgres init script (PLAN.md Week 7)
-- Runs once on first `postgres` container startup (docker-entrypoint-initdb.d).

-- Separate database for the MLflow tracking server's backend store.
CREATE DATABASE mlflow;

-- Audit log written by scripts/stream_consumer.py: one row per scored
-- transaction from the live aml_transactions stream.
CREATE TABLE IF NOT EXISTS scored_transactions (
    id BIGSERIAL PRIMARY KEY,
    account_id TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    decision TEXT NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scored_transactions_scored_at
    ON scored_transactions (scored_at);
