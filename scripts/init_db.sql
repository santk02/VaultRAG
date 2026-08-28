-- VaultRAG Database Schema
-- This script initializes the PostgreSQL database with required tables

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    doc_id        VARCHAR(36) UNIQUE NOT NULL,
    filename      VARCHAR(255) NOT NULL,
    page_count    INTEGER,
    chunk_count   INTEGER,
    content_hash  VARCHAR(64),
    uploaded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    chunk_id     VARCHAR(36) UNIQUE NOT NULL,
    doc_id       VARCHAR(36) NOT NULL REFERENCES documents(doc_id),
    chunk_index  INTEGER,
    page_number  INTEGER,
    text         TEXT NOT NULL,
    token_count  INTEGER
);

CREATE TABLE IF NOT EXISTS queries (
    id             SERIAL PRIMARY KEY,
    query_id       VARCHAR(36) UNIQUE NOT NULL,
    user_id        VARCHAR(255),
    question       TEXT,
    mode           VARCHAR(20),
    model_used     VARCHAR(80),
    chunk_ids      TEXT,
    answer         TEXT,
    latency_ms     FLOAT,
    token_cost     FLOAT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id);
CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Document metadata and tracking';
COMMENT ON TABLE chunks IS 'Text chunks with page numbers for citation accuracy';
COMMENT ON TABLE queries IS 'Audit trail of all queries for compliance';
COMMENT ON COLUMN chunks.page_number IS 'Critical for citations - must be preserved through pipeline';
