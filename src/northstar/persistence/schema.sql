CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_title TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    document_version TEXT NOT NULL,
    document_type TEXT NOT NULL,
    department TEXT NOT NULL,
    jurisdiction TEXT NOT NULL,
    classification TEXT NOT NULL,
    section_title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding VECTOR(384) NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_filename, document_version, chunk_index, content_hash)
);
