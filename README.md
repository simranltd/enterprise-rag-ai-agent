# Northstar Financial Services Knowledge AI Agent

## Project purpose

This repository is a portfolio, learning, team demonstration, and technical
interview project for an enterprise Knowledge AI Agent. The future application
will help authorized Northstar employees ask natural-language questions about
internal policies and procedures, then receive grounded answers with source
citations.

**Northstar Financial Services is entirely fictional.** All documents in
`data/sample_documents/` are synthetic content created for public GitHub
demonstration and testing. They contain no real company data or personal
information.

## Business scenario

Northstar maintains internal guidance for employees and operations teams,
including:

- Employee Handbook
- Expense Policy
- Cybersecurity Policy
- Client Onboarding SOP
- AML Procedure
- IT Access Policy
- Remote Work Policy
- Incident Response Procedure

Representative future questions include:

- Can employees work outside Canada?
- What approvals are required for expenses above $5,000?
- What happens if a user loses access to MFA?
- What identification is required for a new client?
- What AML checks are required for high-risk clients?
- When must access be revoked for terminated employees?

## Planned architecture

The approved target architecture separates document processing from question
answering:

1. Documents are validated, extracted, normalized, chunked, enriched with
   metadata, embedded, and indexed.
2. An authenticated API applies authorization and metadata filters before
   retrieving evidence.
3. Hybrid retrieval combines semantic/vector search with keyword search and
   optional reranking.
4. A managed large language model generates an answer only from retrieved
   evidence.
5. Citation validation, confidence checks, observability, and evaluation
   protect answer quality.

The initial implementation is intentionally being built in phases. This
repository currently contains only the Phase 0 foundation and fictional domain
assets; no AI, API, database, or infrastructure implementation is included yet.

## Future RAG flow

```text
User question
  -> authentication and authorization
  -> conversation context and query clarification
  -> metadata-aware hybrid retrieval
  -> reranking and evidence sufficiency check
  -> grounded LLM response
  -> citation validation and audit telemetry
```

If the indexed documents do not provide sufficient evidence, the future agent
will explicitly say that it cannot answer from Northstar's available
documents rather than inventing a policy.

## Future vector database role

Document chunks will eventually be stored with embeddings, source locations,
document versions, and access metadata. A vector database will support
semantic similarity search, while keyword search will preserve exact matching
for amounts, acronyms, dates, and policy names. PostgreSQL with `pgvector` is
the planned learning-oriented starting point; Azure AI Search is an important
managed production alternative.

## Future Azure deployment target

The production-ready target is Microsoft Azure using services such as:

- Azure Container Apps for the API and ingestion worker
- Azure Blob Storage for source documents
- Azure Database for PostgreSQL and/or Azure AI Search for indexed content
- Azure OpenAI for embeddings and generation
- Azure Service Bus for asynchronous ingestion
- Microsoft Entra ID for identity
- Azure Key Vault for secrets
- Azure Monitor and Application Insights for observability

Infrastructure as code, managed identities, private networking, separate
environments, and deployment safeguards will be addressed in later phases.

## Repository layout

```text
apps/                 Future API and background-worker entry points
src/northstar/        Future domain and application components
data/sample_documents Synthetic Northstar policies and procedures
data/evaluation_sets  Future RAG evaluation datasets
infrastructure/       Future Docker and Azure infrastructure definitions
tests/                Future unit, integration, retrieval, and API tests
docs/                 Architecture, security, and operations documentation
frontend/             Future React/TypeScript user interface
```

## Phased roadmap

1. **Phase 0 - Foundation:** repository structure, public-safe configuration
   templates, documentation, and synthetic Northstar documents.
2. **Phase 1 - Ingestion foundation:** extraction, normalization, chunking,
   metadata, and ingestion status tracking.
3. **Phase 2 - Retrieval:** embeddings, PostgreSQL/`pgvector`, keyword search,
   hybrid scoring, and metadata filters.
4. **Phase 3 - Grounded RAG API:** LLM adapter, prompts, citations,
   abstention, and conversation persistence.
5. **Phase 4 - User experience:** React chat experience, source display, and
   feedback.
6. **Phase 5 - Security and governance:** identity, authorization, auditing,
   retention, and upload controls.
7. **Phase 6 - Evaluation and reliability:** golden datasets, regression
   metrics, tracing, latency, cost, and quality gates.
8. **Phase 7 - Controlled agent tools:** safe knowledge and escalation tools.
9. **Phase 8 - Azure deployment:** infrastructure as code, CI/CD, and managed
   services.
10. **Phase 9 - Production hardening:** performance, disaster recovery,
    security testing, and operational runbooks.

## Public-demo safety

The sample documents use fictional names, values, departments, and procedures.
They are not legal, regulatory, employment, cybersecurity, or financial
advice. They exist only to exercise future ingestion, retrieval, versioning,
metadata filtering, citation, conflict, and insufficient-evidence behavior.

## Current status

Phase 0 only. Do not treat the repository as a working AI application yet.
Implementation of later phases requires explicit approval.
