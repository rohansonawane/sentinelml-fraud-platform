# Security

## Reporting

Open a GitHub issue or email the maintainer. Do not file a public issue for a live production credential leak.

## What this repo stores

- `.env` is gitignored. Copy `.env.example`.
- Audit DB default is `sqlite:///artifacts/audit.db` (local).
- Compose `full` profile Postgres listens on **5433**, not the host’s existing `:5432`.
- Kubernetes `secret.yaml` is a **placeholder**. Rotate `CHANGE_ME`.

## API

- Max body size enforced (`max_body_bytes`).
- Batch score capped at 256 rows.
- Structured logs include `request_id`.
- `/ready` is 503 if the champion bundle is missing.
- Redis down → in-memory online store (degraded features, not a silent auth bypass).

## Decision engine

BLOCK is a **research action**. Do not connect this API to a payment switch without legal/business review. Default REVIEW when the score is ambiguous.

## Dependencies

Dependabot is enabled. CI does not download huge models. Pin Python 3.11 via uv.
