# Deployment Notes

`prod-agent` defaults to local development settings. Treat tool calls as privileged
backend actions: an EXECUTE tool may send messages, submit refunds, deploy code,
or touch another external system.

## Minimum Safe Setup

- Keep the server bound to `127.0.0.1` unless it is behind a reverse proxy.
- Terminate TLS at nginx, Caddy, a cloud load balancer, or an equivalent proxy.
- Set `PROD_AGENT_API_TOKEN` before exposing any `/api/*` endpoint.
- Set a separate `PROD_AGENT_APPROVAL_TOKEN` when approvers should be distinct
  from normal task submitters.
- Store SQLite databases on persistent storage.
- Back up `agent_checkpoints.db` if tasks are business critical.

## Local Demo

```bash
cp .env.example .env
docker compose up --build
```

The demo provider does not need API keys. Open `http://127.0.0.1:8000`.

## Production Shape

Recommended first production shape:

```text
client/browser
  -> HTTPS reverse proxy
  -> prod-agent server on 127.0.0.1:8000
  -> SQLite/Postgres checkpoints
  -> worker process
```

For team or public access, prefer:

- nginx/Caddy with TLS
- static Bearer tokens at minimum
- network allowlists where possible
- separate approval token
- audit logs retained outside the application host

## Example Environment

```bash
PROD_AGENT_PROVIDER=anthropic
PROD_AGENT_MODEL=claude-opus-4-8
ANTHROPIC_API_KEY=sk-...
PROD_AGENT_API_TOKEN=submit-secret
PROD_AGENT_APPROVAL_TOKEN=approve-secret
PROD_AGENT_CHECKPOINT_DB=/var/lib/prod-agent/agent_checkpoints.db
PROD_AGENT_BUSINESS_DB=/var/lib/prod-agent/demo_business.db
```

## Reverse Proxy

See `deploy/nginx.example.conf`. The app itself intentionally does not manage
TLS certificates, IP allowlists, or identity federation.
