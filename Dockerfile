FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROD_AGENT_CHECKPOINT_DB=/data/agent_checkpoints.db \
    PROD_AGENT_BUSINESS_DB=/data/demo_business.db

WORKDIR /app

RUN addgroup --system prodagent && adduser --system --ingroup prodagent prodagent

COPY pyproject.toml README.md ./
COPY agent ./agent

RUN pip install --no-cache-dir ".[anthropic,openai]"

RUN mkdir -p /data && chown -R prodagent:prodagent /data /app
USER prodagent

EXPOSE 8000

CMD ["python", "-m", "agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
