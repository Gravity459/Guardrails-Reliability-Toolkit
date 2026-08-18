# Guardrails Reliability Toolkit

An open-source FastAPI toolkit that enforces reliable structured LLM outputs using Guardrails AI, with auto-correction, fallback handling, and measurable failure reduction benchmarking.

## Gemini rate-limit retries

The shared Gemini client in `app/llm_client.py` honors `google.rpc.RetryInfo.retryDelay` on **HTTP 429** `RESOURCE_EXHAUSTED` responses. SDK auto-retry is disabled (`attempts=1`) so those short exponential waits cannot burn the budget before the server delay is read.

On a 429, the client waits at least the server's `retryDelay` (for example `"53s"`), then retries. Later failures grow from that wait and are capped by `LLM_RETRY_MAX_DELAY`. If `retryDelay` is missing, it tries the HTTP `Retry-After` header, then a `"Please retry in …s"` message, then `LLM_RETRY_INITIAL_DELAY`.

Daily quota exhaustion (`QuotaFailure` with a `PerDay` `quotaId`) is not retried. Exhausted 429s surface as `llm_call_failed` in the guarded runner, not as `guard_failed`.

### Defaults

| Setting | Default | Meaning |
| --- | --- | --- |
| Attempts | `5` | 1 original request + up to 4 retries |
| Initial delay | `1.0` s | Fallback wait when `retryDelay` is missing |
| Max delay | `60.0` s | Cap between attempts |

At a free-tier limit such as 15 RPM, a 200-prompt benchmark still needs on the order of 13+ minutes of wall time even with retries.

### Optional environment overrides

| Variable | Purpose |
| --- | --- |
| `LLM_RETRY_ATTEMPTS` | Total attempts including the first request |
| `LLM_RETRY_INITIAL_DELAY` | Fallback wait in seconds when `retryDelay` is missing |
| `LLM_RETRY_MAX_DELAY` | Maximum wait between attempts in seconds |

Invalid or non-positive values fall back to the defaults above.

### Related env vars

| Variable | Purpose |
| --- | --- |
| `LLM_API_KEY` | Gemini API key |
| `LLM_PRIMARY_MODEL` | Model id (for example `gemini-3.5-flash-lite`) |

### Tests

```bash
./venv/bin/python -m pytest tests/ -v
```
