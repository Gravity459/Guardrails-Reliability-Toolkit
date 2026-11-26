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

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml`, exact versions in `uv.lock`.

```bash
uv sync                              # create .venv and install everything (incl. pytest)
uv run uvicorn app.main:app --reload # run the API
```

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for dev tools).

## Benchmark

200 prompts in `benchmark/prompts.json`, each with a hand-written `answer`.
Both runs use the same system prompt (`app/prompts.py`) and temperature 0.

```bash
uv run python -m benchmark.baseline_runner   # plain Gemini -> results_baseline.json
uv run python -m benchmark.guarded_runner    # Gemini + Guardrails -> results_guarded.json
uv run python -m benchmark.report            # score both, write benchmark/report.md
```

- Runners only save raw outputs; `benchmark/scorer.py` grades them.
- Scoring: model and camera must match the answer **exactly** (case-sensitive);
  storage/battery spacing and case are normalised ("256 GB" = "256GB"); year is an integer.
- Responses are cached in `benchmark/cache/`, so reruns don't call Gemini.
  Use `--refresh` to call again. Changing the system prompt invalidates the cache;
  for the guarded run, so does editing `app/guard.py` or `app/schemas/guarded_schema.py`.
- Latency excludes rate-limit (429) waits; those are saved separately as `wait_ms`.
- Failed calls (network, quota) are never cached and get one retry after the main pass.
  If the daily quota runs out, the run stops and saves; rerun later and only the missing
  prompts are called. Failed calls are left out of the scores (the report shows "scored: X/200");
  `guard_failed` is a real guard result and stays in.
- Guardrails validators (`guardrails-ai-valid-length`, `guardrails-ai-valid-range`) are
  normal PyPI packages installed by `uv sync`. No `guardrails configure` or token needed.
- Guardrails usage telemetry is turned off in code (`app/guard.py`), so no traces leave your machine.
- Guarded validators use `on_fail="reask"`: a failing value is sent back to Gemini with the
  error, once (max 2 calls per prompt). If it still fails, the result is `guard_failed`.

### Tests

```bash
uv run pytest -v
```

Tests need no API key and no network: Gemini calls are mocked and `tests/conftest.py` sets a dummy key.
