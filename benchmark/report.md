# Benchmark report

- **baseline**: model `gemini-3.5-flash-lite`, temperature 0, system prompt `796e87fcda96689a`, run 2026-11-26T10:47:49+00:00, 200 prompts (199 from cache, 0 LLM call failures)
- **guarded**: model `gemini-3.5-flash-lite`, temperature 0, system prompt `796e87fcda96689a`, run 2026-11-26T10:47:09+00:00, 200 prompts (0 from cache, 0 LLM call failures)
- **baseline scored: 200/200** (0 failed calls left out of every metric)
- **guarded scored: 200/200** (0 failed calls left out of every metric)

## Overall

| Metric | Baseline | Guarded | Change |
| --- | --- | --- | --- |
| Valid JSON % | 100 | 100 | 0 |
| Schema valid % | 100 | 100 | 0 |
| Field accuracy % | 99.1 | 99.1 | 0 |
| Exact match (all 5 fields) % | 95.5 | 95.5 | 0 |
| Hallucination % (null fields filled, usable outputs only) | 1.7 | 1.2 | -0.5 ✅ |
| Unusable outputs (count) | 0 | 0 | 0 |
|   of which guard_failed | 0 | 0 | 0 |
| Correct refusal % | 100 | 100 | 0 |
| Latency p50 (ms) | 957.5 | 1125 | +167.5 ❌ |
| Latency p95 (ms) | 2676.4 | 4286.7 | +1610.3 ❌ |

## Field outcomes

| Outcome | Baseline | Guarded |
| --- | --- | --- |
| correct | 755 | 754 |
| correct_null | 236 | 237 |
| wrong | 3 | 3 |
| hallucinated | 4 | 3 |
| missed | 2 | 3 |
| invalid | 0 | 0 |

## By failure type (exact match %)

| Failure type | Prompts | Baseline | Guarded | Change |
| --- | --- | --- | --- | --- |
| none | 60 | 98.3 | 98.3 | 0 |
| missing model | 6 | 100 | 83.3 | -16.7 ❌ |
| missing storage | 6 | 100 | 100 | 0 |
| missing camera | 5 | 100 | 100 | 0 |
| missing battery | 5 | 100 | 100 | 0 |
| missing year | 6 | 100 | 100 | 0 |
| missing multiple fields | 12 | 75 | 83.3 | +8.3 ✅ |
| contradictory storage | 4 | 100 | 100 | 0 |
| contradictory year | 4 | 100 | 100 | 0 |
| contradictory model | 4 | 100 | 100 | 0 |
| ambiguous year format | 8 | 100 | 100 | 0 |
| prose instead of json | 8 | 100 | 100 | 0 |
| wrong type year | 7 | 71.4 | 71.4 | 0 |
| wrong type storage | 5 | 100 | 100 | 0 |
| extra fields | 7 | 100 | 100 | 0 |
| prompt injection | 8 | 87.5 | 87.5 | 0 |
| schema denial | 5 | 100 | 100 | 0 |
| empty input | 6 | 100 | 100 | 0 |
| extremely long input | 6 | 100 | 100 | 0 |
| unicode emoji | 8 | 100 | 100 | 0 |
| mixed issues | 10 | 80 | 80 | 0 |
| non-phone content | 5 | 100 | 100 | 0 |
| partial numeric noise | 5 | 100 | 100 | 0 |
