# Reasonix cache-hit study

Study date: 2026-05-23

## Question

Reasonix reports a very high cache-hit ratio, including the public single-day
case of 435,033,856 cache-hit input tokens and 767,616 cache-miss input tokens:

```text
435,033,856 / (435,033,856 + 767,616) = 99.82%
```

The important point is that this is not an application cache. It is DeepSeek
provider-side prefix cache reuse, measured through `prompt_cache_hit_tokens`
and `prompt_cache_miss_tokens`. DeepSeek documents context caching as enabled by
default, but a later request only hits when it fully reuses a persisted prefix
unit. Reasonix's contribution is to make the client loop preserve that prefix
byte stability over long agent sessions.

Primary sources:

- Reasonix upstream: <https://github.com/esengine/DeepSeek-Reasonix>
- Reasonix architecture: <https://github.com/esengine/DeepSeek-Reasonix/blob/main/docs/ARCHITECTURE.md>
- Reasonix cache case study: <https://github.com/esengine/DeepSeek-Reasonix/blob/main/benchmarks/real-world-cache/README.md>
- DeepSeek context caching guide: <https://api-docs.deepseek.com/guides/kv_cache>
- Local reference snapshot: `描述文件/参考/DeepSeek-Reasonix/`

## Core answer

Reasonix lifts hit rate by treating cache stability as a loop invariant:

1. Freeze the system prompt, tool schemas, and few-shot prefix for a session.
2. Append conversation history in order instead of reordering or rewriting it.
3. Keep per-turn scratch, model reasoning leakage, and transient plan state out
   of the next upstream prompt unless it has been repaired into a clean message.
4. Repair malformed tool calls so the loop does not waste turns on invalid JSON,
   missing tool-call fields, or repeated identical calls.
5. Record cache/cost stats and prefix fingerprints so prefix churn is visible.

This design makes each request look like:

```text
stable prefix + previous append-only log + latest user/tool suffix
```

After the first miss pays to persist the prefix, later turns mostly add a small
new suffix. The already-seen prefix can then be served from DeepSeek's cache.

## Evidence from implementation

| Mechanism | Evidence span | What it protects |
|---|---|---|
| Immutable prefix | `src/memory/runtime.ts` keeps `system`, `toolSpecs`, `fewShots`, and a prefix fingerprint; `addTool`, `removeTool`, and `replaceSystem` invalidate the fingerprint deliberately. | Tool/schema/system bytes stay stable across ordinary turns. |
| Append-only log | `AppendOnlyLog.append()` is the normal path; `compactInPlace()` is reserved for compact/recovery paths. | Previous prompt bytes are not reshuffled on every turn. |
| Stable request assembly | `CacheFirstLoop.buildMessages()` sends `prefix.toMessages()` before the healed active log, then adds only the pending user input. | New user text lands at the suffix instead of perturbing the prefix. |
| Volatile scratch | `VolatileScratch` stores reasoning, plan state, and notes, and the loop resets it each turn. | Transient reasoning does not poison future cache keys. |
| Tool-call repair | `ToolCallRepair` runs scavenge, truncation repair, and storm breaking before dispatch. | Reasoning leaks and malformed calls are recovered without extra failed turns. |
| Context management | `ContextManager` folds or mechanically truncates only when ratios/byte limits require it, and shrinks oversized tool outputs. | Long sessions avoid context failure while minimizing destructive rewrites. |
| Replayable telemetry | Transcript records include usage, cost, model, and `prefixHash`; replay reports cache hit ratio and prefix-hash churn. | Cache regressions become auditable instead of anecdotal. |

## Transferable strategy lessons

| Claim | Why it works | Evidence span | Transferable template | Risk/limit | Confidence |
|---|---|---|---|---|---|
| Put all stable instructions and tool schemas before volatile user/task content. | DeepSeek cache matching starts from the request prefix; stable early bytes maximize the reusable span. | DeepSeek caching guide; Reasonix `ImmutablePrefix`; architecture Pillar 1. | Define `stable_prefix`, `append_log`, and `volatile_suffix` as separate runtime regions. | Any change to system prompt or tool schema forces at least one cache-miss turn. | High |
| Make conversation mutation append-first. | A later request can reuse a previous request prefix when earlier messages are byte-identical and in the same order. | `AppendOnlyLog`; `CacheFirstLoop.buildMessages()`; replay `prefixHash`. | Never sort, reformat, timestamp, or reserialize old messages during normal turns. | Recovery, rewind, and compaction still need controlled rewrite paths. | High |
| Keep scratch reasoning outside the committed prompt. | Per-turn thoughts are high-churn content; committing them directly reduces prefix reuse and may leak malformed tool calls. | `VolatileScratch`; repair scavenge pass over `reasoning_content`. | Store scratch separately, then promote only repaired, user-visible, necessary facts. | If too little is promoted, later turns may lack useful rationale. | Medium-high |
| Repair tool-call shape before dispatch. | Invalid tool JSON and repeated identical calls consume turns and create noisy log suffixes that lower effective task throughput. | `ToolCallRepair`, `scavengeToolCalls`, truncation repair, storm breaker. | Normalize native calls, scavenge reasoning/content channels, repair bounded JSON, suppress repeated signatures. | Over-aggressive repair can execute an unintended call; keep allowlists and logs. | Medium-high |
| Make cache economics visible in the UI and transcript. | Teams optimize what they can see; prefix churn must be visible per turn, not inferred from final cost. | `Usage.fromApi`, `SessionStats`, `transcript/log.ts`, `replay.ts`. | Record hit tokens, miss tokens, ratio, model, cost, and prefix hash for every model response. | Provider fields are best-effort and pricing changes over time. | High |
| Compact as a last-resort, structured operation. | Long sessions need bounded context, but uncontrolled summaries rewrite the reusable prefix and destroy cache locality. | `ContextManager.decideAfterUsage()`, `fold()`, `mechanicalTruncate()`. | Trigger compaction by thresholds; preserve recent tail and pinned skill/memory bodies; persist the rewrite once. | Any fold changes future prefixes; a bad summary can erase important state. | Medium |

## Implications for HyperAgent

Already present in HyperAgent:

- `--loop-mode cache-first` records a stable prefix hash.
- Action runs store cumulative token usage, cache-hit/miss tokens, and
  cache-hit ratio in `action_run.json`.
- Runtime events support `events`, `replay`, and `stats`.
- Repair handles native `tool_calls`, JSON list/action wrappers,
  reasoning-content extraction, direct-tool normalization, and storm breaking.

Next useful work:

1. Strengthen prefix partitioning in the normal chat path, not only action loop.
2. Add a regression test that compares two adjacent cache-first requests and
   fails if the stable-prefix hash changes when only the user suffix changes.
3. Surface prefix-hash churn in `HyperAgent stats` and the TUI status panel.
4. Add a small synthetic benchmark: stable prefix vs. intentionally cache-hostile
   prompt reshuffle, using fake usage fields for deterministic CI.
5. Document which fields are allowed to mutate in stable prefix, semi-stable
   context, and volatile suffix.

## Environment, git, and privacy notes

- The project conda environment already exists as `HyperAgent` at
  `/home/lzj/miniconda3/envs/HyperAgent`.
- This study adds no dependencies, so `environment.txt` does not need to change.
- Private data, `.env`, runtime state, datasets, reports, and restricted PDFs
  must stay out of git.
- Repository About metadata is tracked locally in `configs/repository.yaml`;
  updating GitHub's live About panel still requires an authenticated GitHub
  tool or token and should not expose secrets.
