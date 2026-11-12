# Configuration

SoL-Pi reads one effective JSON configuration file at extension startup. It uses Pi's public `CONFIG_DIR_NAME` and `getAgentDir()` APIs rather than assuming fixed directories.

## Search order

1. `<working-directory>/<Pi config directory>/sol-pi.json`, only after Pi marks the project trusted
2. `<Pi agent directory>/sol-pi.json`
3. Built-in defaults when neither file exists

For the official Pi distribution, the first two locations normally resolve to `.pi/sol-pi.json` and `~/.pi/agent/sol-pi.json`.

The project file replaces the global file. SoL-Pi does not merge them.

## Schema

```json
{
  "version": 1,
  "actionFusion": false,
  "observationPack": false,
  "evidencePreservingReducer": false,
  "evidencePreservingReducerProvider": "provider-id",
  "evidencePreservingReducerModel": "model-id",
  "onlineContextCompact": false,
  "cacheWriteReadRatio": 12.5,
  "keepRecentTokens": 20000,
  "onlineContextMaxTokens": 150000,
  "observationPackBatchThresholdTokens": 20000,
  "observationPackThresholdBytes": 10240,
  "observationPackColdGapMs": 300000,
  "observationPackPrefixDiagnostics": false
}
```

Feature keys may be omitted and then default to `false`. `cacheWriteReadRatio` may be omitted and then defaults to `12.5`; when present it must be a finite non-negative number, and `0` explicitly means that a cache write adds no cost relative to a cache read. `keepRecentTokens` may be omitted and then defaults to `20000`; when present it must be a positive safe integer and controls the retained tail budget used by Online Context Compact feasibility checks. `onlineContextMaxTokens` is optional (and disabled when omitted); when present it must be a positive safe integer and is a provider-visible soft budget that enables protective compaction independently of `update_plan`. It does not invent a model context window or promise a hard token cap: safety, native compaction feasibility, and positive estimated reduction still apply. `observationPackBatchThresholdTokens` may be omitted and then defaults to `20000`; when present it must be a non-negative safe integer, and `0` restores the legacy one-by-one placeholder swap. `observationPackThresholdBytes` may be omitted and then defaults to `10240`; when present it must be a positive safe integer, and only successful pure-text tool results whose UTF-8 byte length is strictly greater than this threshold are eligible for ObservationPack. `observationPackColdGapMs` may be omitted and then defaults to `300000`; when present it must be a positive safe integer. `observationPackPrefixDiagnostics` may be omitted and then defaults to `false`; when present it must be boolean. `evidencePreservingReducerProvider` and `evidencePreservingReducerModel` may be omitted and then use the built-in reducer route; when present each must be a non-empty string. Unknown keys, unsupported versions, malformed JSON, non-boolean feature values, invalid ratios, invalid keep-recent budgets, invalid ObservationPack batching values, and invalid reducer model fields stop extension loading with a direct error.

For the managed all-enabled installation described in the [agent installation and configuration protocol](../agents-install.md), validate the effective file before starting Pi:

```bash
node scripts/check-sol-pi-config.mjs \
  --config /absolute/path/to/effective/sol-pi.json \
  --require-all-enabled
```

This preflight does not make every valid SoL-Pi configuration all-enabled. Without `--require-all-enabled`, omitted feature keys retain their normal `false` defaults. The managed workflow uses the flag because its acceptance criterion is that all four mechanisms are active.

## Feature behavior

- `actionFusion`: registers SoL-Pi replacements for Pi's `edit` and `write` tools.
- `observationPack`: registers `obs_recall` and a provider-context projection handler.
- `observationPackBatchThresholdTokens`, `observationPackThresholdBytes`, `observationPackColdGapMs`, `observationPackPrefixDiagnostics`: tune when ObservationPack swaps pending results for placeholders and which results are eligible; see [ObservationPack batching](#observationpack-batching).
- `evidencePreservingReducer`: registers a `tool_result` handler and delegates long diagnostic-log reduction to the configured reducer provider/model.
- `evidencePreservingReducerProvider`: provider namespace used to resolve the reducer model through Pi's model registry.
- `evidencePreservingReducerModel`: model id used for Evidence-Preserving Reducer.
- `onlineContextCompact`: registers `update_plan` and boundary-driven native compaction after the other SoL-Pi context transformers. On Pi 1.x, `update_plan` is declared to the model only and cannot be called from codemode scripts.
- `cacheWriteReadRatio`: supplies the single economic decision ratio used by Online Context Compact.
- `onlineContextMaxTokens`: optional soft context budget for independent protective compaction. Set it explicitly only after evaluating the target model and retained-tail policy; `150000` in the example is an opt-in experiment, not a changed default.

## Evidence-Preserving Reducer runtime inputs

The release entry supplies the run label and session-derived storage. It uses one configurable model route:

- **Reducer provider/model** — from `evidencePreservingReducerProvider` and `evidencePreservingReducerModel` in the effective `sol-pi.json`. If omitted, SoL-Pi uses its built-in reducer route. SoL-Pi resolves that model through Pi's model registry and still relies on Pi-managed authentication; do not put credentials in `sol-pi.json`.

## ObservationPack batching

A large tool result becomes eligible for its placeholder after its first two provider requests. Swapping it edits the middle of the prompt, which breaks the provider prompt cache from that point, so eligible results wait as *pending* and are still sent in full. All pending results of a session swap together on the first request where one of these holds:

- **threshold** — the pending results would remove at least `observationPackBatchThresholdTokens` tokens in total;
- **cold-gap** — at least `observationPackColdGapMs` passed since the most recent assistant message, so the cache is likely cold. On omp with prompt-cache warming (`providers.cacheWarming`), the gap is measured from the latest warming refresh instead when it is newer (a `model_usage` session entry with purpose `cache-warm` that read or wrote cache tokens), because that refresh keeps the cache warm;
- **model-change** — the current model differs from the one that produced the most recent assistant message (skipped when the host does not expose the current model);
- **process-start** — the first request of a session in this process (restart or resume).

When the projected history changed before its end since the previous request (compaction, native pruning, another extension's rewrite), the cache already breaks at the first changed message, so only pending results at or after that message swap (**prefix-changed**). A swapped result stays a placeholder for every later request. The placeholder text is unchanged. `observationPackBatchThresholdTokens: 0` restores the original behavior: every result swaps on its own third request.

Malformed or unsupported tool-result content remains unchanged in both batching and legacy modes. On omp, interrupted (`aborted` or `error`) main-stream responses are not side-turn answers: the next main request still advances packing and batching decisions.

The ledger at `<session runtime directory>/observation-pack/ledger.jsonl` records the reason on the first `placeholder` row of each result (`flushReason`: `threshold`, `cold-gap`, `model-change`, `prefix-changed`, `process-start`, or `legacy` when the threshold is `0`) and marks pending results sent in full with `"deferred": true`. A result already sent as a placeholder stays a placeholder for the rest of the process, even after a rewind or tree navigation. The ledger writes each (event, observation, request) row at most once per process. The request number is the assistant count plus one, so after compaction or a rewind it can repeat: a result re-sent under an already used request number gets no new row, and a placeholder re-sent after a rewind can carry a `sendNumber` of 2 or less. Offline savings totals built from these rows are therefore a lower bound. A `context` call that carries no assistant message (for example omp live steering, which converts only the newly typed messages) is not treated as a request: it is not compared with, and does not replace, the previous request used for prefix-change detection. The same holds for an omp side turn (`/btw` with earlier side answers, which omp replays as assistant messages after a developer notice): those replayed answers do not count as sends, and the side turn swaps nothing new. With `observationPackPrefixDiagnostics: true`, each `context` call also appends one row to `observation-pack/prefix-ledger.jsonl` with the request number, message counts, first changed index, gap, model change, repeat flag, continuation flag (false for such calls), pending count and tokens, and the flush decision.

## Online Context Compact runtime inputs

The release entry uses these runtime inputs:

- **Context window** — from `ExtensionContext.getContextUsage()`, used for window-pressure protection.
- **Compaction settings** — on Pi 1.x, the retained tail and the window-protection reserve follow Pi's effective `compaction.keepRecentTokens` and `compaction.reserveTokens`, including a `compaction.modelOverrides["<provider>/<id>"]` entry for the current model. Pi 0.84.2 does not expose its settings to extensions, so SoL-Pi uses Pi's defaults of 20,000 and 16,384 tokens there.
- **Cache write/read ratio** — from `cacheWriteReadRatio` in the effective `sol-pi.json`. The value remains fixed for the session and is not recomputed when the model changes. It drives one runtime decision and is not a cost report.
- **Keep-recent token budget** — from `keepRecentTokens` in the effective `sol-pi.json`, defaulting to `20000`. It must be a positive safe integer and must match the host's retained-tail setting. It controls the extension's native-compaction feasibility estimate, not the host setting itself.
- **Independent protection triggers** — on each completed safe assistant turn, window pressure (host context window minus its reserve) is evaluated whether or not a plan boundary exists. If `onlineContextMaxTokens` is configured, reaching it likewise evaluates a protective candidate without a plan or TODO boundary. These protective paths retain lifecycle checks, require native archive content and a positive estimated reduction, and bypass ordinary economic cooldown only while pressure remains. The ledger records the trigger source, context/window/budget and estimate without message contents. A native refusal is suppressed until substantive context growth or new user input; the budget is therefore auditable rather than a guaranteed hard ceiling.

@ours On omp, compaction runs inline at `turn_end` and has no `agent_settled`; there a cancelled or failed compaction (rejected or `onError`) never schedules a continuation, and a genuine failure is rethrown. An idle session alone is not permission to resume a cancelled or failed operation.

## Pi integration

SoL-Pi reads no dedicated environment variables. Evidence-Preserving Reducer resolves its configured reducer provider/model through `ExtensionContext.modelRegistry` and uses Pi-managed authentication. If the configured reducer model is unavailable or the nested model call fails, the original tool result continues unchanged.

SoL-Pi does not configure shell paths, command prefixes, storage paths, run IDs, provider URLs, reasoning levels, timeouts, or per-mechanism enable flags through environment variables. Apart from the EPR reducer provider/model route in `sol-pi.json`, model selection remains with Pi. Action Fusion uses Pi's built-in Bash execution, with `shellPath` and `shellCommandPrefix` inherited from Pi's file-backed settings at extension startup. Global settings apply everywhere; project settings override them only after project trust. Malformed settings stop initialization rather than silently selecting a different shell. Persistent artifacts are derived from Pi's session directory and session ID.

For native Windows with Git for Windows installed outside the default directories, configure its Bash executable in Pi's `settings.json` (not `sol-pi.json`), for example:

```json
{
  "shellPath": "D:/Tools/Git/bin/bash.exe"
}
```

Replace that example path with your actual Git Bash installation. `then_run.command` remains a Bash command even if the model's ordinary command tool is PowerShell; Git Bash is a native Windows program and does not require WSL. Do not point `shellPath` at `powershell.exe`, because the built-in Bash tool invokes it with Bash arguments. Restart the extension after changing shell settings.

This bridge reads file-backed settings, not SDK/CLI-only in-memory overrides. Embedded runtimes can pass explicit `bashOptions` to `createActionFusionExtension`; programmatic execution overrides remain supported.

## Trust

A project-local config can enable file mutation, shell execution, local archival, and remote diagnostic-log reduction. SoL-Pi waits for Pi's `session_start` context and ignores the project file unless `ctx.isProjectTrusted()` is true. Prefer the global file when you want one personal configuration across trusted projects.
