# Pi Compatibility

SoL-Pi is developed and tested against `@earendil-works/pi-coding-agent` 0.84.2. Its public API surface is also type-checked and exercised against Pi 0.81.1, the base used by the original Pi fork. The runtime range is deliberately expressed as a peer dependency because Pi owns installation and upgrade of its packages.

SoL-Pi imports only public package exports:

- `createEditToolDefinition`
- `createWriteToolDefinition`
- `createBashToolDefinition`
- extension types and `ExtensionAPI.registerTool`
- `context`, `before_provider_request`, `tool_result`, `turn_end`, `agent_settled`, and `session_before_tree` extension events
- native compaction events, `ExtensionContext.getContextUsage()`, and `ExtensionContext.compact()`
- `ExtensionContext.model` and `ExtensionContext.modelRegistry`
- the public session-manager methods exposed through `ExtensionContext`

Host extras that Pi's published typings do not declare — `ctx.invokeTool`, a tool's `approval` declaration, and its `formatApprovalDetails` and `approvalCacheTargets` hooks — are read only after a feature check, through a narrow cast, and are absent on Pi. `ctx.executeTool` is declared by Pi's typings but not exposed by OMP 18.13, and is feature-checked the same way. A host that has these keeps its own behaviour; a host that lacks them runs the public definitions unchanged.

## Action Fusion

The built-in edit/write definitions capture their working directory, so SoL-Pi caches one definition per `ctx.cwd`. Its own per-file queue surrounds the built-in mutation and follow-up command. It does not nest Pi's built-in mutation queue.

On a host that offers same-name tool delegation (`ctx.invokeTool`, as OMP does), the mutation runs through that host tool instead of a load-time copy. Plan-mode restrictions, `@register` contents and the active edit schema then apply exactly as they do without SoL-Pi. On a host that also declares an `approval` (OMP), the fused `edit` tool is registered at `session_start` from the live tool schema, so a session setting such as `edit.mode: replace` keeps its parameters; a host without `approval` keeps the load-time schema. The tool description is not re-read: under a model that OMP forces to a different edit mode (DeepSeek and Kimi-class models use `replace`), the description can still describe the configured mode. A mode change later in the session takes effect in a new session. Pi publishes no `ctx.invokeTool`, so it keeps the load-time definition.

The same host classifies tool calls through an `approval` declaration. A call carrying `then_run` is therefore classified as command execution, scoped to the host's `bash` approval policy: `tools.approval.bash: deny` rejects the call, and an unset `bash` policy falls back to the mutation tool's own policy. A `deny` from the built-in tool is never relaxed, and a must-prompt signal survives. Two consequences: the host consults one policy per call, so a set `bash` policy replaces a set `edit`/`write` policy for calls that carry `then_run`; and approval is per call, so a `bash: deny` rejects the whole call, mutation included. The host's own approval cache targets pass through untouched, because the host uses them for `tools.restrictedFolders`, and the approval prompt names the command next to the files.

Where the host exposes a nested tool call (`ctx.executeTool`, as Pi's development typings do), the command runs as that `bash` tool, so the host's blocking hooks apply. OMP 18.13 exposes none, so there the command runs through the public bash definition, built on isolated settings: the session's `bash` settings (shell, compound-command handling, direnv, auto-background) and policy keyed on the command text, such as OMP's bash pattern deny list or its interception rules, do not apply to it. Where a command must be blocked, use `tools.approval.bash` rather than a pattern rule.

The fused `edit` and `write` parameters are emitted as plain JSON Schema: the host's own parameter shape plus `then_run`, with optionality carried by `required` alone. A host that publishes its schema through `toJsonSchema()` (OMP) returns plain objects, which must not be wrapped in TypeBox helpers.

Action Fusion decodes `file://` targets with Node's `fileURLToPath()` before resolving the queue and hash-check path. It strips a leading `@` only where Pi does — before `/`, `~`, a Windows absolute path, or a URL — so `@config.json` stays a file named `@config.json`. A target named through a host URL scheme (`local://` and similar) or another user's home (`~alice/x`) is left to the host instead of being read from disk.

The queue covers only fused operations registered by this SoL-Pi instance. External processes, direct built-in-tool calls outside the replacement, and unrelated extensions are not globally locked. Every target of a mutation is locked, in sorted canonical order, so two calls sharing a target cannot interleave or deadlock. SoL-Pi hashes every target immediately before launching `then_run` and skips the command if it observes an intervening content change; a target removed by the mutation must stay absent, and a skipped command still reports the mutation that landed.

## ObservationPack

ObservationPack changes only the messages projected through the public `context` event. Stored session history remains intact. Original bytes and the JSONL ledger live under the session-derived SoL-Pi directory.

## Evidence-Preserving Reducer

The reducer handles public `tool_result` events and resolves the configured reducer provider/model through Pi's model registry before calling `ExtensionContext.modelRegistry.complete()` when available. For the Pi 0.81.1 fork, which exposes no registry `complete()` method, it resolves authentication for that reducer model through `getApiKeyAndHeaders()` and calls the shared `@earendil-works/pi-ai/compat` completion API. The reducer preserves the original result whenever the configured reducer model is unavailable or eligibility, model-call, schema, source-hash, exact-quote, size, or likely-secret checks fail.

All persistent paths use `SessionManager.getSessionDir()` and `getSessionId()`, which are present in both the fork and Pi 0.84.2. SoL-Pi creates no configurable storage-path surface.

The unpublished shared artifact layout is not read or migrated. Each session starts from its own `<sessionDir>/sol-pi/<sessionId>/` directory.

## Online Context Compact

Online Context Compact uses ordinary public `context` and `before_provider_request` handlers instead of fork-only post-transform observer methods. Public handlers run in extension load order, so the SoL-Pi entrypoint registers Online Context Compact after its other context transformers. A third-party transformer loaded later is outside the context-growth observation used by its estimate.

Pi does not expose its active retained-tail compaction setting through the public extension context. The standalone extension therefore uses the documented Pi 0.84.2 default of 20,000 tokens for its economic estimate. Its programmatic factory accepts an explicit matching value for a non-default Pi setting.

Pi 0.84.2's `ExtensionContext.compact()` aborts the active agent before it summarizes, and `agent_settled` fires only once a whole run has drained every turn, retry, auto-compaction, and queued continuation. A plan boundary that selects compaction therefore saves its plan and progress state, calls `ExtensionContext.abort()` to stop the run, and runs compaction from the `agent_settled` that stop produces. The handler awaits the compaction's own `onComplete`/`onError` callbacks. On success, the extension sends a hidden reminder through public `ExtensionAPI.sendMessage()` with `triggerTurn: true`, so Pi starts a new turn against the compacted context and rebuilds the plan even when the native summary omits that instruction.

A settlement barrier keeps the original `agent_settled` dispatch open until the triggered continuation settles. Print- and JSON-mode processes therefore complete the compact-and-continue sequence within the same Pi invocation; an outer driver does not need to resume the session or send `Continue working`. This continuation is armed only by a successful boundary compaction. Cancelling or exiting does not schedule one. A Pi build that never emits `agent_settled` starts no boundary compaction.

Pi 0.84.2 does not return a promise from `ExtensionAPI.sendMessage()`. The barrier is therefore verified for standalone SoL-Pi and depends on Pi starting the requested turn synchronously. A later-loaded third-party extension that performs long asynchronous work in its own `agent_settled` handler is outside this guarantee and needs an integration test with that extension set.

Pi reports the session as idle while an extension-requested manual compaction is running. SoL-Pi cancels `session_before_tree` during that interval to prevent tree navigation from moving the active leaf underneath the compaction. Navigation works normally after the compaction callback settles.

Online Context Compact reads `ExtensionContext.getContextUsage()` for both the context window and the provider-counted context size. When Pi reports no size — as it does between a compaction and the next answered request — the boundary falls back to its own estimate.

The standalone entry passes `cacheWriteReadRatio` from `sol-pi.json` directly into Online Context Compact's economic check. It does not inspect model price metadata. Changing models during a session does not change the ratio; users who want a different decision policy update the configuration and start a new session.

## Interactive TUI

The lightning savings treatment uses Pi 0.84.2's public `renderCall`,
`renderResult`, `ctx.ui.notify()`, and keyed `ctx.ui.setStatus()` APIs. It checks
`ctx.mode === "tui"` rather than `ctx.hasUI`, because RPC mode also reports UI
support. The renderer therefore changes only the interactive terminal display;
it does not change session messages, provider requests, tool results, JSON
events, print output, or RPC UI requests.

## Test doubles

The test suite drives every extension through the same public `ExtensionAPI` and `ExtensionContext` surface Pi provides, over a real public `SessionManager`, with no provider registered. That keeps the suite zero-spend and independent of the deleted Pi monorepo test harness. Suites that need a genuine session tree — branch order, compaction entries, custom entries, resume — use `SessionManager.inMemory()` or `SessionManager.create()` rather than reimplementing them.
