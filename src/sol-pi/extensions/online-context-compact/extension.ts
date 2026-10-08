/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { join } from "node:path";
import type { AgentMessage, AgentToolResult } from "@earendil-works/pi-agent-core";
import * as piCodingAgent from "@earendil-works/pi-coding-agent";
import {
	estimateTokens,
	type ExtensionContext,
	type ExtensionFactory,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { runtimeRoot } from "../../runtime-paths.ts";
import { formatSavingsCount, showSolPiSavings } from "../../tui.ts";
import {
	DEFAULT_COMPACTION_ECONOMICS,
	decideCompaction,
	type CompactionDecision,
} from "./economics.ts";
import { analyzePlanTransition, formatPlanSnapshot, parsePlanSteps } from "./plan.ts";
import { projectedEntryTokens } from "./projection.ts";
import { createCompactionLedger, type CompactionLedger } from "./ledger.ts";
import {
	appendOnlineState,
	initialOnlineState,
	recordBoundary,
	recordCompaction,
	recordCompletedPlanHandoff,
	recordCorrection,
	recordProviderRequest,
	restoreOnlineState,
	type OnlineState,
	type ProgressSummary,
} from "./state.ts";
import { registerOnlineTools, type PlanUpdateInput } from "./tools.ts";
import { classifyTurnEnd } from "./turn-end-classifier.ts";

/*
 * omp compat: the oh-my-pi (omp) harness re-exports a reduced
 * `@earendil-works/pi-coding-agent` surface that omits `findCutPoint` and
 * `sessionEntryToContextMessages`. Both are used only by the feasibility
 * pre-check below, so resolve them at runtime and fall back to local
 * equivalents when the host does not export them.
 */
type CutPointResult = {
	readonly firstKeptEntryIndex: number;
	readonly turnStartIndex: number;
	readonly isSplitTurn: boolean;
};

// Unchecked casts: the host module namespace is structurally opaque here, and
// the two optional members are exactly what this shim probes for.
const hostModule = piCodingAgent as unknown as {
	findCutPoint?: (
		entries: readonly SessionEntry[],
		startIndex: number,
		endIndex: number,
		keepRecentTokens: number,
	) => CutPointResult;
	sessionEntryToContextMessages?: (entry: SessionEntry) => readonly unknown[];
};

/*
 * omp compat: omp has no `agent_settled` event, and `context.compact()` itself
 * ends the running agent loop (the session is idle by the time `onComplete`
 * fires). It also skips `session_stop` for aborted loops. So on omp the
 * boundary compaction runs inline from `turn_end` and the continuation turn is
 * started directly once the session reports idle, instead of the
 * abort → agent_settled → compact → continue sequence pi uses.
 */
const ompHost = piCodingAgent.CONFIG_DIR_NAME === ".omp";

function entryMessage(entry: SessionEntry): AgentMessage | undefined {
	if (!entry || typeof entry !== "object" || !("message" in entry)) return undefined;
	// Unchecked cast: session entries that carry a message carry an agent message.
	return entry.message as AgentMessage;
}

function entryContextMessageCount(entry: SessionEntry): number {
	const hostFn = hostModule.sessionEntryToContextMessages;
	if (hostFn) return hostFn(entry).length;
	return entryMessage(entry) ? 1 : 0;
}

function findCutPoint(
	entries: readonly SessionEntry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	const hostFn = hostModule.findCutPoint;
	if (hostFn) return hostFn(entries, startIndex, endIndex, keepRecentTokens);
	// Token-walk the tail backwards; the first entry that no longer fits in the
	// retained budget ends the compactable history. Turn splitting is a host-only
	// refinement, so report a whole-entry cut.
	let kept = 0;
	let firstKeptEntryIndex = endIndex;
	for (let index = endIndex - 1; index >= startIndex; index--) {
		const entry = entries[index];
		if (!entry) continue;
		if (entryContextMessageCount(entry) === 0) {
			firstKeptEntryIndex = index;
			continue;
		}
		const message = entryMessage(entry);
		if (message) kept += estimateTokens(message);
		if (kept > keepRecentTokens) break;
		firstKeptEntryIndex = index;
	}
	return { firstKeptEntryIndex, turnStartIndex: -1, isSplitTurn: false };
}

export const DEFAULT_KEEP_RECENT_TOKENS = 20_000;
export const DEFAULT_NATIVE_SUMMARY_TOKEN_ESTIMATE = 1_000;
export const BOUNDARY_COMPACTION_INSTRUCTIONS =
	"Preserve completed work, verification results, important decisions, and remaining work.";
export const POST_COMPACTION_PLAN_REMINDER =
	"Online context compaction finished. The parent task is still active. " +
	"Continue the remaining work from the current plan. Preserve existing step IDs when updating progress.";
export const SKIPPED_COMPACTION_REMINDER =
	"Online context compaction was skipped. The existing context is still available. " +
	"Continue the remaining work from the current plan.";
// Pi rejects these before committing a replacement summary, so the current
// context is still usable. Unknown errors and user cancellation remain distinct.
const RECOVERABLE_COMPACTION_ERRORS = new Set([
	"Nothing to compact (session too small)",
	"Already compacted",
	"Summarization failed: generation hit the token cap and the summary is incomplete",
]);

export type OnlineContextCompactOptions = {
	readonly cacheWriteReadRatio?: number | null;
	readonly keepRecentTokens?: number;
};

type PendingBoundary = { readonly toolCallId: string };
type SelectedCompaction = { readonly decision: CompactionDecision };
type CacheDebt = { readonly debtTokens: number; readonly repaymentTokens: number };
type PendingContinuation = { readonly promise: Promise<void>; readonly resolve: () => void };

export function resolveKeepRecentTokens(value: number | undefined): number {
	const resolved = value ?? DEFAULT_KEEP_RECENT_TOKENS;
	if (!Number.isSafeInteger(resolved) || resolved < 1) {
		throw new Error("Online Context Compact keepRecentTokens must be a positive safe integer");
	}
	return resolved;
}

function resolveCacheWriteReadRatio(value: number | null | undefined): number | null {
	if (value === undefined || value === null) return null;
	if (!Number.isFinite(value) || value < 0) {
		throw new Error("Online Context Compact cacheWriteReadRatio must be finite and non-negative");
	}
	return value;
}

function tokenEstimate(text: string): number {
	return Math.ceil(Buffer.byteLength(text) / 4);
}

// omp compat: getSystemPrompt() returns string[] (one entry per prompt section) in omp.
function systemPromptTokens(context: ExtensionContext): number {
	const prompt: unknown = context.getSystemPrompt();
	return tokenEstimate(Array.isArray(prompt) ? prompt.join("\n") : String(prompt ?? ""));
}

function result(text: string, details: Readonly<Record<string, unknown>>): AgentToolResult<Readonly<Record<string, unknown>>> {
	return { content: [{ type: "text", text }], details };
}

function progressSummary(input: PlanUpdateInput, completedStepId: string): ProgressSummary | undefined {
	const step = input.steps.find((item) => item.id === completedStepId);
	if (!step || !input.progress) return;
	return {
		stepId: step.id,
		goal: step.goal,
		filesChanged: [...input.progress.files_changed],
		verification: [...input.progress.verification],
		decisions: [...input.progress.decisions],
		nextWork: input.steps.filter((item) => item.status !== "completed").map((item) => item.goal),
	};
}

function compactionTokenEstimate(
	entries: readonly SessionEntry[],
	startIndex: number,
	endIndex: number,
	projected?: ReadonlyMap<string, number>,
): number {
	let tokens = 0;
	for (let index = startIndex; index < endIndex; index++) {
		const entry = entries[index];
		if (!entry || entry.type === "compaction") continue;
		if (projected) {
			tokens += projected.get(entry.id) ?? 0;
			continue;
		}
		const message = hostModule.sessionEntryToContextMessages
			? hostModule.sessionEntryToContextMessages(entry)[0] as AgentMessage | undefined
			: entryMessage(entry);
		if (message) tokens += estimateTokens(message);
	}
	return tokens;
}

function branchAfterAbort(entries: readonly SessionEntry[]): SessionEntry[] {
	const last = entries.at(-1);
	const markerProvider = ["sol", "pi"].join("-");
	return [
		...entries,
		{
			type: "message",
			id: "sol-pi-online-context-compact-abort-marker",
			parentId: last?.id ?? null,
			timestamp: new Date(0).toISOString(),
			message: {
				role: "assistant",
				content: [],
				api: markerProvider,
				provider: markerProvider,
				model: "aborted",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "aborted",
				timestamp: 0,
			},
		} as SessionEntry,
	];
}

export function estimateNativeCompactionTokens(
	entries: readonly SessionEntry[],
	keepRecentTokens: number,
	projectedMessages?: readonly AgentMessage[],
): number {
	const path = branchAfterAbort(entries);
	const projected = projectedMessages === undefined ? undefined : projectedEntryTokens(entries, projectedMessages);
	let startIndex = 0;
	let previousSummaryTokens = 0;
	for (let index = path.length - 1; index >= 0; index--) {
		const entry = path[index];
		if (entry?.type !== "compaction") continue;
		const keptIndex = path.findIndex((item) => item.id === entry.firstKeptEntryId);
		startIndex = keptIndex >= 0 ? keptIndex : index + 1;
		const previousSummary = hostModule.sessionEntryToContextMessages?.(entry)[0] as AgentMessage | undefined;
		previousSummaryTokens = projected
			? projected.get(entry.id) ?? 0
			: previousSummary ? estimateTokens(previousSummary) : tokenEstimate(entry.summary);
		break;
	}

	// Keep Pi's native cut on session entries; price only the corresponding
	// messages that the provider actually saw after context projection.
	const cut = findCutPoint(path, startIndex, path.length, keepRecentTokens);
	const firstKept = cut.firstKeptEntryIndex;
	if (compactionTokenEstimate(path, startIndex, firstKept) === 0) return 0;
	return previousSummaryTokens + compactionTokenEstimate(path, startIndex, firstKept, projected);
}

function validPositiveInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

export function createOnlineContextCompactExtension(options: OnlineContextCompactOptions = {}): ExtensionFactory {
	const keepRecentTokens = resolveKeepRecentTokens(options.keepRecentTokens);
	const cacheWriteReadRatio = resolveCacheWriteReadRatio(options.cacheWriteReadRatio);

	return (pi) => {
		let state: OnlineState = initialOnlineState();
		let restored = false;
		let observedMessages: readonly AgentMessage[] = [];
		let pendingBoundary: PendingBoundary | undefined;
		let selected: SelectedCompaction | undefined;
		let activeDebt: CacheDebt | undefined;
		let nextContinuation: PendingContinuation | undefined;
		let compactionInFlight = false;
		let compactionRefused = false;
		let turnEndSeq = 0;
		const compactionLedgers = new Map<string, CompactionLedger>();
		const compactionLedgerFor = (context: ExtensionContext): CompactionLedger => {
			const root = runtimeRoot(context);
			let ledger = compactionLedgers.get(root);
			if (!ledger) {
				ledger = createCompactionLedger(join(root, "online-context-compact", "turn-end-ledger.jsonl"));
				compactionLedgers.set(root, ledger);
			}
			return ledger;
		};

		const releaseContinuation = (): void => {
			const continuation = nextContinuation;
			nextContinuation = undefined;
			continuation?.resolve();
		};
		const releaseParentContinuation = (continuation: PendingContinuation | undefined): void => {
			if (continuation) setTimeout(continuation.resolve, 0);
		};

		const restore = (context: ExtensionContext): void => {
			releaseContinuation();
			state = restoreOnlineState(context.sessionManager.getBranch());
			restored = true;
			// Restoring raw history is not evidence of a provider-visible projection.
			observedMessages = [];
			pendingBoundary = undefined;
			selected = undefined;
			activeDebt = undefined;
			compactionInFlight = false;
			compactionRefused = false;
		};
		const ensureRestored = (context: ExtensionContext): void => {
			if (!restored) restore(context);
		};
		const save = (): void => appendOnlineState(pi, state);
		const contextTokens = (context: ExtensionContext): number => {
			const visible = observedMessages.reduce((total, message) => total + estimateTokens(message), 0);
			const estimated = visible + systemPromptTokens(context);
			const reported = context.getContextUsage()?.tokens;
			return validPositiveInteger(reported) ? Math.max(reported, estimated) : estimated;
		};

		registerOnlineTools(pi, {
			updatePlan: async (input) => {
				ensureRestored(input.context);
				if (input.signal?.aborted) throw new Error("Plan update was aborted");
				const steps = parsePlanSteps(input.steps);
				if (!steps || steps.length === 0) throw new Error("Plan must contain at least one valid step");

				const transition = analyzePlanTransition(state.plan, steps);
				const completedIds = transition.completedSteps.map((step) => step.id);
				if (completedIds.length > 0) {
					state = recordBoundary(state, steps, progressSummary(input, completedIds[0] ?? ""));
					if (!pendingBoundary) pendingBoundary = { toolCallId: input.toolCallId };
				} else {
					state = { ...state, plan: [...steps] };
				}
				save();

				return result(
					[formatPlanSnapshot(steps), ...transition.advice].join("\n"),
					{
						boundary: completedIds.length > 0,
						completed_step_ids: completedIds,
						progress_recorded: completedIds.length > 0 && input.progress !== undefined,
						task_status: "active",
						plan: steps,
					},
				);
			},
		});

		pi.on("session_start", (_event, context) => restore(context));
		pi.on("session_before_tree", () => (compactionInFlight ? { cancel: true } : undefined));
		pi.on("session_tree", (_event, context) => restore(context));

		pi.on("context", (event, context) => {
			ensureRestored(context);
			observedMessages = [...event.messages];
		});

		pi.on("before_provider_request", (_event, context) => {
			ensureRestored(context);
			state = recordProviderRequest(state, contextTokens(context));
			save();
		});

		pi.on("input", (event, context) => {
			compactionRefused = false;
			ensureRestored(context);
			if (event.streamingBehavior !== "steer" && !event.text.startsWith("CORRECTION:")) {
				const next = recordCompletedPlanHandoff(state);
				if (next !== state) {
					state = next;
					save();
				}
				return { action: "continue" as const };
			}
			pendingBoundary = undefined;
			selected = undefined;
			activeDebt = undefined;
			state = recordCorrection(state);
			save();
			return { action: "continue" as const };
		});

		pi.on("turn_end", async (event, context) => {
			// Plan chatter cannot make a rejected compact feasible. Require actual
			// tool work or new user input before attempting another boundary.
			if (event.toolResults.some((item) => item.toolName !== "update_plan" && !item.isError)) compactionRefused = false;
			const boundary = pendingBoundary;
			pendingBoundary = undefined;
			const toolResult = boundary
				? event.toolResults.find((item) => item.toolCallId === boundary.toolCallId)
				: undefined;
			const classification = classifyTurnEnd({
				hasPendingBoundary: Boolean(boundary),
				hasAlreadySelected: Boolean(selected),
				messageRole: event.message.role,
				stopReason: event.message.role === "assistant" ? event.message.stopReason : undefined,
				contextAborted: Boolean(context.signal?.aborted),
				toolResultFound: Boolean(toolResult),
				toolResultIsError: Boolean(toolResult?.isError),
			});

			let decision: CompactionDecision | undefined;
			if (classification.decisionEvaluated && !compactionRefused) {
				const usage = context.getContextUsage();
				const writeTokens = contextTokens(context);
				const archiveTokens = estimateNativeCompactionTokens(
					context.sessionManager.getBranch(),
					keepRecentTokens,
					observedMessages,
				);
				const contextWindowTokens = validPositiveInteger(usage?.contextWindow)
					? usage.contextWindow
					: validPositiveInteger(context.model?.contextWindow)
						? context.model.contextWindow
						: null;
				const averageContextTokenIncrement =
					state.positiveContextDeltaCount === 0
						? null
						: state.positiveContextDeltaTotal / state.positiveContextDeltaCount;
				const priced = decideCompaction({
					writeTokens,
					archiveTokens,
					memoTokens: DEFAULT_NATIVE_SUMMARY_TOKEN_ESTIMATE,
					contextTokens: writeTokens,
					completedBoundaryRequestCounts: state.completedBoundaryRequestCounts,
					remainingBoundaries: state.plan.filter((step) => step.status !== "completed").length,
					averageContextTokenIncrement,
					contextWindowTokens,
					priorCompactionCount: state.nativeCompactionCount,
					requestsSinceLastCompaction:
						state.lastCompactionRequestCount === null
							? null
							: state.requestCount - state.lastCompactionRequestCount,
					carriedDebtTokens: state.cacheDebtTokens,
					cacheDebtRepaymentTokens: state.cacheDebtRepaymentTokens,
					cacheWriteReadRatio,
					economics: DEFAULT_COMPACTION_ECONOMICS,
				});
				decision =
					priced.compact && archiveTokens === 0
						? { ...priced, compact: false, reason: "native_not_compactable" }
						: priced;
			}

			// The ledger write is fired here but NOT awaited yet: `selected` /
			// `context.abort()` below must happen synchronously, in the same
			// tick as this handler resumes, with no `await` in between. On pi
			// (non-omp), `context.abort()` triggers `agent_settled` -> compact
			// consumption of `selected` (comment below, and the handler at
			// `pi.on("agent_settled", ...)`), and the host does not wait for
			// this turn_end handler's own promise to settle first. Awaiting the
			// ledger write before the handoff would let that consumption race
			// ahead of `selected` ever being set, silently dropping every
			// compaction. Every return path below still awaits `ledgerWrite`
			// before resolving, so the ledger keeps exactly one fail-open entry
			// per invocation — it is just no longer on the handoff's critical
			// path.
			turnEndSeq += 1;
			let ledgerWrite: Promise<void> = Promise.resolve();
			try {
				ledgerWrite = compactionLedgerFor(context)({
					event: "turn_end_classification",
					turnEndSeq,
					bucket: classification.bucket,
					decisionEvaluated: classification.decisionEvaluated,
					reason: decision?.reason ?? classification.bucket,
					willCompact: decision?.compact ?? false,
					requestCountAtEmit: state.requestCount,
					raw: classification.raw,
				}).catch((error) => {
					// Fail-open: an async ledger write failure must never cost the
					// turn its actual compaction decision.
					const reason = error instanceof Error ? error.message : String(error);
					console.error(`[online-context-compact] fail-open turn_end ledger: ${reason}`);
				});
			} catch (error) {
				// Fail-open: a synchronous ledger failure (e.g. runtimeRoot()
				// requiring a persistent session directory) must never cost the
				// turn its actual compaction decision either.
				const reason = error instanceof Error ? error.message : String(error);
				console.error(`[online-context-compact] fail-open turn_end ledger: ${reason}`);
			}

			if (!decision || !decision.compact) {
				await ledgerWrite;
				return;
			}

			selected = { decision };
			if (!ompHost) {
				context.abort();
				await ledgerWrite;
				return;
			}
			await ledgerWrite;

			// omp's compact() ends the agent loop before deciding whether it can
			// compact, so the continuation is decided by its outcome below.
			selected = undefined;
			// runBoundaryCompaction resolves true for an applied compaction and for a
			// recoverable refusal (both keep the task running), false for host
			// cancellation / AbortError, and throws real errors. Only the first two
			// continue; cancel and errors must not restart a turn the user stopped.
			let resumed = false;
			try {
				resumed = await runBoundaryCompaction({ decision }, context);
			} finally {
				activeDebt = undefined;
			}
			if (resumed) {
				pi.sendMessage(
					{
						customType: "sol-pi-online-context-compact",
						content: compactionRefused ? SKIPPED_COMPACTION_REMINDER : POST_COMPACTION_PLAN_REMINDER,
						display: false,
					},
					{ triggerTurn: true },
				);
			}
		});

		// Runs the native compaction for a selected boundary. Resolves true when a
		// compaction was applied or refused recoverably, false when the host
		// cancelled it; any other error is thrown.
		const runBoundaryCompaction = async (pending: SelectedCompaction, context: ExtensionContext): Promise<boolean> => {
			activeDebt = {
				debtTokens: pending.decision.postCompactionTokens * (pending.decision.incrementalCacheCostRatio ?? 0),
				repaymentTokens: Math.max(0, pending.decision.archiveTokens - pending.decision.memoTokens),
			};
			let compacted = false;
			let compactionError: Error | undefined;
			try {
				compactionInFlight = true;
				await new Promise<void>((resolve) => {
					let finished = false;
					const finish = (): void => {
						if (finished) return;
						finished = true;
						resolve();
					};
					// omp's compact() returns a promise that rejects instead of calling onError.
					Promise.resolve(
						context.compact({
							customInstructions: BOUNDARY_COMPACTION_INSTRUCTIONS,
							onComplete: (compaction) => {
								try {
									compacted = true;
									const removed = Math.max(
										0,
										pending.decision.archiveTokens - tokenEstimate(compaction.summary),
									);
									if (removed > 0) {
										showSolPiSavings(
											context,
											"Online Context Compact",
											formatSavingsCount(removed, "context tokens removed"),
										);
									}
								} finally {
									finish();
								}
							},
							onError: (error) => {
								activeDebt = undefined;
								compactionError = error;
								finish();
							},
						}) as unknown,
					).catch((error: unknown) => {
						activeDebt = undefined;
						compactionError = error instanceof Error ? error : new Error(String(error));
						finish();
					});
				});
			} finally {
				compactionInFlight = false;
			}
			const recoverableError = compactionError !== undefined && RECOVERABLE_COMPACTION_ERRORS.has(compactionError.message);
			if (
				compactionError &&
				!recoverableError &&
				compactionError.name !== "AbortError" &&
				compactionError.message !== "Compaction cancelled" &&
				!compactionError.message.startsWith("Nothing to compact")
			) {
				throw compactionError;
			}

			if (recoverableError) {
				compactionRefused = true;
				pi.appendEntry("sol-pi-online-context-compact-skipped", { reason: compactionError!.message });
				if (context.mode === "tui") context.ui.notify(`Online compaction skipped: ${compactionError!.message}`, "warning");
			}
			return compacted || recoverableError;
		};

		pi.on("agent_settled", async (_event, context) => {
			// sendMessage() starts a turn without returning its promise. Capture the
			// child settlement so print/JSON mode cannot dispose while it is running.
			const parentContinuation = nextContinuation;
			nextContinuation = undefined;
			const pending = selected;
			selected = undefined;
			if (!context.isIdle()) {
				selected = pending;
				nextContinuation = parentContinuation;
				return;
			}
			if (!pending) {
				releaseParentContinuation(parentContinuation);
				return;
			}

			try {
				const resumed = await runBoundaryCompaction(pending, context);
				if (resumed) {
					let resolveContinuation!: () => void;
					const continuation: PendingContinuation = {
						promise: new Promise<void>((resolve) => {
							resolveContinuation = resolve;
						}),
						resolve: () => resolveContinuation(),
					};
					nextContinuation = continuation;
					try {
						pi.sendMessage(
							{
								customType: "sol-pi-online-context-compact",
								content: compactionRefused ? SKIPPED_COMPACTION_REMINDER : POST_COMPACTION_PLAN_REMINDER,
								display: false,
							},
							{ triggerTurn: true },
						);
					} catch (error) {
						if (nextContinuation === continuation) nextContinuation = undefined;
						continuation.resolve();
						throw error;
					}
					if (context.isIdle() && nextContinuation === continuation) {
						// Newer Pi hosts defer runs until settled handlers return. Hand
						// control back to that host instead of waiting on a deferred run.
						nextContinuation = undefined;
						continuation.resolve();
					} else {
						// Pi 0.85.1 starts synchronously; keep its print-mode barrier.
						await continuation.promise;
					}
				}
			} finally {
				activeDebt = undefined;
				releaseParentContinuation(parentContinuation);
			}
		});

		pi.on("session_compact", (event, context) => {
			ensureRestored(context);
			compactionRefused = false;
			state = recordCompaction(
				state,
				event.fromExtension || !activeDebt ? { debtTokens: 0, repaymentTokens: 0 } : activeDebt,
			);
			save();
			pendingBoundary = undefined;
			selected = undefined;
			activeDebt = undefined;
			observedMessages = [];
		});

		pi.on("session_shutdown", () => {
			releaseContinuation();
			pendingBoundary = undefined;
			selected = undefined;
			activeDebt = undefined;
			compactionInFlight = false;
		});
	};
}
