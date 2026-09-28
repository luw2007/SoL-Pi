/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * ObservationPack - keep large tool results reachable without replaying them.
 *
 * A large tool result is sent in full for its first few provider requests, then
 * replaced with a short, stable placeholder for every later request. The
 * original bytes are archived by observation id outside the provider context,
 * and the agent pulls exact pages back with the registered `obs_recall` tool.
 *
 * Each swap edits the middle of the prompt and breaks the provider prompt
 * cache from there, so eligible results wait as full text until a batch swap
 * pays off or the cache is already cold (see `batching.ts`). A batch threshold
 * of `0` keeps the original one-by-one swap.
 *
 * The mechanism never edits history in place. It rewrites only at the
 * projection layer (`pi.on("context")`), so the stored session stays intact and
 * recall keeps working after native compaction or a session resume.
 *
 * Storage lives under the active Pi session directory.
 */

import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { runtimeRoot } from "../../runtime-paths.ts";
import { formatSavingsCount, renderSolPiTool, showSolPiSavings } from "../../tui.ts";
import {
	currentModel,
	DEFAULT_BATCH_THRESHOLD_TOKENS,
	DEFAULT_COLD_GAP_MS,
	type FlushReason,
	firstChangedIndex,
	messageFingerprint,
	type RequestTiming,
	requestTiming,
} from "./batching.ts";
import { createLedger, type Ledger } from "./ledger.ts";
import {
	countLines,
	createObservation,
	ensureStored,
	estimateTokens,
	FULL_SENDS,
	isObservationId,
	isPureTextResult,
	type Observation,
	observationPath,
	placeholderFor,
	type RecallChunk,
	readRecallChunk,
} from "./observation.ts";

const RECALL_MAX_BYTES = 16 * 1024;
const RECALL_MAX_LINES = 400;
const RECALL_HEADER_RESERVE_BYTES = 512;
const RECALL_HEADER_LINES = 2;

const RECALL_LIMITS = {
	maxBytes: RECALL_MAX_BYTES - RECALL_HEADER_RESERVE_BYTES,
	maxLines: RECALL_MAX_LINES - RECALL_HEADER_LINES,
};

export interface ObservationPackOptions {
	/** Σ removable pending tokens that triggers a batch swap; `0` keeps the legacy immediate swap. */
	readonly batchThresholdTokens?: number;
	/** Idle gap since the last assistant message after which pending observations swap. */
	readonly coldGapMs?: number;
	/** Append one prefix/flush diagnostics row per `context` call. */
	readonly prefixDiagnostics?: boolean;
	/** Clock for the cold-gap rule (epoch milliseconds). */
	readonly now?: () => number;
}

/** Per session root state, kept for the life of this process. */
interface RootState {
	/** Observations already sent as placeholders; they never flip back to full text. */
	readonly placeholders: Set<string>;
	/** Ledger rows written for `loggedRequest`, so a repeated `context` call adds no duplicates. */
	readonly logged: Set<string>;
	loggedRequest: number;
	/** What the previous request of this root sent, for prefix-change detection. */
	previous?: { readonly request: number; readonly fingerprints: readonly string[] };
}

/** One large pure-text tool result seen in the current request. */
interface Candidate {
	readonly index: number;
	readonly observation: Observation;
	readonly sendNumber: number;
	readonly kind: "full" | "placeholder" | "pending";
	readonly swap?: {
		readonly message: AgentMessage;
		readonly bytes: number;
		readonly tokens: number;
		readonly removedTokens: number;
	};
}

/** Signals computed from the projection this root sends before any new swap. */
interface RequestView {
	readonly fingerprints: string[];
	readonly previous: RootState["previous"];
	readonly firstChanged: number | undefined;
	readonly timing: RequestTiming;
}

function failOpen(error: unknown, scope: string): void {
	const reason = error instanceof Error ? error.message : String(error);
	console.error(`[observationpack] fail-open for ${scope}: ${reason}`);
}

export function createObservationPackExtension(options: ObservationPackOptions = {}): ExtensionFactory {
	const batchThresholdTokens = options.batchThresholdTokens ?? DEFAULT_BATCH_THRESHOLD_TOKENS;
	const coldGapMs = options.coldGapMs ?? DEFAULT_COLD_GAP_MS;
	const prefixDiagnostics = options.prefixDiagnostics ?? false;
	const now = options.now ?? Date.now;

	return (pi: ExtensionAPI) => {
		const sentCounts = new Map<string, number>();
		const states = new Map<string, RootState>();
		const ledgers = new Map<string, Ledger>();
		const ledgerAt = (root: string, name: string): Ledger => {
			const path = join(root, "observation-pack", name);
			let ledger = ledgers.get(path);
			if (!ledger) {
				ledger = createLedger(path);
				ledgers.set(path, ledger);
			}
			return ledger;
		};
		const ledgerFor = (ctx: ExtensionContext): Ledger => ledgerAt(runtimeRoot(ctx), "ledger.jsonl");
		const stateFor = (root: string): RootState => {
			let state = states.get(root);
			if (!state) {
				state = { placeholders: new Set(), logged: new Set(), loggedRequest: 0 };
				states.set(root, state);
			}
			return state;
		};
		const recordOnce = async (
			root: string,
			state: RootState,
			entry: { readonly event: string; readonly id: string; readonly request: number } & Record<string, unknown>,
		): Promise<void> => {
			if (state.loggedRequest !== entry.request) {
				state.loggedRequest = entry.request;
				state.logged.clear();
			}
			const key = `${entry.event}\0${entry.id}`;
			if (state.logged.has(key)) return;
			await ledgerAt(root, "ledger.jsonl")(entry);
			state.logged.add(key);
		};
		const inspectRequest = (state: RootState, sent: readonly AgentMessage[], ctx: ExtensionContext): RequestView => {
			const fingerprints = sent.map(messageFingerprint);
			const previous = state.previous;
			return {
				fingerprints,
				previous,
				firstChanged: previous ? firstChangedIndex(previous.fingerprints, fingerprints) : undefined,
				timing: requestTiming(sent, currentModel(ctx), now()),
			};
		};
		const writeDiagnostics = async (
			root: string,
			request: number,
			view: RequestView,
			pending: { readonly count: number; readonly tokens: number },
			flushReason: FlushReason | undefined,
			flushedCount: number,
		): Promise<void> => {
			try {
				await ledgerAt(root, "prefix-ledger.jsonl")({
					request,
					messageCount: view.fingerprints.length,
					prevMessageCount: view.previous?.fingerprints.length ?? null,
					firstChangedIndex: view.firstChanged ?? null,
					gapMs: view.timing.gapMs ?? null,
					modelChanged: view.timing.modelChanged,
					repeat: view.previous?.request === request,
					pendingCount: pending.count,
					pendingTokens: pending.tokens,
					flush: flushedCount > 0,
					flushReason: flushedCount > 0 ? (flushReason ?? null) : null,
					flushedCount,
				});
			} catch (error) {
				failOpen(error, "prefix diagnostics");
			}
		};

		pi.registerTool({
			name: "obs_recall",
			label: "Recall Observation",
			description: "Read a stored large tool result by observation id and byte offset.",
			promptSnippet: "Recall a paged excerpt from a previously replaced large tool result",
			renderShell: "self",
			parameters: Type.Object({
				id: Type.String({ description: "Observation id from a placeholder" }),
				offset: Type.Optional(Type.Integer({ minimum: 0, description: "Byte offset, default 0" })),
			}),
			async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
				if (!isObservationId(params.id)) throw new Error(`Unknown observation id: ${params.id}`);
				const offset = params.offset ?? 0;
				let chunk: RecallChunk;
				try {
					chunk = await readRecallChunk(observationPath(runtimeRoot(ctx), params.id), offset, RECALL_LIMITS);
				} catch (error) {
					if (error instanceof Error && "code" in error && error.code === "ENOENT") {
						throw new Error(`Unknown observation id: ${params.id}`);
					}
					throw error;
				}
				const header = [
					`[obs_recall id=${params.id} offset=${offset} next_offset=${chunk.nextOffset} eof=${chunk.eof}]`,
					`[chunk_bytes=${chunk.bytes} chunk_lines=${chunk.lines}; use next_offset to continue]`,
				].join("\n");
				const content = `${header}\n${chunk.text}`;
				if (Buffer.byteLength(content, "utf8") > RECALL_MAX_BYTES || countLines(content) > RECALL_MAX_LINES) {
					throw new Error("Recall output exceeded its hard limit");
				}
				await ledgerFor(ctx)({
					event: "recall",
					id: params.id,
					offset,
					bytes: chunk.bytes,
					lines: chunk.lines,
					nextOffset: chunk.nextOffset,
					eof: chunk.eof,
				});
				return {
					content: [{ type: "text", text: content }],
					details: {
						id: params.id,
						offset,
						bytes: chunk.bytes,
						lines: chunk.lines,
						nextOffset: chunk.nextOffset,
						eof: chunk.eof,
					},
				};
			},
			renderCall(params, theme) {
				const offset = params.offset ?? 0;
				const base = new Text(theme.fg("dim", `Recall ${params.id} from byte ${offset}`), 0, 0);
				return renderSolPiTool(theme, "Observation Pack", "full observation replay avoided", base);
			},
			renderResult(result, { isPartial }, theme) {
				const details = result.details as { bytes?: number; lines?: number } | undefined;
				const base = new Text(
					theme.fg(
						isPartial ? "warning" : "dim",
						isPartial
							? "Recalling the requested slice..."
							: `Recalled ${details?.bytes ?? 0} bytes across ${details?.lines ?? 0} lines`,
					),
					0,
					0,
				);
				return renderSolPiTool(theme, "Observation Pack", "full observation replay avoided", base);
			},
		});

		const projectLegacy = async (
			messages: readonly AgentMessage[],
			ctx: ExtensionContext,
			root: string,
			state: RootState,
			priorAssistantCounts: readonly number[],
			requestIndex: number,
		): Promise<AgentMessage[]> => {
			const projected = [...messages];
			const swapped: number[] = [];
			for (let index = 0; index < messages.length; index += 1) {
				const message = messages[index];
				if (!message || !isPureTextResult(message)) continue;

				try {
					const observation = createObservation(message, root);
					if (!observation) continue;
					await ensureStored(observation);

					const sendCountKey = `${root}\0${observation.id}`;
					const previousSends = sentCounts.get(sendCountKey) ?? priorAssistantCounts[index] ?? 0;
					if (previousSends < FULL_SENDS) {
						await recordOnce(root, state, {
							event: "full",
							id: observation.id,
							request: requestIndex,
							tool: observation.toolName,
							originalBytes: observation.bytes,
							originalLines: observation.lines,
							originalTokens: observation.tokens,
							contentHash: observation.contentHash,
						});
						sentCounts.set(sendCountKey, previousSends + 1);
						continue;
					}

					const placeholder = placeholderFor(observation);
					const placeholderTokens = estimateTokens(placeholder);
					const removedTokens = Math.max(0, observation.tokens - placeholderTokens);
					const first = !state.placeholders.has(observation.id);
					await recordOnce(root, state, {
						event: "placeholder",
						id: observation.id,
						request: requestIndex,
						sendNumber: previousSends + 1,
						tool: observation.toolName,
						originalBytes: observation.bytes,
						originalLines: observation.lines,
						originalTokens: observation.tokens,
						placeholderBytes: Buffer.byteLength(placeholder, "utf8"),
						placeholderTokens,
						removedTokens,
						...(first ? { flushReason: "legacy" satisfies FlushReason } : {}),
					});
					if (previousSends === FULL_SENDS) {
						showSolPiSavings(
							ctx,
							"Observation Pack",
							formatSavingsCount(removedTokens, "context tokens avoided"),
						);
					}
					projected[index] = { ...message, content: [{ type: "text", text: placeholder }] };
					sentCounts.set(sendCountKey, previousSends + 1);
					if (first) {
						state.placeholders.add(observation.id);
						swapped.push(index);
					}
				} catch (error) {
					// Fail open: a packing failure must never cost the agent its observation.
					failOpen(error, "tool result");
				}
			}

			if (prefixDiagnostics) {
				try {
					const beforeSwap = [...projected];
					for (const index of swapped) beforeSwap[index] = messages[index] as AgentMessage;
					const view = inspectRequest(state, beforeSwap, ctx);
					await writeDiagnostics(root, requestIndex, view, { count: 0, tokens: 0 }, "legacy", swapped.length);
					state.previous = { request: requestIndex, fingerprints: projected.map(messageFingerprint) };
				} catch (error) {
					failOpen(error, "prefix diagnostics");
				}
			}
			return projected;
		};

		const projectBatched = async (
			messages: readonly AgentMessage[],
			ctx: ExtensionContext,
			root: string,
			state: RootState,
			priorAssistantCounts: readonly number[],
			requestIndex: number,
		): Promise<AgentMessage[]> => {
			// Classify every large result. The send count is derived from history,
			// so a repeated `context` call for the same request cannot advance it.
			const candidates: Candidate[] = [];
			for (let index = 0; index < messages.length; index += 1) {
				const message = messages[index];
				if (!message || !isPureTextResult(message)) continue;
				try {
					const observation = createObservation(message, root);
					if (!observation) continue;
					await ensureStored(observation);
					const previousSends = priorAssistantCounts[index] ?? 0;
					const sendNumber = previousSends + 1;
					if (previousSends < FULL_SENDS) {
						candidates.push({ index, observation, sendNumber, kind: "full" });
						continue;
					}
					const text = placeholderFor(observation);
					const tokens = estimateTokens(text);
					candidates.push({
						index,
						observation,
						sendNumber,
						kind: state.placeholders.has(observation.id) ? "placeholder" : "pending",
						swap: {
							message: { ...message, content: [{ type: "text", text }] },
							bytes: Buffer.byteLength(text, "utf8"),
							tokens,
							removedTokens: Math.max(0, observation.tokens - tokens),
						},
					});
				} catch (error) {
					failOpen(error, "tool result");
				}
			}

			// Decide which pending observations swap on this request.
			const pending = candidates.filter((candidate) => candidate.kind === "pending");
			const pendingTokens = pending.reduce((total, candidate) => total + (candidate.swap?.removedTokens ?? 0), 0);
			const beforeFlush = [...messages];
			for (const candidate of candidates) {
				if (candidate.kind === "placeholder" && candidate.swap) beforeFlush[candidate.index] = candidate.swap.message;
			}
			let view: RequestView | undefined;
			let flushReason: FlushReason | undefined;
			let flushing = new Set<number>();
			try {
				view = inspectRequest(state, beforeFlush, ctx);
				const { gapMs, modelChanged } = view.timing;
				const flushAll: FlushReason | undefined = !view.previous
					? "process-start"
					: modelChanged
						? "model-change"
						: gapMs !== undefined && gapMs >= coldGapMs
							? "cold-gap"
							: pendingTokens >= batchThresholdTokens
								? "threshold"
								: undefined;
				if (flushAll) {
					flushReason = flushAll;
					flushing = new Set(pending.map((candidate) => candidate.index));
				} else if (view.firstChanged !== undefined) {
					// The cache already breaks at firstChanged; swapping anything earlier would move the break up.
					const from = view.firstChanged;
					flushReason = "prefix-changed";
					flushing = new Set(pending.flatMap((candidate) => (candidate.index >= from ? [candidate.index] : [])));
				}
			} catch (error) {
				failOpen(error, "observation batching");
				flushReason = undefined;
				flushing = new Set();
			}

			// Project and record in message order.
			const projected = [...messages];
			let flushedCount = 0;
			for (const candidate of candidates) {
				const { index, observation, sendNumber, swap } = candidate;
				try {
					const swapNow = candidate.kind === "placeholder" || flushing.has(index);
					if (!swap || !swapNow) {
						await recordOnce(root, state, {
							event: "full",
							id: observation.id,
							request: requestIndex,
							tool: observation.toolName,
							originalBytes: observation.bytes,
							originalLines: observation.lines,
							originalTokens: observation.tokens,
							contentHash: observation.contentHash,
							...(candidate.kind === "pending" ? { deferred: true } : {}),
						});
						continue;
					}
					const first = candidate.kind === "pending";
					await recordOnce(root, state, {
						event: "placeholder",
						id: observation.id,
						request: requestIndex,
						sendNumber,
						tool: observation.toolName,
						originalBytes: observation.bytes,
						originalLines: observation.lines,
						originalTokens: observation.tokens,
						placeholderBytes: swap.bytes,
						placeholderTokens: swap.tokens,
						removedTokens: swap.removedTokens,
						...(first ? { flushReason } : {}),
					});
					projected[index] = swap.message;
					if (first) {
						state.placeholders.add(observation.id);
						flushedCount += 1;
						showSolPiSavings(
							ctx,
							"Observation Pack",
							formatSavingsCount(swap.removedTokens, "context tokens avoided"),
						);
					}
				} catch (error) {
					// Fail open: a packing failure must never cost the agent its observation.
					failOpen(error, "tool result");
				}
			}

			if (view) {
				try {
					const fingerprints = [...view.fingerprints];
					for (const { index } of candidates) {
						if (projected[index] !== beforeFlush[index]) fingerprints[index] = messageFingerprint(projected[index]);
					}
					state.previous = { request: requestIndex, fingerprints };
				} catch (error) {
					failOpen(error, "observation batching");
				}
				if (prefixDiagnostics) {
					await writeDiagnostics(
						root,
						requestIndex,
						view,
						{ count: pending.length, tokens: pendingTokens },
						flushReason,
						flushedCount,
					);
				}
			}
			return projected;
		};

		pi.on("context", async (event, ctx: ExtensionContext) => {
			const root = runtimeRoot(ctx);
			const state = stateFor(root);
			// How many provider requests each message has already been part of,
			// counted by the assistant messages that follow it.
			const priorAssistantCounts = new Array<number>(event.messages.length);
			let assistantCount = 0;

			for (let index = event.messages.length - 1; index >= 0; index -= 1) {
				priorAssistantCounts[index] = assistantCount;
				if (event.messages[index]?.role === "assistant") assistantCount += 1;
			}

			const requestIndex = assistantCount + 1;
			const project = batchThresholdTokens === 0 ? projectLegacy : projectBatched;
			return { messages: await project(event.messages, ctx, root, state, priorAssistantCounts, requestIndex) };
		});
	};
}

export { DEFAULT_BATCH_THRESHOLD_TOKENS, DEFAULT_COLD_GAP_MS, type FlushReason } from "./batching.ts";
export {
	createObservation,
	FULL_SENDS,
	type Observation,
	PLACEHOLDER_EXCERPT_BYTES,
	placeholderFor,
	THRESHOLD_BYTES,
} from "./observation.ts";

export function registerObservationPack(pi: ExtensionAPI, options: ObservationPackOptions = {}): void {
	createObservationPackExtension(options)(pi);
}

export default registerObservationPack;
