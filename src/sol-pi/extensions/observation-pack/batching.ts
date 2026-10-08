/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * Cache-aware batching signals for ObservationPack.
 *
 * Swapping a large result for its placeholder edits the middle of the prompt,
 * which breaks the provider prompt cache from that point on. Eligible
 * observations therefore wait as "pending" (still sent in full) until a swap is
 * worth it or the cache is already cold: enough removable tokens accumulated,
 * a long idle gap (measured from the last assistant message or a later host
 * cache-warming refresh), a model change, an upstream prefix change, or a fresh
 * process. Every helper here is pure so the hook stays easy to test.
 */

import { createHash } from "node:crypto";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

/** Σ removable pending tokens that triggers a batch swap; `0` restores the legacy immediate swap. */
export const DEFAULT_BATCH_THRESHOLD_TOKENS = 20_000;
/** Idle gap after which the provider cache is assumed cold. */
export const DEFAULT_COLD_GAP_MS = 300_000;

export type FlushReason = "threshold" | "cold-gap" | "model-change" | "prefix-changed" | "process-start" | "legacy";

/** Cheap content fingerprint of one projected message (symbol-keyed host metadata is ignored). */
export function messageFingerprint(message: AgentMessage | undefined): string {
	return createHash("sha1")
		.update(JSON.stringify(message) ?? "")
		.digest("base64");
}

/**
 * First index where the current projection differs from the previous request's,
 * or where messages were removed; `undefined` when the previous projection is an
 * unchanged prefix of the current one.
 */
export function firstChangedIndex(previous: readonly string[], current: readonly string[]): number | undefined {
	const shared = Math.min(previous.length, current.length);
	for (let index = 0; index < shared; index += 1) {
		if (previous[index] !== current[index]) return index;
	}
	return current.length < previous.length ? current.length : undefined;
}

export interface ModelRef {
	readonly provider: string;
	readonly id: string;
}

/** The host's current model, when exposed; never throws (some hosts omit or guard it). */
export function currentModel(ctx: ExtensionContext): ModelRef | undefined {
	try {
		const model = (ctx as { model?: unknown }).model as { provider?: unknown; id?: unknown } | undefined;
		if (!model || typeof model.provider !== "string" || typeof model.id !== "string") return undefined;
		return { provider: model.provider, id: model.id };
	} catch {
		return undefined;
	}
}

/** A replayed side answer: text-only content and no usage (omp rebuilds earlier `/btw` answers this way). */
function isReplayedAnswer(message: AgentMessage): boolean {
	const { content, usage, stopReason } = message as { content?: unknown; usage?: Record<string, unknown>; stopReason?: unknown };
	// An interrupted main-stream response (Esc / provider error) also has zero usage; it is a request, not a replayed answer.
	if (stopReason === "aborted" || stopReason === "error") return false;
	if (!Array.isArray(content) || !content.every((block) => (block as { type?: unknown } | null)?.type === "text")) {
		return false;
	}
	return ["input", "output", "cacheRead", "cacheWrite", "totalTokens"].every((key) => Number(usage?.[key] ?? 0) === 0);
}

/**
 * Index of the `developer` message that opens an omp side-turn suffix, or `undefined`.
 *
 * omp side turns (`/btw`, `session.runEphemeralTurn`) run the main root's `context`
 * hook on `[...main messages, developer notice, ...(user, replayed answer)*, user]`.
 * The replayed answers would count as extra sends of every main observation, so the
 * suffix is recognised only when it holds at least one of them; without one it adds
 * no assistant and changes nothing.
 */
export function sideTurnStart(messages: readonly AgentMessage[]): number | undefined {
	if (messages.at(-1)?.role !== "user") return undefined;
	let answers = 0;
	for (let index = messages.length - 2; index >= 0; index -= 1) {
		const message = messages[index] as AgentMessage;
		const role = (message as { role?: unknown }).role;
		if (role === "developer") return answers > 0 ? index : undefined;
		if (role === "user") continue;
		if (role !== "assistant" || !isReplayedAnswer(message)) return undefined;
		answers += 1;
	}
	return undefined;
}

/** Session entries walked back from the leaf when looking for a cache-warming refresh. */
const CACHE_WARM_SCAN_LIMIT = 256;

/**
 * Epoch milliseconds of the latest prompt-cache warming refresh recorded after the
 * most recent assistant message of the session branch, or `undefined`.
 *
 * omp (`providers.cacheWarming`) replays the last projected request shortly before
 * its cache entry expires and records it only as a `model_usage` session entry with
 * purpose `cache-warm` (or `cache-warm:extension-override`), not as a message. Such a
 * refresh keeps the cache warm, so the cold gap is measured from it. A refresh that
 * read or wrote no cache tokens does not count. Hosts without warming or without
 * `getLeafEntry`/`getEntry` yield `undefined`; never throws.
 */
export function lastCacheWarmAt(ctx: ExtensionContext): number | undefined {
	try {
		const manager = (ctx as { sessionManager?: unknown }).sessionManager as
			| { getLeafEntry?: () => unknown; getEntry?: (id: string) => unknown }
			| undefined;
		if (typeof manager?.getLeafEntry !== "function" || typeof manager.getEntry !== "function") return undefined;
		let entry = manager.getLeafEntry();
		for (let steps = 0; entry && steps < CACHE_WARM_SCAN_LIMIT; steps += 1) {
			const current = entry as {
				type?: unknown;
				parentId?: unknown;
				purpose?: unknown;
				timestamp?: unknown;
				usage?: { cacheRead?: unknown; cacheWrite?: unknown };
				message?: { role?: unknown };
			};
			if (current.type === "message" && current.message?.role === "assistant") return undefined;
			if (
				current.type === "model_usage" &&
				typeof current.purpose === "string" &&
				current.purpose.startsWith("cache-warm") &&
				typeof current.timestamp === "string"
			) {
				const cacheTokens = Number(current.usage?.cacheRead ?? 0) + Number(current.usage?.cacheWrite ?? 0);
				const at = Date.parse(current.timestamp);
				if (cacheTokens > 0 && Number.isFinite(at)) return at;
			}
			entry = typeof current.parentId === "string" ? manager.getEntry(current.parentId) : undefined;
		}
		return undefined;
	} catch {
		return undefined;
	}
}

export interface RequestTiming {
	/**
	 * Milliseconds since the most recent assistant message, or since a later cache-warming
	 * refresh; `undefined` without a usable assistant timestamp.
	 */
	readonly gapMs: number | undefined;
	/** The current model differs from the one that produced the most recent assistant message. */
	readonly modelChanged: boolean;
}

export function requestTiming(
	messages: readonly AgentMessage[],
	model: ModelRef | undefined,
	now: number,
	warmedAt?: number,
): RequestTiming {
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		const message = messages[index];
		if (message?.role !== "assistant") continue;
		const timestamp = (message as { timestamp?: unknown }).timestamp;
		const gapMs =
			typeof timestamp === "number" && Number.isFinite(timestamp)
				? now - Math.max(timestamp, warmedAt ?? Number.NEGATIVE_INFINITY)
				: undefined;
		const { provider, model: modelId } = message as { provider?: unknown; model?: unknown };
		const modelChanged =
			model !== undefined &&
			typeof provider === "string" &&
			typeof modelId === "string" &&
			(provider !== model.provider || modelId !== model.id);
		return { gapMs, modelChanged };
	}
	return { gapMs: undefined, modelChanged: false };
}
