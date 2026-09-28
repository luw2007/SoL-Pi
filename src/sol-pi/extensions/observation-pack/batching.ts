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
 * a long idle gap, a model change, an upstream prefix change, or a fresh
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

export interface RequestTiming {
	/** Milliseconds since the most recent assistant message; `undefined` without a usable timestamp. */
	readonly gapMs: number | undefined;
	/** The current model differs from the one that produced the most recent assistant message. */
	readonly modelChanged: boolean;
}

export function requestTiming(messages: readonly AgentMessage[], model: ModelRef | undefined, now: number): RequestTiming {
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		const message = messages[index];
		if (message?.role !== "assistant") continue;
		const timestamp = (message as { timestamp?: unknown }).timestamp;
		const gapMs = typeof timestamp === "number" && Number.isFinite(timestamp) ? now - timestamp : undefined;
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
