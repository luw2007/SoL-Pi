/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import type * as PiCodingAgent from "@earendil-works/pi-coding-agent";
import type { CompactOptions } from "@earendil-works/pi-coding-agent";
import { fauxAssistantMessage } from "@earendil-works/pi-ai/providers/faux";
import { describe, expect, it, vi } from "vitest";
import { createOnlineContextCompactExtension, POST_COMPACTION_PLAN_REMINDER, SKIPPED_COMPACTION_REMINDER } from "../src/sol-pi/extensions/online-context-compact/extension.ts";
import { restoreOnlineState } from "../src/sol-pi/extensions/online-context-compact/state.ts";
import { FakePi, FakeSessionManager, fakeContext } from "./helpers.ts";

// Make the extension take its omp path: compaction runs inline from turn_end.
vi.mock("@earendil-works/pi-coding-agent", async (importOriginal) => ({
	...(await importOriginal<typeof PiCodingAgent>()),
	CONFIG_DIR_NAME: ".omp",
}));

type Outcome = "success" | { onError: Error } | { reject: Error };

async function scenario(outcome: Outcome) {
	const manager = new FakeSessionManager();
	const messages = [{ role: "user" as const, content: "old ".repeat(4_000), timestamp: 1 }, fauxAssistantMessage("tail ".repeat(400))];
	for (const message of messages) manager.appendMessage(message);
	const pi = new FakePi(manager);
	createOnlineContextCompactExtension({ cacheWriteReadRatio: 12.5, keepRecentTokens: 1 })(pi.asExtensionApi());
	const abort = vi.fn();
	const compact = vi.fn((options: CompactOptions = {}) => {
		if (outcome === "success") {
			const entry = { type: "compaction", id: "compact-test", parentId: manager.getLeafId(), timestamp: new Date().toISOString(), summary: "memo", firstKeptEntryId: manager.entries[1]!.id, tokensBefore: 195_000 };
			void pi.emit("session_compact", { compactionEntry: entry, fromExtension: false }, ctx).then(() => options.onComplete?.(entry));
			return undefined;
		}
		// omp's compact() rejects instead of calling onError.
		if ("reject" in outcome) return Promise.reject(outcome.reject);
		options.onError?.(outcome.onError);
		return undefined;
	});
	// A stopped omp session never goes idle again by itself; a continuation after
	// cancel/error would have to come from the removed waitForIdle fallback.
	const ctx = fakeContext(manager, { abort, compact, isIdle: () => true,
		getContextUsage: () => ({ tokens: 195_000, contextWindow: 200_000, percent: 97.5 }) });
	await pi.emit("session_start", {}, ctx);
	await pi.emitContext(messages, ctx);
	async function boundary(id: string) {
		await pi.emit("before_provider_request", {}, ctx);
		const steps = [{ id, goal: "do work", status: "completed" }, { id: "remaining", goal: "remaining work", status: "pending" }];
		await pi.tool("update_plan").execute(`${id}-open`, { steps: steps.map((step) => step.id === id ? { ...step, status: "in_progress" } : step) }, undefined, undefined, ctx);
		await pi.emit("before_provider_request", {}, ctx);
		await pi.tool("update_plan").execute(id, { steps }, undefined, undefined, ctx);
		await pi.emit("turn_end", { message: fauxAssistantMessage("boundary"), toolResults: [{ toolCallId: id, toolName: "update_plan", isError: false }] }, ctx);
	}
	return { pi, manager, ctx, abort, compact, boundary };
}

describe("Online Context Compact on omp turn_end", () => {
	it("compacts inline and continues once after success", async () => {
		const { pi, manager, ctx, abort, compact, boundary } = await scenario("success");
		await boundary("first");
		expect(abort).not.toHaveBeenCalled();
		expect(compact).toHaveBeenCalledOnce();
		expect(pi.sentMessages).toHaveLength(1);
		expect(pi.sentMessages[0]?.message.content).toBe(POST_COMPACTION_PLAN_REMINDER);
		expect(restoreOnlineState(manager.entries).nativeCompactionCount).toBe(1);
		await pi.emit("agent_settled", {}, ctx);
		expect(compact).toHaveBeenCalledOnce();
	});

	it.each([
		"Nothing to compact (session too small)",
		"Already compacted",
		"Summarization failed: generation hit the token cap and the summary is incomplete",
	])("continues with the skipped reminder after recoverable refusal: %s", async (message) => {
		const { pi, manager, boundary } = await scenario({ onError: new Error(message) });
		await boundary("first");
		expect(pi.sentMessages).toHaveLength(1);
		expect(pi.sentMessages[0]?.message.content).toBe(SKIPPED_COMPACTION_REMINDER);
		expect(restoreOnlineState(manager.entries)).toMatchObject({ nativeCompactionCount: 0, cacheDebtTokens: 0, cacheDebtRepaymentTokens: 0 });
	});

	it.each([
		["onError cancelled", { onError: new Error("Compaction cancelled") }],
		["onError AbortError", { onError: Object.assign(new Error("stopped"), { name: "AbortError" }) }],
		["rejected cancelled", { reject: new Error("Compaction cancelled") }],
		["rejected AbortError", { reject: Object.assign(new Error("stopped"), { name: "AbortError" }) }],
	] as const)("does not continue after host cancellation (%s)", async (_name, outcome) => {
		const { pi, manager, ctx, compact, boundary } = await scenario(outcome);
		await expect(boundary("first")).resolves.toBeUndefined();
		expect(pi.sentMessages).toHaveLength(0);
		expect(restoreOnlineState(manager.entries)).toMatchObject({ nativeCompactionCount: 0, cacheDebtTokens: 0, cacheDebtRepaymentTokens: 0 });
		await pi.emit("agent_settled", {}, ctx);
		expect(compact).toHaveBeenCalledOnce();
		expect(pi.sentMessages).toHaveLength(0);
	});

	it.each([
		["onError", { onError: new Error("summarizer unavailable") }],
		["rejected", { reject: new Error("summarizer unavailable") }],
	] as const)("propagates genuine failures without continuing (%s)", async (_name, outcome) => {
		const { pi, manager, ctx, compact, boundary } = await scenario(outcome);
		await expect(boundary("first")).rejects.toThrow("summarizer unavailable");
		expect(pi.sentMessages).toHaveLength(0);
		expect(restoreOnlineState(manager.entries)).toMatchObject({ nativeCompactionCount: 0, cacheDebtTokens: 0, cacheDebtRepaymentTokens: 0 });
		await pi.emit("agent_settled", {}, ctx);
		expect(compact).toHaveBeenCalledOnce();
	});
});
