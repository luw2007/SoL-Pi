/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionContext, SessionEntry } from "@earendil-works/pi-coding-agent";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG } from "../src/sol-pi/config.ts";
import { registerConfiguredFeatures } from "../src/sol-pi/index.ts";
import {
	firstChangedIndex,
	lastCacheWarmAt,
	requestTiming,
	sideTurnStart,
} from "../src/sol-pi/extensions/observation-pack/batching.ts";
import {
	createObservation,
	createObservationPackExtension,
	FULL_SENDS,
	type ObservationPackOptions,
	placeholderFor,
	THRESHOLD_BYTES,
} from "../src/sol-pi/extensions/observation-pack/index.ts";
import { FakePi, FakeSessionManager, fakeContext } from "./helpers.ts";

const SESSION_ID = "session-a";
const PROVIDER = "provider-a";
const MODEL = "model-a";
const roots: string[] = [];

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
	vi.restoreAllMocks();
});

async function sessionRoot(): Promise<string> {
	const value = await mkdtemp(join(tmpdir(), "observationpack-batching-"));
	roots.push(value);
	return value;
}

function user(text: string): AgentMessage {
	return { role: "user", content: text, timestamp: 0 } as AgentMessage;
}

function assistant(timestamp: number, text = "calling a tool", provider = PROVIDER, model = MODEL): AgentMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-completions",
		provider,
		model,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
		stopReason: "toolUse",
		timestamp,
	} as unknown as AgentMessage;
}

/** Just over the size threshold: roughly 2,250 removable tokens once replaced. */
function result(step: number): ToolResultMessage {
	const line = `observation ${step} line\n`;
	const text = `result ${step}\n${line.repeat(Math.ceil((THRESHOLD_BYTES + 64) / line.length))}`;
	return {
		role: "toolResult",
		toolCallId: `call-${step}`,
		toolName: "read",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: step,
	};
}

function removableTokens(message: ToolResultMessage): number {
	const observation = createObservation(message, "/unused")!;
	return observation.tokens - Math.ceil(placeholderFor(observation).length / 4);
}

/** History after `steps` tool rounds: [user, a1, r1, a2, r2, ...]; assistant k is stamped k seconds. */
function history(steps: number): AgentMessage[] {
	const messages: AgentMessage[] = [user("start")];
	for (let step = 1; step <= steps; step += 1) messages.push(assistant(step * 1_000), result(step));
	return messages;
}

function resultIndex(step: number): number {
	return step * 2;
}

function textOf(message: AgentMessage | undefined): string {
	if (!message || message.role !== "toolResult") throw new Error("expected tool result");
	return message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

function isPlaceholder(message: AgentMessage | undefined): boolean {
	return textOf(message).startsWith("[large tool result replaced");
}

interface Harness {
	readonly pi: FakePi;
	readonly clock: { now: number };
	readonly sessionDir: string;
	context(overrides?: Partial<ExtensionContext>): ExtensionContext;
	send(messages: readonly AgentMessage[], overrides?: Partial<ExtensionContext>): Promise<AgentMessage[]>;
	ledger(): Promise<Record<string, unknown>[]>;
	prefixLedger(): Promise<Record<string, unknown>[]>;
}

async function harness(options: ObservationPackOptions = {}, sessionDir?: string): Promise<Harness> {
	const dir = sessionDir ?? (await sessionRoot());
	const clock = { now: 0 };
	const pi = new FakePi();
	createObservationPackExtension({ batchThresholdTokens: 5_000, now: () => clock.now, ...options })(pi.asExtensionApi());
	const readJsonl = async (name: string) => {
		const path = join(dir, "sol-pi", SESSION_ID, "observation-pack", name);
		const text = await readFile(path, "utf8").catch(() => "");
		return text
			.split("\n")
			.filter(Boolean)
			.map((line) => JSON.parse(line) as Record<string, unknown>);
	};
	const context = (overrides: Partial<ExtensionContext> = {}) =>
		fakeContext(dir, { model: { provider: PROVIDER, id: MODEL } as never, ...overrides });
	return {
		pi,
		clock,
		sessionDir: dir,
		context,
		async send(messages, overrides) {
			// Warm cache by default: one second after the latest assistant message.
			const last = messages.findLast((message) => message.role === "assistant");
			clock.now = Math.max(clock.now, ((last as { timestamp?: number } | undefined)?.timestamp ?? 0) + 1_000);
			return pi.emitContext(messages, context(overrides));
		},
		ledger: () => readJsonl("ledger.jsonl"),
		prefixLedger: () => readJsonl("prefix-ledger.jsonl"),
	};
}

/** Session manager with omp-style leaf/parent lookups and cache-warm `model_usage` entries. */
class WarmingSessionManager extends FakeSessionManager {
	getLeafEntry(): SessionEntry | undefined {
		return this.entries.find((entry) => entry.id === this.leafId);
	}

	getEntry(id: string): SessionEntry | undefined {
		return this.entries.find((entry) => entry.id === id);
	}

	appendWarm(at: number, usage = { cacheRead: 50_000, cacheWrite: 0 }, purpose = "cache-warm"): void {
		const id = `warm-${this.entries.length + 1}`;
		this.entries.push({
			type: "model_usage",
			id,
			parentId: this.leafId,
			timestamp: new Date(at).toISOString(),
			purpose,
			usage,
		} as unknown as SessionEntry);
		this.leafId = id;
	}
}

async function sendSteps(h: Harness, from: number, to: number): Promise<AgentMessage[]> {
	let projected: AgentMessage[] = [];
	for (let steps = from; steps <= to; steps += 1) projected = await h.send(history(steps));
	return projected;
}

describe("observation pack cache-aware batching", () => {
	it("keeps pending observations full below the threshold and swaps them together once crossed", async () => {
		const h = await harness();
		expect(removableTokens(result(1)) * 2).toBeLessThan(5_000);
		expect(removableTokens(result(1)) * 3).toBeGreaterThanOrEqual(5_000);

		const three = await sendSteps(h, 1, 3);
		expect(isPlaceholder(three[resultIndex(1)])).toBe(false);
		const four = await h.send(history(4));
		expect(isPlaceholder(four[resultIndex(1)])).toBe(false);
		expect(isPlaceholder(four[resultIndex(2)])).toBe(false);

		const five = await h.send(history(5));
		expect([1, 2, 3].map((step) => isPlaceholder(five[resultIndex(step)]))).toEqual([true, true, true]);
		expect([4, 5].map((step) => isPlaceholder(five[resultIndex(step)]))).toEqual([false, false]);

		const ledger = await h.ledger();
		const deferred = ledger.filter((row) => row.event === "full" && row.deferred === true);
		expect(deferred.map((row) => [row.request, row.tool])).toEqual([
			[4, "read"],
			[5, "read"],
			[5, "read"],
		]);
		const flushed = ledger.filter((row) => row.event === "placeholder");
		expect(flushed).toHaveLength(3);
		expect(flushed.every((row) => row.flushReason === "threshold" && row.request === 6)).toBe(true);
	});

	it("never flips a swapped observation back to full text", async () => {
		const h = await harness();
		await sendSteps(h, 1, 5);
		for (const steps of [6, 7, 8]) {
			const projected = await h.send(history(steps));
			expect([1, 2, 3].every((step) => isPlaceholder(projected[resultIndex(step)]))).toBe(true);
		}
		// Even when the rest of the prefix changes, the swapped result stays swapped.
		const rewritten = history(8);
		rewritten[1] = assistant(1_000, "rewritten by another extension");
		const projected = await h.send(rewritten);
		expect([1, 2, 3].every((step) => isPlaceholder(projected[resultIndex(step)]))).toBe(true);

		const flushedIds = (await h.ledger()).flatMap((row) => (row.flushReason ? [row.id] : []));
		expect(flushedIds).toHaveLength(6);
		expect(new Set(flushedIds).size).toBe(flushedIds.length);
	});

	it("keeps a swapped observation as a placeholder after a rewind leaves fewer assistants after it", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		expect(isPlaceholder((await sendSteps(h, 1, 3))[resultIndex(1)])).toBe(true);
		const rewound = await h.send([...history(1), assistant(2_000, "retry from checkpoint")]);
		expect(isPlaceholder(rewound[resultIndex(1)])).toBe(true);
	});

	it("ignores context calls without an assistant message for prefix tracking (omp live steering)", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, prefixDiagnostics: true });
		await sendSteps(h, 1, 4); // results 1 and 2 are pending
		await h.send([user("steer: also check the tests")]);
		const projected = await h.send([...history(4), user("steer: also check the tests"), assistant(5_000), result(5)]);
		expect([1, 2].map((step) => isPlaceholder(projected[resultIndex(step)]))).toEqual([false, false]);
		expect((await h.ledger()).filter((row) => row.flushReason !== undefined)).toEqual([]);
		const rows = await h.prefixLedger();
		expect(rows.map((row) => [row.continuation, row.firstChangedIndex, row.flush])).toEqual([
			[true, null, false],
			[true, null, false],
			[true, null, false],
			[true, null, false],
			[false, null, false],
			[true, null, false],
		]);
		expect(rows[5]?.prevMessageCount).toBe(history(4).length);
	});

	it("does not re-log an (event, id, request) row when a request number repeats non-consecutively", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (const steps of [1, 2, 1, 2]) await h.send(history(steps));
		const keys = (await h.ledger()).map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(keys).toHaveLength(3);
		expect(new Set(keys).size).toBe(keys.length);
	});

	it("does not advance counts or duplicate ledger rows on a repeated context call", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		const once = await h.send(history(1));
		const again = await h.send(history(1));
		const thrice = await h.send(history(1));
		expect(textOf(once[resultIndex(1)])).toBe(textOf(result(1)));
		expect(textOf(again[resultIndex(1)])).toBe(textOf(result(1)));
		expect(textOf(thrice[resultIndex(1)])).toBe(textOf(result(1)));
		await h.send(history(2));
		await h.send(history(2));
		const swapped = await h.send(history(3));
		const repeated = await h.send(history(3));
		expect(isPlaceholder(swapped[resultIndex(1)])).toBe(true);
		expect(textOf(repeated[resultIndex(1)])).toBe(textOf(swapped[resultIndex(1)]));

		const rows = await h.ledger();
		const keys = rows.map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(new Set(keys).size).toBe(keys.length);
		expect(rows.filter((row) => row.flushReason !== undefined)).toHaveLength(1);
	});

	it("announces each observation once, when it first becomes a placeholder", async () => {
		const h = await harness();
		const notify = vi.fn();
		const tui = { mode: "tui", hasUI: true, ui: { notify, setStatus: vi.fn() } as never } as Partial<ExtensionContext>;
		for (let steps = 1; steps <= 7; steps += 1) {
			await h.send(history(steps), tui);
			await h.send(history(steps), tui);
		}
		expect(notify).toHaveBeenCalledTimes(3);
	});

	it("flushes pending observations after a cold gap", async () => {
		const h = await harness({ coldGapMs: 60_000, batchThresholdTokens: 100_000 });
		const warm = await sendSteps(h, 1, 3);
		expect(isPlaceholder(warm[resultIndex(1)])).toBe(false);

		h.clock.now = 4_000 + 59_999;
		expect(isPlaceholder((await h.send(history(4)))[resultIndex(1)])).toBe(false);
		h.clock.now = 5_000 + 60_000;
		const cold = await h.send(history(5));
		expect(isPlaceholder(cold[resultIndex(1)])).toBe(true);
		expect(isPlaceholder(cold[resultIndex(2)])).toBe(true);
		expect(isPlaceholder(cold[resultIndex(3)])).toBe(true);
		const reasons = (await h.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []));
		expect(reasons).toEqual(["cold-gap", "cold-gap", "cold-gap"]);
	});

	it("uses the default five-minute cold gap", async () => {
		const h = await harness();
		await sendSteps(h, 1, 2);
		h.clock.now = 3_000 + 300_000;
		expect(isPlaceholder((await h.send(history(3)))[resultIndex(1)])).toBe(true);
	});

	it("measures the cold gap from a later omp cache-warming refresh", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = new WarmingSessionManager([], SESSION_ID, h.sessionDir);
		const ctx = { sessionManager: manager as never };
		for (let steps = 1; steps <= 2; steps += 1) {
			manager.appendMessage(assistant(steps * 1_000));
			await h.send(history(steps), ctx);
		}
		manager.appendMessage(assistant(3_000));
		// Idle for 20 minutes, kept warm by refreshes every 270 s; the last one 60 s ago.
		for (let at = 3_000 + 270_000; at <= 3_000 + 1_140_000; at += 270_000) manager.appendWarm(at);
		manager.appendMessage(user("next prompt"));
		h.clock.now = 3_000 + 1_140_000 + 60_000;
		const warm = await h.pi.emitContext(history(3), h.context(ctx));
		expect(isPlaceholder(warm[resultIndex(1)])).toBe(false);

		// Warming stopped: the gap since the last refresh reaches the default five minutes.
		manager.appendMessage(assistant(4_000));
		manager.appendWarm(h.clock.now + 1_000, { cacheRead: 50_000, cacheWrite: 0 }, "cache-warm:extension-override");
		h.clock.now += 1_000 + 300_000;
		const cold = await h.pi.emitContext(history(4), h.context(ctx));
		expect([1, 2].map((step) => isPlaceholder(cold[resultIndex(step)]))).toEqual([true, true]);
		expect((await h.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []))).toEqual([
			"cold-gap",
			"cold-gap",
		]);
	});

	it("ignores cache-warming entries without cache tokens or before the last assistant", () => {
		const manager = new WarmingSessionManager([], SESSION_ID);
		const ctx = fakeContext(manager as never);
		expect(lastCacheWarmAt(ctx)).toBeUndefined();
		manager.appendWarm(10_000);
		manager.appendMessage(assistant(20_000));
		manager.appendWarm(30_000, { cacheRead: 0, cacheWrite: 0 });
		manager.appendMessage(user("next"));
		expect(lastCacheWarmAt(ctx)).toBeUndefined();
		manager.appendWarm(40_000, { cacheRead: 0, cacheWrite: 9_000 });
		manager.appendWarm(50_000);
		expect(lastCacheWarmAt(ctx)).toBe(50_000);
		expect(requestTiming([assistant(20_000)], undefined, 60_000, lastCacheWarmAt(ctx)).gapMs).toBe(10_000);

		// Hosts without leaf lookups, or with throwing ones, give no warming signal.
		expect(lastCacheWarmAt(fakeContext(new FakeSessionManager()))).toBeUndefined();
		const broken = Object.assign(new WarmingSessionManager([], SESSION_ID), {
			getLeafEntry: () => {
				throw new Error("unavailable");
			},
		});
		expect(lastCacheWarmAt(fakeContext(broken as never))).toBeUndefined();
	});

	it("recognises an omp side-turn suffix only with replayed answers", () => {
		const developer = { role: "developer", content: [{ type: "text", text: "side rules" }], timestamp: 0 } as unknown as AgentMessage;
		const main = history(2);
		const replayed = assistant(9_000, "earlier side answer");
		expect(sideTurnStart([...main, developer, user("q1"), replayed, user("q2")])).toBe(main.length);
		// A first side question adds no assistant, so it needs no special handling.
		expect(sideTurnStart([...main, developer, user("q1")])).toBeUndefined();
		// A real reply (with usage) or a tool call after a developer reminder is the main stream.
		const real = { ...(replayed as object), usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15 } } as AgentMessage;
		expect(sideTurnStart([...main, developer, user("q1"), real, user("q2")])).toBeUndefined();
		const toolCall = { ...(replayed as object), content: [{ type: "toolCall", id: "c", name: "read", arguments: {} }] } as AgentMessage;
		expect(sideTurnStart([...main, developer, user("q1"), toolCall, user("q2")])).toBeUndefined();
		expect(sideTurnStart([...main, developer, user("q1"), replayed, result(9), user("q2")])).toBeUndefined();
		expect(sideTurnStart([...main, developer, user("q1"), replayed])).toBeUndefined();
		expect(sideTurnStart([user("q1"), replayed, user("q2")])).toBeUndefined();
	});

	it("a side turn neither counts its replayed answers as sends nor flushes nor replaces the previous request", async () => {
		const h = await harness({ batchThresholdTokens: 1, prefixDiagnostics: true });
		await sendSteps(h, 1, 3); // r1 swapped (threshold), r2 sent twice, r3 once
		const developer = { role: "developer", content: [{ type: "text", text: "side rules" }], timestamp: 0 } as unknown as AgentMessage;
		const side = [...history(3), developer, user("q1"), assistant(9_000, "a1"), user("q2"), assistant(9_000, "a2"), user("q3")];
		h.clock.now = 10_000_000; // a cold gap would flush a main request
		const sideOut = await h.pi.emitContext(side, h.context());
		expect([1, 2, 3].map((step) => isPlaceholder(sideOut[resultIndex(step)]))).toEqual([true, false, false]);
		const diagnostics = await h.prefixLedger();
		expect(diagnostics.at(-1)).toMatchObject({ continuation: false, flush: false, request: 4, prevMessageCount: null });
		// The next main request swaps r2 (third send) but not r3 (second send), with no prefix change.
		h.clock.now = 0;
		const main = await h.send(history(4));
		expect([1, 2, 3].map((step) => isPlaceholder(main[resultIndex(step)]))).toEqual([true, true, false]);
		expect(diagnostics.length + 1).toBe((await h.prefixLedger()).length);
		expect((await h.prefixLedger()).at(-1)).toMatchObject({ continuation: true, firstChangedIndex: null, flushReason: "threshold" });
	});

	it("flushes pending observations when the model changes", async () => {
		const h = await harness();
		await sendSteps(h, 1, 2);
		const switched = await h.send(history(3), { model: { provider: PROVIDER, id: "model-b" } as never });
		expect(isPlaceholder(switched[resultIndex(1)])).toBe(true);
		expect((await h.ledger()).find((row) => row.flushReason)?.flushReason).toBe("model-change");
	});

	it("skips the model signal without throwing when the host exposes no model", async () => {
		const h = await harness();
		await sendSteps(h, 1, 2);
		const unset = await h.send(history(3), { model: undefined });
		expect(isPlaceholder(unset[resultIndex(1)])).toBe(false);

		const throwing = h.context();
		Object.defineProperty(throwing, "model", {
			get() {
				throw new Error("model unavailable");
			},
		});
		const guarded = await h.pi.emitContext(history(4), throwing);
		expect(isPlaceholder(guarded[resultIndex(1)])).toBe(false);
		expect(isPlaceholder(guarded[resultIndex(2)])).toBe(false);
	});

	it("flushes only pending observations at or after the first changed index", async () => {
		const h = await harness();
		await sendSteps(h, 1, 3);
		const changed = history(4);
		// Rewrite the assistant between result 1 and result 2.
		const changedIndex = resultIndex(1) + 1;
		changed[changedIndex] = assistant(2_000, "rewritten");
		const projected = await h.send(changed);
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(false);
		expect(isPlaceholder(projected[resultIndex(2)])).toBe(true);
		const flushed = (await h.ledger()).filter((row) => row.flushReason);
		expect(flushed.map((row) => row.flushReason)).toEqual(["prefix-changed"]);
	});

	it("treats compaction (removed leading messages) as a prefix change from index zero", async () => {
		const h = await harness();
		await sendSteps(h, 1, 4);
		const compacted = [user("summary of earlier work"), ...history(5).slice(resultIndex(1) + 1)];
		const projected = await h.send(compacted);
		// result 2 and result 3 are pending and both after the change at index 0.
		expect(isPlaceholder(projected[2])).toBe(true);
		expect(isPlaceholder(projected[4])).toBe(true);
		expect(isPlaceholder(projected[6])).toBe(false);
		expect((await h.ledger()).filter((row) => row.flushReason).map((row) => row.flushReason)).toEqual([
			"prefix-changed",
			"prefix-changed",
		]);
	});

	it("flushes eligible pending observations on the first request of a fresh process", async () => {
		const sessionDir = await sessionRoot();
		const first = await harness({}, sessionDir);
		const before = await sendSteps(first, 1, 4);
		expect(isPlaceholder(before[resultIndex(1)])).toBe(false);

		const restarted = await harness({}, sessionDir);
		const resumed = await restarted.send(history(4));
		expect(isPlaceholder(resumed[resultIndex(1)])).toBe(true);
		expect(isPlaceholder(resumed[resultIndex(2)])).toBe(true);
		expect(isPlaceholder(resumed[resultIndex(3)])).toBe(false);
		const reasons = (await restarted.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []));
		expect(reasons).toEqual(["process-start", "process-start"]);
	});

	it("keeps state per session root", async () => {
		const sessionDir = await sessionRoot();
		const h = await harness({}, sessionDir);
		await sendSteps(h, 1, 3);
		const other = fakeContext(new FakeSessionManager([], "session-b", sessionDir), {
			model: { provider: PROVIDER, id: MODEL } as never,
		});
		// A different root has no previous request yet, so its eligible result swaps at once.
		const projected = await h.pi.emitContext(history(3), other);
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(true);
		expect(isPlaceholder((await h.send(history(3)))[resultIndex(1)])).toBe(false);
	});

	it("sends byte-identical placeholders in batched and legacy mode", async () => {
		const batched = await harness({ batchThresholdTokens: 1 });
		const legacy = await harness({ batchThresholdTokens: 0 });
		const fromBatched = await sendSteps(batched, 1, 3);
		const fromLegacy = await sendSteps(legacy, 1, 3);
		const expected = placeholderFor(createObservation(result(1), "/unused")!);
		expect(textOf(fromBatched[resultIndex(1)])).toBe(expected);
		expect(textOf(fromLegacy[resultIndex(1)])).toBe(expected);
		expect(expected.split("\n").slice(0, 3)).toEqual([
			`[large tool result replaced after its first ${FULL_SENDS} provider requests]`,
			expect.stringMatching(/^id: obs_[a-f0-9]{24}$/u),
			"tool: read",
		]);
	});

	it("keeps the legacy per-call immediate swap when the threshold is zero", async () => {
		const h = await harness({ batchThresholdTokens: 0 });
		// Legacy counts every context call as one send, including repeats.
		const calls: boolean[] = [];
		for (let call = 0; call < 4; call += 1) calls.push(isPlaceholder((await h.send(history(1)))[resultIndex(1)]));
		expect(calls).toEqual([false, false, true, true]);

		// In a growing conversation every result swaps on its own third request.
		const g = await harness({ batchThresholdTokens: 0 });
		for (let steps = 1; steps <= 6; steps += 1) {
			const projected = await g.send(history(steps));
			for (let step = 1; step <= steps; step += 1) {
				expect(isPlaceholder(projected[resultIndex(step)])).toBe(steps - step >= FULL_SENDS);
			}
		}
		const reasons = (await g.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []));
		expect(reasons).toEqual(["legacy", "legacy", "legacy", "legacy"]);
		expect((await g.ledger()).some((row) => row.deferred)).toBe(false);
	});

	it("writes prefix diagnostics only when enabled", async () => {
		const off = await harness();
		await sendSteps(off, 1, 5);
		expect(await off.prefixLedger()).toEqual([]);

		const on = await harness({ prefixDiagnostics: true });
		await sendSteps(on, 1, 4);
		await on.send(history(4));
		await on.send(history(5));
		const rows = await on.prefixLedger();
		expect(rows).toHaveLength(6);
		for (const row of rows) {
			expect(Object.keys(row)).toEqual(
				expect.arrayContaining([
					"timestamp",
					"request",
					"messageCount",
					"prevMessageCount",
					"firstChangedIndex",
					"gapMs",
					"modelChanged",
					"repeat",
					"pendingCount",
					"pendingTokens",
					"flush",
					"flushReason",
					"flushedCount",
				]),
			);
		}
		expect(rows[0]).toMatchObject({ request: 2, prevMessageCount: null, firstChangedIndex: null, repeat: false });
		expect(rows[3]).toMatchObject({ request: 5, pendingCount: 2, flush: false, flushReason: null, gapMs: 1_000 });
		expect(rows[4]).toMatchObject({ request: 5, repeat: true, firstChangedIndex: null, flush: false });
		expect(rows[5]).toMatchObject({ request: 6, pendingCount: 3, flush: true, flushReason: "threshold", flushedCount: 3 });

		const legacy = await harness({ batchThresholdTokens: 0, prefixDiagnostics: true });
		await sendSteps(legacy, 1, 3);
		const legacyRows = await legacy.prefixLedger();
		expect(legacyRows.at(-1)).toMatchObject({ request: 4, firstChangedIndex: null, flush: true, flushReason: "legacy", flushedCount: 1 });
	});

	it("fails open to full text when diagnostics cannot be written", async () => {
		const errors: string[] = [];
		vi.spyOn(console, "error").mockImplementation((...values: unknown[]) => {
			errors.push(values.map(String).join(" "));
		});
		const h = await harness({ prefixDiagnostics: true });
		await sendSteps(h, 1, 1);
		const path = join(h.sessionDir, "sol-pi", SESSION_ID, "observation-pack", "prefix-ledger.jsonl");
		await rm(path);
		await mkdir(path);
		const projected = await h.send(history(2));
		expect(textOf(projected[resultIndex(1)])).toBe(textOf(result(1)));
		expect(errors.some((error) => error.includes("prefix diagnostics"))).toBe(true);
	});

	it("wires the sol-pi.json batching keys through the configured entrypoint", async () => {
		const sessionDir = await sessionRoot();
		const pi = new FakePi();
		registerConfiguredFeatures(pi.asExtensionApi(), {
			...DEFAULT_CONFIG,
			observationPack: true,
			observationPackBatchThresholdTokens: 0,
			observationPackPrefixDiagnostics: true,
		});
		let projected: AgentMessage[] = [];
		for (let call = 0; call < 3; call += 1) projected = await pi.emitContext(history(1), fakeContext(sessionDir));
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(true);
		const diagnostics = await readFile(
			join(sessionDir, "sol-pi", SESSION_ID, "observation-pack", "prefix-ledger.jsonl"),
			"utf8",
		);
		expect(diagnostics.trim().split("\n")).toHaveLength(3);
	});

	it("computes the first changed index and request timing", () => {
		expect(firstChangedIndex(["a", "b"], ["a", "b", "c"])).toBeUndefined();
		expect(firstChangedIndex(["a", "b", "c"], ["a", "x", "c"])).toBe(1);
		expect(firstChangedIndex(["a", "b", "c"], ["a", "b"])).toBe(2);
		expect(firstChangedIndex([], ["a"])).toBeUndefined();

		const messages = history(2);
		expect(requestTiming(messages, { provider: PROVIDER, id: MODEL }, 2_500)).toEqual({ gapMs: 500, modelChanged: false });
		expect(requestTiming(messages, { provider: "other", id: MODEL }, 2_500).modelChanged).toBe(true);
		expect(requestTiming(messages, undefined, 2_500).modelChanged).toBe(false);
		expect(requestTiming([user("only")], undefined, 1)).toEqual({ gapMs: undefined, modelChanged: false });
	});
});
