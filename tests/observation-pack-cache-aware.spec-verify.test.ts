/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * Verifier edge-case tests for ObservationPack cache-aware batching (spec B1-B9).
 * The legacy-equivalence block compares against the pre-batching implementation
 * kept verbatim in tests/fixtures/observation-pack-original-0bff376.ts.
 */
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadSolPiConfig } from "../src/sol-pi/config.ts";
import {
	createObservation,
	createObservationPackExtension,
	type ObservationPackOptions,
	placeholderFor,
	THRESHOLD_BYTES,
} from "../src/sol-pi/extensions/observation-pack/index.ts";
import { createObservationPackExtension as createOriginalExtension } from "./fixtures/observation-pack-original-0bff376.ts";
import { FakePi, FakeSessionManager, fakeContext } from "./helpers.ts";

const PROVIDER = "provider-a";
const MODEL = "model-a";
const dirs: string[] = [];

afterEach(async () => {
	await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
	vi.restoreAllMocks();
});

async function tempDir(prefix = "op-spec-verify-"): Promise<string> {
	const dir = await mkdtemp(join(tmpdir(), prefix));
	dirs.push(dir);
	return dir;
}

function user(text: string): AgentMessage {
	return { role: "user", content: text, timestamp: 0 } as AgentMessage;
}

function assistant(timestamp: unknown, text = "calling a tool", provider: unknown = PROVIDER, model: unknown = MODEL): AgentMessage {
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

/** Large pure-text result; `scale` multiplies its size (1 = just over the 10 KiB threshold). */
function result(step: number, scale = 1, callId = `call-${step}`): ToolResultMessage {
	const line = `observation ${step} line\n`;
	const text = `result ${step}\n${line.repeat(Math.ceil(((THRESHOLD_BYTES + 64) * scale) / line.length))}`;
	return {
		role: "toolResult",
		toolCallId: callId,
		toolName: "read",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: step,
	};
}

function removable(message: ToolResultMessage): number {
	const observation = createObservation(message, "/unused")!;
	return observation.tokens - Math.ceil(placeholderFor(observation).length / 4);
}

/** [user, a1, r1, a2, r2, ...]; assistant k stamped k seconds. */
function history(steps: number, make: (step: number) => AgentMessage = (step) => result(step)): AgentMessage[] {
	const messages: AgentMessage[] = [user("start")];
	for (let step = 1; step <= steps; step += 1) messages.push(assistant(step * 1_000), make(step));
	return messages;
}

const at = (step: number): number => step * 2;

function textOf(message: AgentMessage | undefined): string {
	if (!message || message.role !== "toolResult") throw new Error("expected tool result");
	return message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

function isPlaceholder(message: AgentMessage | undefined): boolean {
	return textOf(message).startsWith("[large tool result replaced");
}

type Row = Record<string, unknown>;

async function readJsonl(path: string): Promise<Row[]> {
	const text = await readFile(path, "utf8").catch(() => "");
	return text
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line) as Row);
}

interface Harness {
	readonly pi: FakePi;
	readonly clock: { now: number };
	readonly dir: string;
	readonly notify: ReturnType<typeof vi.fn>;
	/** Send with a warm clock (1 s after the latest assistant) unless `now` is given. */
	send(messages: readonly AgentMessage[], extra?: { now?: number; ctx?: Partial<ExtensionContext>; session?: string }): Promise<AgentMessage[]>;
	ledger(session?: string): Promise<Row[]>;
	prefix(session?: string): Promise<Row[]>;
}

async function harness(options: ObservationPackOptions = {}, dir?: string): Promise<Harness> {
	const sessionDir = dir ?? (await tempDir());
	const clock = { now: 0 };
	const notify = vi.fn();
	const pi = new FakePi();
	createObservationPackExtension({ now: () => clock.now, ...options })(pi.asExtensionApi());
	const opDir = (session: string) => join(sessionDir, "sol-pi", session, "observation-pack");
	return {
		pi,
		clock,
		dir: sessionDir,
		notify,
		async send(messages, extra = {}) {
			if (extra.now !== undefined) clock.now = extra.now;
			else {
				const last = messages.findLast((message) => message.role === "assistant") as { timestamp?: unknown } | undefined;
				const stamp = typeof last?.timestamp === "number" ? last.timestamp : 0;
				clock.now = Math.max(clock.now, stamp + 1_000);
			}
			const ctx = fakeContext(new FakeSessionManager([], extra.session ?? "session-a", sessionDir), {
				model: { provider: PROVIDER, id: MODEL } as never,
				mode: "tui",
				hasUI: true,
				ui: { notify, setStatus: vi.fn() } as never,
				...extra.ctx,
			});
			return pi.emitContext(messages, ctx);
		},
		ledger: (session = "session-a") => readJsonl(join(opDir(session), "ledger.jsonl")),
		prefix: (session = "session-a") => readJsonl(join(opDir(session), "prefix-ledger.jsonl")),
	};
}

function keyOf(row: Row): string {
	return `${String(row.event)}|${String(row.id)}|${String(row.request)}`;
}

function flushRows(rows: Row[]): Row[] {
	return rows.filter((row) => row.flushReason !== undefined);
}

// ---------------------------------------------------------------------------
describe("B1 repeated context call for the same request", () => {
	it("is idempotent in batched mode across a whole conversation (projection, ledger, toast)", async () => {
		const h = await harness({ batchThresholdTokens: 5_000 });
		for (let steps = 1; steps <= 7; steps += 1) {
			const first = await h.send(history(steps));
			const second = await h.send(history(steps));
			const third = await h.send(history(steps));
			expect(JSON.stringify(second)).toBe(JSON.stringify(first));
			expect(JSON.stringify(third)).toBe(JSON.stringify(first));
		}
		const rows = await h.ledger();
		const keys = rows.map(keyOf);
		expect(new Set(keys).size).toBe(keys.length);
		const flushed = flushRows(rows).map((row) => row.id);
		expect(new Set(flushed).size).toBe(flushed.length);
		expect(h.notify).toHaveBeenCalledTimes(flushed.length);
	});

	it("does not advance the send count when the hook fires many times before the next assistant turn", async () => {
		// threshold 1: every eligible observation flushes immediately, so only the count gates the swap.
		const h = await harness({ batchThresholdTokens: 1 });
		for (let call = 0; call < 10; call += 1) expect(isPlaceholder((await h.send(history(1)))[at(1)])).toBe(false);
		for (let call = 0; call < 10; call += 1) expect(isPlaceholder((await h.send(history(2)))[at(1)])).toBe(false);
		expect(isPlaceholder((await h.send(history(3)))[at(1)])).toBe(true);
		const placeholderRows = (await h.ledger()).filter((row) => row.event === "placeholder");
		expect(placeholderRows).toHaveLength(1);
		expect(placeholderRows[0]).toMatchObject({ request: 4, sendNumber: 3 });
	});

	it("repeated call of a flushing request re-sends the same placeholders without a second flushReason", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await h.send(history(1));
		await h.send(history(2));
		const flushing = await h.send(history(3));
		const repeat = await h.send(history(3));
		expect(JSON.stringify(repeat)).toBe(JSON.stringify(flushing));
		expect(flushRows(await h.ledger())).toHaveLength(1);
		expect(h.notify).toHaveBeenCalledTimes(1);
	});
});

// ---------------------------------------------------------------------------
describe("B4a threshold crossing", () => {
	it("uses >= and flips every pending observation together in one request", async () => {
		const r = removable(result(1));
		// Two pending results remove exactly 2r; at T = 2r they flush together, at T = 2r + 1 they wait.
		const exact = await harness({ batchThresholdTokens: 2 * r });
		const above = await harness({ batchThresholdTokens: 2 * r + 1 });
		for (const h of [exact, above]) for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		const exactOut = await exact.send(history(4));
		const aboveOut = await above.send(history(4));
		expect([1, 2, 3, 4].map((step) => isPlaceholder(exactOut[at(step)]))).toEqual([true, true, false, false]);
		expect([1, 2, 3, 4].map((step) => isPlaceholder(aboveOut[at(step)]))).toEqual([false, false, false, false]);
		const flushed = flushRows(await exact.ledger());
		expect(flushed.map((row) => [row.flushReason, row.request])).toEqual([
			["threshold", 5],
			["threshold", 5],
		]);
		const deferred = (await above.ledger()).filter((row) => row.deferred === true);
		expect(deferred.map((row) => row.request)).toEqual([4, 5, 5]);
	});

	it("a single large observation above the threshold flushes alone and immediately", async () => {
		const big = (step: number) => result(step, step === 1 ? 12 : 1);
		const h = await harness({ batchThresholdTokens: 20_000 });
		expect(removable(big(1) as ToolResultMessage)).toBeGreaterThanOrEqual(20_000);
		await h.send(history(1, big));
		await h.send(history(2, big));
		const out = await h.send(history(3, big));
		expect(isPlaceholder(out[at(1)])).toBe(true);
		expect(flushRows(await h.ledger()).map((row) => row.flushReason)).toEqual(["threshold"]);
	});

	it("already-swapped observations do not count toward the pending sum", async () => {
		const r = removable(result(1));
		const h = await harness({ batchThresholdTokens: 2 * r });
		for (let steps = 1; steps <= 4; steps += 1) await h.send(history(steps)); // r1+r2 flush at request 5
		const out5 = await h.send(history(5)); // r3 pending alone (r < 2r)
		expect(isPlaceholder(out5[at(3)])).toBe(false);
		const out6 = await h.send(history(6)); // r3 + r4 pending = 2r
		expect([1, 2, 3, 4].map((step) => isPlaceholder(out6[at(step)]))).toEqual([true, true, true, true]);
	});
});

// ---------------------------------------------------------------------------
describe("B4b cold gap", () => {
	it("measures the gap from the most recent assistant message only", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, coldGapMs: 10_000 });
		await h.send(history(1));
		await h.send(history(2));
		// Latest assistant is at 3000; 9999 ms later is still warm although assistant 1 is 12 s old.
		expect(isPlaceholder((await h.send(history(3), { now: 12_999 }))[at(1)])).toBe(false);
		expect(isPlaceholder((await h.send(history(4), { now: 14_000 }))[at(1)])).toBe(true);
	});

	it("does not flush or throw without a usable assistant timestamp", async () => {
		const odd = (stamp: unknown) => (steps: number) => {
			const messages = history(steps);
			for (let step = 1; step <= steps; step += 1) messages[at(step) - 1] = assistant(stamp);
			return messages;
		};
		for (const stamp of ["2026-09-28T00:00:00Z", undefined, Number.NaN, null]) {
			const g = await harness({ batchThresholdTokens: 1_000_000, coldGapMs: 10 });
			await g.send(odd(stamp)(1), { now: 10_000_000 });
			await g.send(odd(stamp)(2), { now: 10_000_000 });
			const out = await g.send(odd(stamp)(3), { now: 10_000_000 });
			expect(isPlaceholder(out[at(1)])).toBe(false);
		}
	});

	it("never swaps observations that are still within their full sends", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, coldGapMs: 10 });
		const out = await h.send(history(2), { now: 10_000_000 });
		expect(isPlaceholder(out[at(1)])).toBe(false);
		expect(isPlaceholder(out[at(2)])).toBe(false);
	});
});

// ---------------------------------------------------------------------------
describe("B4c model change", () => {
	it("flushes on a provider-only change and stops flushing once the new model has answered", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		await h.send(history(1));
		await h.send(history(2));
		const other = { model: { provider: "provider-b", id: MODEL } as never };
		const switched = await h.send(history(3), { ctx: other });
		expect(isPlaceholder(switched[at(1)])).toBe(true);
		// Next turn: the latest assistant came from provider-b; r2 is now pending and must wait.
		const next = history(4);
		next[at(4) - 1] = assistant(4_000, "calling a tool", "provider-b", MODEL);
		const after = await h.send(next, { ctx: other });
		expect(isPlaceholder(after[at(2)])).toBe(false);
		expect(flushRows(await h.ledger()).map((row) => row.flushReason)).toEqual(["model-change"]);
	});

	it("skips the signal without throwing for partial host models or assistants without provider/model", async () => {
		const models: unknown[] = [{ provider: PROVIDER }, { id: MODEL }, { provider: 1, id: 2 }, null, "model-a"];
		for (const model of models) {
			const h = await harness({ batchThresholdTokens: 1_000_000 });
			await h.send(history(1), { ctx: { model: model as never } });
			await h.send(history(2), { ctx: { model: model as never } });
			const out = await h.send(history(3), { ctx: { model: model as never } });
			expect(isPlaceholder(out[at(1)])).toBe(false);
		}
		const bare = (steps: number) => {
			const messages = history(steps);
			for (let step = 1; step <= steps; step += 1) {
				const bareAssistant = assistant(step * 1_000, "x") as unknown as Record<string, unknown>;
				delete bareAssistant.provider;
				delete bareAssistant.model;
				messages[at(step) - 1] = bareAssistant as unknown as AgentMessage;
			}
			return messages;
		};
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		await h.send(bare(1));
		await h.send(bare(2));
		const out = await h.send(bare(3), { ctx: { model: { provider: "zzz", id: "zzz" } as never } });
		expect(isPlaceholder(out[at(1)])).toBe(false);
	});

	it("does not throw when ctx.model is a throwing getter on the very first request", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		const ctx = fakeContext(new FakeSessionManager([], "session-a", h.dir));
		Object.defineProperty(ctx, "model", {
			get() {
				throw new Error("no model");
			},
		});
		await expect(h.pi.emitContext(history(3), ctx)).resolves.toHaveLength(7);
	});
});

// ---------------------------------------------------------------------------
describe("B4d prefix changed", () => {
	it("flushes a pending observation located exactly at the first changed index", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		const changed = history(4);
		// Change the assistant message right before r2.
		changed[at(2) - 1] = assistant(2_000, "rewritten assistant 2");
		const out = await h.send(changed);
		// k = at(2) - 1 < at(2): r2 flushed, r1 (index < k) deferred.
		expect(isPlaceholder(out[at(1)])).toBe(false);
		expect(isPlaceholder(out[at(2)])).toBe(true);
		const rows = await h.ledger();
		expect(rows.filter((row) => row.request === 5 && row.deferred === true)).toHaveLength(1);
	});

	it("flushes nothing when the change lies after every pending observation", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, prefixDiagnostics: true });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		// Replace the last message of the previous request (r3) with a small result.
		const changed = history(4);
		changed[at(3)] = { ...result(3), content: [{ type: "text", text: "small" }] } as AgentMessage;
		const out = await h.send(changed);
		expect(isPlaceholder(out[at(1)])).toBe(false);
		expect(isPlaceholder(out[at(2)])).toBe(false);
		const last = (await h.prefix()).at(-1)!;
		expect(last).toMatchObject({ firstChangedIndex: at(3), flush: false, flushReason: null, flushedCount: 0 });
	});

	it("does not see its own already-applied placeholders as a prefix change", async () => {
		const h = await harness({ batchThresholdTokens: 1, prefixDiagnostics: true });
		for (let steps = 1; steps <= 6; steps += 1) await h.send(history(steps));
		const rows = await h.prefix();
		// First row has no previous; every later append-only request must report no change.
		expect(rows.slice(1).map((row) => row.firstChangedIndex)).toEqual([null, null, null, null, null]);
	});

	it("treats truncation at the end as a change at the new length (no pending flushed)", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, prefixDiagnostics: true });
		for (let steps = 1; steps <= 4; steps += 1) await h.send(history(steps));
		const truncated = history(4).slice(0, -1);
		const out = await h.send(truncated);
		expect(isPlaceholder(out[at(1)])).toBe(false);
		expect((await h.prefix()).at(-1)).toMatchObject({ firstChangedIndex: truncated.length, flushedCount: 0 });
	});
});

// ---------------------------------------------------------------------------
describe("message removal / compaction", () => {
	it("a message removed from the middle flushes only pending observations at or after it", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, prefixDiagnostics: true });
		for (let steps = 1; steps <= 4; steps += 1) await h.send(history(steps));
		// Native pruning drops assistant 2 (index 3). r1 (index 2) stays before k.
		const pruned = history(5);
		pruned.splice(at(2) - 1, 1);
		const out = await h.send(pruned);
		expect(isPlaceholder(out[at(1)])).toBe(false); // index 2 < k = 3
		expect(isPlaceholder(out[3])).toBe(true); // r2 moved to index 3
		expect(isPlaceholder(out[5])).toBe(true); // r3
		expect(isPlaceholder(out[7])).toBe(false); // r4, not yet eligible
		expect((await h.prefix()).at(-1)).toMatchObject({ firstChangedIndex: 3, flushReason: "prefix-changed", flushedCount: 2 });
	});

	it("after compaction removes every pending observation nothing is flushed and nothing throws", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 4; steps += 1) await h.send(history(steps));
		const compacted = [user("summary"), assistant(9_000), user("continue")];
		await expect(h.send(compacted)).resolves.toHaveLength(3);
		expect(flushRows(await h.ledger())).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
describe("B2 monotonic placeholder", () => {
	it("keeps a swapped observation as placeholder through later cold/warm/model/prefix variations", async () => {
		const h = await harness({ batchThresholdTokens: 1, coldGapMs: 60_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		const variants: Array<[AgentMessage[], Parameters<Harness["send"]>[1]]> = [
			[history(4), {}],
			[history(5), { now: 1_000_000 }],
			[history(6), { ctx: { model: { provider: "p", id: "m" } as never } }],
			[[user("rewritten"), ...history(7).slice(1)], {}],
			[history(7), { ctx: { model: undefined } }],
		];
		for (const [messages, extra] of variants) {
			const out = await h.send(messages, extra);
			expect(isPlaceholder(out[at(1)])).toBe(true);
		}
	});

	it("keeps sending the placeholder when a later request of the same root has fewer assistants after it (branch/tree navigation)", async () => {
		// Legacy (sentCounts) keeps the placeholder here; batched derives the count from history only.
		for (const batchThresholdTokens of [0, 1]) {
			const h = await harness({ batchThresholdTokens });
			for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
			const swapped = await h.send(history(3));
			expect(isPlaceholder(swapped[at(1)])).toBe(true);
			// User navigates back to an earlier point of the same session and continues from there.
			const branch = [...history(2), user("try another approach")];
			const out = await h.send(branch);
			expect({ batchThresholdTokens, placeholder: isPlaceholder(out[at(1)]) }).toEqual({
				batchThresholdTokens,
				placeholder: true,
			});
		}
	});
});

// ---------------------------------------------------------------------------
describe("B4e fresh instance (restart)", () => {
	it("flushes eligible pending once; later requests follow the batching rules", async () => {
		const dir = await tempDir();
		const first = await harness({ batchThresholdTokens: 1_000_000 }, dir);
		for (let steps = 1; steps <= 4; steps += 1) await first.send(history(steps));
		const restarted = await harness({ batchThresholdTokens: 1_000_000 }, dir);
		const resumed = await restarted.send(history(4));
		const repeat = await restarted.send(history(4));
		expect(JSON.stringify(repeat)).toBe(JSON.stringify(resumed));
		expect([1, 2, 3].map((step) => isPlaceholder(resumed[at(step)]))).toEqual([true, true, false]);
		const next = await restarted.send(history(5));
		expect(isPlaceholder(next[at(3)])).toBe(false); // pending, warm, below threshold
		const reasons = flushRows(await restarted.ledger()).map((row) => row.flushReason);
		expect(reasons).toEqual(["process-start", "process-start"]);
	});

	it("does nothing special on a restart with no eligible observation", async () => {
		const dir = await tempDir();
		const restarted = await harness({ batchThresholdTokens: 1_000_000 }, dir);
		const out = await restarted.send(history(2));
		expect(out.filter((message) => message.role === "toolResult").every((message) => !isPlaceholder(message))).toBe(true);
		expect(flushRows(await restarted.ledger())).toEqual([]);
		// The next request is no longer process-start: the now-eligible r1 waits.
		expect(isPlaceholder((await restarted.send(history(3)))[at(1)])).toBe(false);
	});

	it("each session root has its own process-start", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		const sub = await h.send(history(3), { session: "subagent-1" });
		expect(isPlaceholder(sub[at(1)])).toBe(true);
		expect(flushRows(await h.ledger("subagent-1")).map((row) => row.flushReason)).toEqual(["process-start"]);
		expect(flushRows(await h.ledger())).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
describe("T = 0 legacy equivalence with the original implementation (0bff376)", () => {
	function variant(step: number): AgentMessage {
		switch (step % 7) {
			case 1:
				return { ...result(step), isError: true } as AgentMessage;
			case 2:
				return {
					...result(step),
					content: [
						{ type: "text", text: "héllo ✓ 漢字\r\n".repeat(900) },
						{ type: "text", text: "second block\n".repeat(20) },
					],
				} as AgentMessage;
			case 3:
				return { ...result(step), content: [{ type: "text", text: "x".repeat(THRESHOLD_BYTES * 2) }] } as AgentMessage;
			case 4:
				return {
					...result(step),
					content: [
						{ type: "text", text: "a".repeat(THRESHOLD_BYTES + 10) },
						{ type: "image", data: "AAAA", mimeType: "image/png" },
					],
				} as AgentMessage;
			case 5:
				return { ...result(step), content: [{ type: "text", text: "small result" }] } as AgentMessage;
			case 6:
				return {
					...result(step),
					content: [{ type: "text", text: `sol_pi_evidence_receipt_v1\n${"y\n".repeat(THRESHOLD_BYTES)}` }],
				} as AgentMessage;
			default:
				return result(step, 1 + (step % 3));
		}
	}

	type Call = { session: string; messages: AgentMessage[] };

	function sequence(): Call[] {
		const calls: Call[] = [];
		for (let steps = 1; steps <= 12; steps += 1) {
			const messages = history(steps, variant);
			calls.push({ session: "root-a", messages });
			if (steps % 3 === 0) calls.push({ session: "root-a", messages }); // double-fired hook
			if (steps === 7) calls.push({ session: "root-a", messages }, { session: "root-a", messages });
			if (steps % 4 === 0) calls.push({ session: "root-b", messages: history(Math.ceil(steps / 2), variant) });
		}
		// Compaction: leading messages dropped, same results kept.
		const compacted = [user("summary"), ...history(12, variant).slice(9)];
		calls.push({ session: "root-a", messages: compacted }, { session: "root-a", messages: compacted });
		calls.push({ session: "root-a", messages: [...compacted, assistant(13_000), variant(13)] });
		// Same content under a different tool call id is a different observation.
		calls.push({ session: "root-a", messages: [...compacted, assistant(13_000), result(1, 1, "dup-call")] });
		return calls;
	}

	async function drive(factory: (pi: FakePi) => void, calls: Call[]) {
		const dir = await tempDir("op-legacy-eq-");
		const pi = new FakePi();
		factory(pi);
		const notify = vi.fn();
		const outputs: string[] = [];
		for (const call of calls) {
			const ctx = fakeContext(new FakeSessionManager([], call.session, dir), {
				mode: "tui",
				hasUI: true,
				ui: { notify, setStatus: vi.fn() } as never,
				model: { provider: PROVIDER, id: MODEL } as never,
			});
			outputs.push(JSON.stringify(await pi.emitContext(call.messages, ctx)));
		}
		const ledgers: Record<string, Row[]> = {};
		for (const session of ["root-a", "root-b"]) {
			ledgers[session] = await readJsonl(join(dir, "sol-pi", session, "observation-pack", "ledger.jsonl"));
		}
		return { outputs, ledgers, toasts: notify.mock.calls.map((args) => String(args[0])) };
	}

	it("produces byte-identical projections and toasts on the same input sequence", async () => {
		const calls = sequence();
		const original = await drive((pi) => createOriginalExtension()(pi.asExtensionApi()), calls);
		const legacy = await drive((pi) => createObservationPackExtension({ batchThresholdTokens: 0 })(pi.asExtensionApi()), calls);
		expect(legacy.outputs).toHaveLength(original.outputs.length);
		for (let index = 0; index < calls.length; index += 1) {
			expect({ call: index, output: legacy.outputs[index] }).toEqual({ call: index, output: original.outputs[index] });
		}
		expect(legacy.toasts).toEqual(original.toasts);
		// The sequence must actually exercise swaps.
		expect(original.outputs.some((output) => output.includes("[large tool result replaced"))).toBe(true);
	});

	it("keeps the original ledger rows, only de-duplicating repeats and adding flushReason legacy", async () => {
		const calls = sequence();
		const original = await drive((pi) => createOriginalExtension()(pi.asExtensionApi()), calls);
		const legacy = await drive((pi) => createObservationPackExtension({ batchThresholdTokens: 0 })(pi.asExtensionApi()), calls);
		const strip = (row: Row) => {
			const { timestamp: _t, flushReason: _f, ...rest } = row;
			return rest;
		};
		// B6: no duplicate (event, id, request) row within a process, even when a request number repeats later.
		const dedupe = (rows: Row[]) => {
			const seen = new Set<string>();
			return rows.filter((row) => {
				const key = keyOf(row);
				if (seen.has(key)) return false;
				seen.add(key);
				return true;
			});
		};
		for (const session of ["root-a", "root-b"]) {
			const originalRows = original.ledgers[session]!;
			const legacyRows = legacy.ledgers[session]!;
			expect(originalRows.length).toBeGreaterThan(legacyRows.length - 1); // sanity
			expect(legacyRows.map(strip)).toEqual(dedupe(originalRows).map(strip));
			const legacyFlushes = legacyRows.filter((row) => row.flushReason !== undefined);
			expect(legacyFlushes.every((row) => row.flushReason === "legacy" && row.event === "placeholder")).toBe(true);
			const ids = legacyFlushes.map((row) => row.id);
			expect(new Set(ids).size).toBe(ids.length);
			expect(new Set(ids)).toEqual(new Set(legacyRows.filter((row) => row.event === "placeholder").map((row) => row.id)));
			expect(legacyRows.some((row) => row.deferred !== undefined)).toBe(false);
		}
	});

	it("legacy with diagnostics on projects exactly like legacy with diagnostics off", async () => {
		const calls = sequence();
		const off = await drive((pi) => createObservationPackExtension({ batchThresholdTokens: 0 })(pi.asExtensionApi()), calls);
		const on = await drive(
			(pi) => createObservationPackExtension({ batchThresholdTokens: 0, prefixDiagnostics: true })(pi.asExtensionApi()),
			calls,
		);
		expect(on.outputs).toEqual(off.outputs);
	});
});

// ---------------------------------------------------------------------------
describe("B5 placeholder bytes", () => {
	it("batched placeholder messages equal the original implementation's placeholder messages", async () => {
		const shapes: Array<(step: number) => AgentMessage> = [
			(step) => result(step),
			(step) => ({ ...result(step), content: [{ type: "text", text: "héllo ✓ 漢字\r\n".repeat(900) }] }) as AgentMessage,
			(step) => ({ ...result(step), content: [{ type: "text", text: "z".repeat(THRESHOLD_BYTES * 3) }] }) as AgentMessage,
			(step) =>
				({
					...result(step),
					content: [
						{ type: "text", text: "block one\n".repeat(700) },
						{ type: "text", text: "block two\n".repeat(700) },
					],
				}) as AgentMessage,
		];
		for (const shape of shapes) {
			const originalDir = await tempDir();
			const original = new FakePi();
			createOriginalExtension()(original.asExtensionApi());
			const batched = await harness({ batchThresholdTokens: 1 });
			let fromOriginal: AgentMessage[] = [];
			let fromBatched: AgentMessage[] = [];
			for (let steps = 1; steps <= 3; steps += 1) {
				fromOriginal = await original.emitContext(history(steps, shape), fakeContext(originalDir));
				fromBatched = await batched.send(history(steps, shape));
			}
			expect(isPlaceholder(fromBatched[at(1)])).toBe(true);
			expect(JSON.stringify(fromBatched[at(1)])).toBe(JSON.stringify(fromOriginal[at(1)]));
		}
	});
});

// ---------------------------------------------------------------------------
describe("B6 ledger fields", () => {
	it("placeholder rows only when sent, first carries flushReason, deferred only for pending", async () => {
		const h = await harness({ batchThresholdTokens: 5_000 });
		const sent: Array<{ request: number; placeholders: Set<string> }> = [];
		for (let steps = 1; steps <= 8; steps += 1) {
			const out = await h.send(history(steps));
			const ids = new Set<string>();
			for (let step = 1; step <= steps; step += 1) {
				if (isPlaceholder(out[at(step)])) ids.add(/^id: (obs_\w+)$/mu.exec(textOf(out[at(step)]))![1]!);
			}
			sent.push({ request: steps + 1, placeholders: ids });
		}
		const rows = await h.ledger();
		for (const { request, placeholders } of sent) {
			const logged = new Set(rows.filter((row) => row.request === request && row.event === "placeholder").map((row) => String(row.id)));
			expect({ request, logged: [...logged].sort() }).toEqual({ request, logged: [...placeholders].sort() });
		}
		const firstById = new Map<string, Row>();
		for (const row of rows) if (row.event === "placeholder" && !firstById.has(String(row.id))) firstById.set(String(row.id), row);
		for (const row of rows.filter((candidate) => candidate.event === "placeholder")) {
			expect(row.flushReason !== undefined).toBe(firstById.get(String(row.id)) === row);
		}
		const stepById = new Map<string, number>();
		for (let step = 1; step <= 8; step += 1) stepById.set(createObservation(result(step), "/unused")!.id, step);
		for (const row of rows.filter((candidate) => candidate.event === "full")) {
			const sendNumber = Number(row.request) - stepById.get(String(row.id))!;
			expect({ id: row.id, request: row.request, deferred: row.deferred === true }).toEqual({
				id: row.id,
				request: row.request,
				deferred: sendNumber > 2,
			});
		}
		const validReasons = ["threshold", "cold-gap", "model-change", "prefix-changed", "process-start", "legacy"];
		expect(flushRows(rows).every((row) => validReasons.includes(String(row.flushReason)))).toBe(true);
	});

	it("keeps every original ledger field on full and placeholder rows", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		const rows = await h.ledger();
		const full = rows.find((row) => row.event === "full")!;
		const placeholder = rows.find((row) => row.event === "placeholder")!;
		expect(Object.keys(full)).toEqual(
			expect.arrayContaining(["timestamp", "event", "id", "request", "tool", "originalBytes", "originalLines", "originalTokens", "contentHash"]),
		);
		expect(Object.keys(placeholder)).toEqual(
			expect.arrayContaining([
				"timestamp",
				"event",
				"id",
				"request",
				"sendNumber",
				"tool",
				"originalBytes",
				"originalLines",
				"originalTokens",
				"placeholderBytes",
				"placeholderTokens",
				"removedTokens",
			]),
		);
	});

	it("fails open to full text when the ledger cannot be written, then flushes once it can", async () => {
		vi.spyOn(console, "error").mockImplementation(() => undefined);
		const h = await harness({ batchThresholdTokens: 1 });
		await h.send(history(1));
		const ledgerPath = join(h.dir, "sol-pi", "session-a", "observation-pack", "ledger.jsonl");
		await rm(ledgerPath);
		await mkdir(ledgerPath);
		await h.send(history(2));
		const blocked = await h.send(history(3));
		expect(textOf(blocked[at(1)])).toBe(textOf(result(1)));
		await rm(ledgerPath, { recursive: true });
		const recovered = await h.send(history(4));
		expect(isPlaceholder(recovered[at(1)])).toBe(true);
	});

	it("fails open when a message cannot be fingerprinted (no throw, observations kept in full)", async () => {
		vi.spyOn(console, "error").mockImplementation(() => undefined);
		const h = await harness({ batchThresholdTokens: 1 });
		const weird = (steps: number) => {
			const messages = history(steps);
			(messages[0] as unknown as Record<string, unknown>).details = { size: 10n };
			return messages;
		};
		for (let steps = 1; steps <= 3; steps += 1) {
			const out = await h.send(weird(steps));
			expect(out).toHaveLength(steps * 2 + 1);
			expect(textOf(out[at(1)])).toBe(textOf(result(1)));
		}
	});
});

// ---------------------------------------------------------------------------
describe("B7 diagnostics", () => {
	it("off by default: no prefix ledger in batched or legacy mode", async () => {
		for (const options of [{}, { batchThresholdTokens: 0 }, { batchThresholdTokens: 1 }]) {
			const h = await harness(options);
			for (let steps = 1; steps <= 5; steps += 1) await h.send(history(steps));
			expect(await h.prefix()).toEqual([]);
		}
	});

	it("on: one row per context call with consistent values; projection unchanged", async () => {
		const off = await harness({ batchThresholdTokens: 5_000, coldGapMs: 60_000 });
		const on = await harness({ batchThresholdTokens: 5_000, coldGapMs: 60_000, prefixDiagnostics: true });
		const plan: Array<[AgentMessage[], Parameters<Harness["send"]>[1]]> = [
			[history(1), {}],
			[history(1), {}],
			[history(2), {}],
			[history(3), {}],
			[history(4), { now: 4_000 + 60_000 }],
			[history(5), { ctx: { model: { provider: "p2", id: MODEL } as never } }],
			[[user("compacted"), ...history(6).slice(5)], {}],
		];
		for (const [messages, extra] of plan) {
			const a = await off.send(messages, extra);
			const b = await on.send(messages, extra);
			expect(JSON.stringify(b)).toBe(JSON.stringify(a));
		}
		const rows = await on.prefix();
		expect(rows).toHaveLength(plan.length);
		expect(rows.every((row) => typeof row.timestamp === "string")).toBe(true);
		expect(rows[1]).toMatchObject({ repeat: true, firstChangedIndex: null, messageCount: 3, prevMessageCount: 3 });
		expect(rows[4]).toMatchObject({ gapMs: 60_000, flushReason: "cold-gap", flush: true, flushedCount: 2 });
		expect(rows[5]).toMatchObject({ modelChanged: true, flushReason: "model-change", flush: true, flushedCount: 1 });
		expect(rows[6]).toMatchObject({ firstChangedIndex: 0 });
		for (const row of rows) {
			expect(row.flush).toBe(Number(row.flushedCount) > 0);
			if (!row.flush) expect(row.flushReason).toBeNull();
		}
		const pendingRow = rows[3]!;
		expect(pendingRow).toMatchObject({ pendingCount: 1, pendingTokens: removable(result(1)), flush: false });
	});

	it("reports gapMs null when the context has no assistant message", async () => {
		const h = await harness({ prefixDiagnostics: true });
		await h.send([user("hello")]);
		expect((await h.prefix())[0]).toMatchObject({ gapMs: null, modelChanged: false, prevMessageCount: null });
	});
});

// ---------------------------------------------------------------------------
describe("B8 config validation parity (config.ts vs preflight script)", () => {
	const SCRIPT = join(process.cwd(), "scripts/check-sol-pi-config.mjs");

	async function both(extra: Record<string, unknown>) {
		const root = await tempDir("op-config-parity-");
		const agentDir = join(root, "agent");
		await mkdir(agentDir, { recursive: true });
		const path = join(agentDir, "sol-pi.json");
		await writeFile(path, JSON.stringify({ version: 1, observationPack: true, ...extra }));
		let loaded: Record<string, unknown> | undefined;
		let loadError: string | undefined;
		try {
			loaded = loadSolPiConfig(join(root, "project"), agentDir, false) as unknown as Record<string, unknown>;
		} catch (error) {
			loadError = String(error);
		}
		const run = spawnSync(process.execPath, [SCRIPT, "--config", path], { encoding: "utf8" });
		const preflight = run.status === 0 ? (JSON.parse(run.stdout).effective_config as Record<string, unknown>) : undefined;
		return { loaded, loadError, preflight, stderr: run.stderr };
	}

	const cases: Array<[string, Record<string, unknown>, boolean]> = [
		["missing keys", {}, true],
		["threshold 0", { observationPackBatchThresholdTokens: 0 }, true],
		["threshold 1", { observationPackBatchThresholdTokens: 1 }, true],
		["threshold max safe", { observationPackBatchThresholdTokens: Number.MAX_SAFE_INTEGER }, true],
		["threshold -1", { observationPackBatchThresholdTokens: -1 }, false],
		["threshold 1.5", { observationPackBatchThresholdTokens: 1.5 }, false],
		["threshold string", { observationPackBatchThresholdTokens: "20000" }, false],
		["threshold null", { observationPackBatchThresholdTokens: null }, false],
		["threshold bool", { observationPackBatchThresholdTokens: true }, false],
		["threshold unsafe", { observationPackBatchThresholdTokens: 2 ** 53 }, false],
		["cold gap 1", { observationPackColdGapMs: 1 }, true],
		["cold gap 0", { observationPackColdGapMs: 0 }, false],
		["cold gap -5", { observationPackColdGapMs: -5 }, false],
		["cold gap 0.5", { observationPackColdGapMs: 0.5 }, false],
		["cold gap null", { observationPackColdGapMs: null }, false],
		["cold gap string", { observationPackColdGapMs: "300000" }, false],
		["diagnostics true", { observationPackPrefixDiagnostics: true }, true],
		["diagnostics false", { observationPackPrefixDiagnostics: false }, true],
		["diagnostics string", { observationPackPrefixDiagnostics: "true" }, false],
		["diagnostics 1", { observationPackPrefixDiagnostics: 1 }, false],
		["diagnostics null", { observationPackPrefixDiagnostics: null }, false],
		["misspelled key", { observationPackColdGapMS: 1 }, false],
	];

	for (const [name, extra, valid] of cases) {
		it(`${name}: both ${valid ? "accept" : "reject"} with the same effective values`, async () => {
			const { loaded, loadError, preflight } = await both(extra);
			expect({ loader: loadError === undefined, preflight: preflight !== undefined }).toEqual({ loader: valid, preflight: valid });
			if (valid) {
				for (const key of ["observationPackBatchThresholdTokens", "observationPackColdGapMs", "observationPackPrefixDiagnostics"]) {
					expect({ key, value: loaded![key] }).toEqual({ key, value: preflight![key] });
				}
			}
		});
	}

	it("defaults: 20000 / 300000 / false in both", async () => {
		const { loaded, preflight } = await both({});
		for (const source of [loaded!, preflight!]) {
			expect(source).toMatchObject({
				observationPackBatchThresholdTokens: 20_000,
				observationPackColdGapMs: 300_000,
				observationPackPrefixDiagnostics: false,
			});
		}
	});

	it("sol-pi.example.json is accepted by both validators", async () => {
		const example = JSON.parse(await readFile(join(process.cwd(), "sol-pi.example.json"), "utf8")) as Record<string, unknown>;
		const { loadError, preflight, stderr } = await both(example);
		expect(loadError).toBeUndefined();
		expect(preflight, stderr).toBeDefined();
	});
});

// ---------------------------------------------------------------------------
describe("non-request context calls on the same root (omp live steering / side requests)", () => {
	it("a live-steering context call carrying only the steering message does not force a flush on the next warm request", async () => {
		// omp (providers.openaiLiveSteering, default true) runs transformContext -> context hook on ONLY the
		// messages typed while a GPT-6 response streams (PJi.toProvider -> e.transformContext(i)).
		const h = await harness({ batchThresholdTokens: 1_000_000, prefixDiagnostics: true });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([user("also check the tests please")]);
		const next = [...history(4), user("also check the tests please")];
		const out = await h.send(next);
		expect([1, 2].map((step) => isPlaceholder(out[at(step)]))).toEqual([false, false]);
		expect(flushRows(await h.ledger())).toEqual([]);
	});

	it("a side request that appends a trailing message does not force a flush on the next warm request", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([...history(3), user("side question")]);
		const out = await h.send(history(4));
		expect([1, 2].map((step) => isPlaceholder(out[at(step)]))).toEqual([false, false]);
	});
});

describe("B6 duplicate rows across non-consecutive calls", () => {
	it("does not duplicate (event, id, request) rows when another context call of the same root interleaves", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		await h.send(history(1));
		await h.send(history(2));
		await h.send([user("steering message")]); // interleaved non-request call (request number 1)
		await h.send(history(2)); // hook fires again for request 3 (e.g. retry)
		const keys = (await h.ledger()).map(keyOf);
		expect(keys.filter((key, index) => keys.indexOf(key) !== index)).toEqual([]);
	});
});

describe("B6 duplicate rows when an interleaved call carries a large result under another request number", () => {
	it("does not re-log (event, id, request) rows already written for this root in this process", async () => {
		// Spec B6: no duplicate row for the same (root, observation id, request, event) within a process.
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		await h.send(history(1)); // request 2
		await h.send(history(2)); // request 3: r1 full, r2 full
		await h.send(history(1)); // interleaved call for request 2 (e.g. tree navigation / side request)
		await h.send(history(2)); // request 3 again
		const keys = (await h.ledger()).map(keyOf);
		expect(keys.filter((key, index) => keys.indexOf(key) !== index)).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
// Round 2 (fix 890ecc3): edge cases around the D1 / D2 / D3 / R5 fixes.
describe("round 2: rewind keeps placeholders without side effects (D1 fix)", () => {
	it("a rewind re-sends the placeholder without a second flushReason, toast or spurious prefix flush", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 2; steps += 1) await h.send(history(steps));
		// Request 4 under another model: r1 (eligible) flushes with model-change.
		const flushed = await h.send(history(3), { ctx: { model: { provider: "p2", id: "m2" } as never } });
		expect(isPlaceholder(flushed[at(1)])).toBe(true);
		await h.send(history(4)); // r2 now pending, warm, below threshold
		const toastsBefore = h.notify.mock.calls.length;
		// Rewind to a point with one assistant after r1, then continue on the branch.
		const branch = [...history(2), user("retry differently")];
		const rewound = await h.send(branch);
		expect([isPlaceholder(rewound[at(1)]), isPlaceholder(rewound[at(2)])]).toEqual([true, false]);
		const continued = await h.send([...branch, assistant(9_000), result(9)]);
		expect([isPlaceholder(continued[at(1)]), isPlaceholder(continued[at(2)])]).toEqual([true, false]);
		expect(h.notify.mock.calls.length).toBe(toastsBefore);
		const rows = await h.ledger();
		expect(flushRows(rows).map((row) => [row.id, row.flushReason])).toEqual([[flushRows(rows)[0]!.id, "model-change"]]);
		const keys = rows.map(keyOf);
		expect(keys.filter((key, index) => keys.indexOf(key) !== index)).toEqual([]);
	});

	it("a rewind to before the swapped observation and back re-applies the placeholder", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([user("start"), user("fresh start")]);
		const back = await h.send(history(1));
		expect(isPlaceholder(back[at(1)])).toBe(true);
	});
});

describe("round 2: non-request context calls (D2 fix)", () => {
	it("two consecutive steering calls followed by a repeat of the previous request do not flush", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([user("steer 1")]);
		await h.send([user("steer 1"), user("steer 2")]);
		const repeat = await h.send(history(3));
		const next = await h.send([...history(4), user("steer 1"), user("steer 2")]);
		for (const out of [repeat, next]) expect([1, 2].map((step) => isPlaceholder(out[at(step)]))).toEqual([false, false]);
		expect(flushRows(await h.ledger())).toEqual([]);
	});

	it("a genuine prefix change across a steering call is still detected (prefix-changed from index 0)", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([user("steer")]);
		const rewritten = [user("rewritten by another extension"), ...history(4).slice(1)];
		const out = await h.send(rewritten);
		expect([1, 2, 3].map((step) => isPlaceholder(out[at(step)]))).toEqual([true, true, false]);
		expect(flushRows(await h.ledger()).map((row) => row.flushReason)).toEqual(["prefix-changed", "prefix-changed"]);
	});

	it("a cold gap after a steering call still flushes with cold-gap", async () => {
		const h = await harness({ batchThresholdTokens: 1_000_000, coldGapMs: 60_000 });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		await h.send([user("steer")], { now: 10_000 });
		const out = await h.send(history(4), { now: 4_000 + 60_000 });
		expect([1, 2].map((step) => isPlaceholder(out[at(step)]))).toEqual([true, true]);
		expect(flushRows(await h.ledger()).map((row) => row.flushReason)).toEqual(["cold-gap", "cold-gap"]);
	});

	it("a steering call as the very first call of a fresh process keeps process-start for the resumed request", async () => {
		const dir = await tempDir();
		const first = await harness({ batchThresholdTokens: 1_000_000 }, dir);
		for (let steps = 1; steps <= 3; steps += 1) await first.send(history(steps));
		const restarted = await harness({ batchThresholdTokens: 1_000_000 }, dir);
		await restarted.send([user("steer")]);
		const out = await restarted.send(history(4));
		expect([1, 2, 3].map((step) => isPlaceholder(out[at(step)]))).toEqual([true, true, false]);
		expect(flushRows(await restarted.ledger()).map((row) => row.flushReason)).toEqual(["process-start", "process-start"]);
	});

	it("a steering call re-sends existing placeholders and writes a continuation:false diagnostics row", async () => {
		const h = await harness({ batchThresholdTokens: 1, prefixDiagnostics: true });
		for (let steps = 1; steps <= 3; steps += 1) await h.send(history(steps));
		// Defensive shape: a zero-assistant call that still carries an already swapped result.
		const odd = await h.send([user("start"), result(1)]);
		expect(isPlaceholder(odd[1])).toBe(true);
		const rows = await h.prefix();
		const last = rows.at(-1)!;
		expect([last.continuation, last.prevMessageCount, last.firstChangedIndex, last.repeat, last.flush]).toEqual([
			false,
			null,
			null,
			false,
			false,
		]);
	});

	it("legacy (T=0) with diagnostics ignores steering calls for prefix tracking and projects like the original", async () => {
		const calls: AgentMessage[][] = [history(1), history(2), [user("steer")], history(3), [user("steer")], history(4), history(4)];
		const run = async (factory: (pi: FakePi) => void) => {
			const dir = await tempDir("op-legacy-steer-");
			const pi = new FakePi();
			factory(pi);
			const outs: string[] = [];
			for (const messages of calls) {
				const ctx = fakeContext(new FakeSessionManager([], "root-a", dir), {
					mode: "tui",
					hasUI: true,
					ui: { notify: vi.fn(), setStatus: vi.fn() } as never,
					model: { provider: PROVIDER, id: MODEL } as never,
				});
				outs.push(JSON.stringify(await pi.emitContext(messages, ctx)));
			}
			return { outs, prefix: await readJsonl(join(dir, "sol-pi", "root-a", "observation-pack", "prefix-ledger.jsonl")) };
		};
		const original = await run((pi) => createOriginalExtension()(pi.asExtensionApi()));
		const legacy = await run((pi) =>
			createObservationPackExtension({ batchThresholdTokens: 0, prefixDiagnostics: true })(pi.asExtensionApi()),
		);
		expect(legacy.outs).toEqual(original.outs);
		expect(legacy.prefix.map((row) => row.continuation)).toEqual([true, true, false, true, false, true, true]);
		// History(3) after the steering call is compared with history(2), not with the steering call.
		expect(legacy.prefix[3]?.prevMessageCount).toBe(history(2).length);
		expect(legacy.prefix.every((row) => row.firstChangedIndex === null || row.continuation === true)).toBe(true);
	});
});

describe("round 2: process-wide ledger de-dup (D3 fix) does not suppress swaps", () => {
	it("after compaction repeats a request number, the placeholder is still sent and not re-toasted", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		for (let steps = 1; steps <= 5; steps += 1) await h.send(history(steps));
		const toasts = h.notify.mock.calls.length;
		// Compaction drops leading turns: the assistant count (request number) falls back to 4.
		const compacted = [user("summary"), ...history(5).slice(3)];
		const out = await h.send(compacted);
		const idx = compacted.findIndex((message) => message.role === "toolResult");
		expect(isPlaceholder(out[idx])).toBe(true);
		expect(h.notify.mock.calls.length).toBe(toasts);
		const keys = (await h.ledger()).map(keyOf);
		expect(keys.filter((key, index) => keys.indexOf(key) !== index)).toEqual([]);
	});

	it("legacy (T=0) writes no duplicate (event, id, request) rows across compaction", async () => {
		const h = await harness({ batchThresholdTokens: 0 });
		for (let steps = 1; steps <= 5; steps += 1) await h.send(history(steps));
		const compacted = [user("summary"), ...history(5).slice(3)];
		await h.send(compacted);
		await h.send([...compacted, assistant(9_000), result(9)]);
		const keys = (await h.ledger()).map(keyOf);
		expect(keys.filter((key, index) => keys.indexOf(key) !== index)).toEqual([]);
	});
});

describe("round 2: malformed tool results (R5 fix)", () => {
	const shapes: Array<[string, unknown]> = [
		["string content", "x".repeat(THRESHOLD_BYTES * 2)],
		["undefined content", undefined],
		["null block", [null]],
		["block without type", [{ text: "x".repeat(THRESHOLD_BYTES * 2) }]],
		["text block with non-string text", [{ type: "text", text: 42 }]],
	];
	for (const [name, content] of shapes) {
		for (const batchThresholdTokens of [1, 20_000]) {
			it(`batched T=${batchThresholdTokens}: ${name} neither throws nor alters the message`, async () => {
				const h = await harness({ batchThresholdTokens });
				const odd = { ...result(1), content } as unknown as AgentMessage;
				const messages = [user("start"), assistant(1_000), odd, assistant(2_000), result(2), assistant(3_000), result(3)];
				const out = await h.send(messages);
				expect(JSON.stringify(out[2])).toBe(JSON.stringify(odd));
			});
		}
	}

	// DEFECT R2-1 (low, pre-existing in 0bff376): legacy T=0 evaluates isPureTextResult outside its try, so a
	// null content block throws out of the context hook (B9). Batched mode is fixed. Flip to `it` once fixed.
	it.fails("legacy T=0 with a null content block does not throw out of the hook (original throws too)", async () => {
		const odd = { ...result(1), content: [null] } as unknown as AgentMessage;
		const messages = [user("start"), assistant(1_000), odd];
		const runOne = async (factory: (pi: FakePi) => void) => {
			const pi = new FakePi();
			factory(pi);
			const dir = await tempDir();
			const ctx = fakeContext(new FakeSessionManager([], "s", dir), { model: { provider: PROVIDER, id: MODEL } as never });
			return pi.emitContext(messages, ctx).then(
				() => "ok",
				() => "throws",
			);
		};
		const original = await runOne((pi) => createOriginalExtension()(pi.asExtensionApi()));
		const legacy = await runOne((pi) => createObservationPackExtension({ batchThresholdTokens: 0 })(pi.asExtensionApi()));
		// B9: never throw out of the hook. (T=0 must equal the original; if the original throws this documents the conflict.)
		expect({ original, legacy }).toEqual({ original, legacy: "ok" });
	});
});
