/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
// Verifier-added edge cases for cache-aware ObservationPack batching (spec B2 / B4 / B6).
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import {
	createObservationPackExtension,
	type ObservationPackOptions,
	THRESHOLD_BYTES,
} from "../src/sol-pi/extensions/observation-pack/index.ts";
import { FakePi, fakeContext } from "./helpers.ts";

const roots: string[] = [];
afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

const user = (text: string) => ({ role: "user", content: text, timestamp: 0 }) as AgentMessage;
const assistant = (timestamp: number) =>
	({
		role: "assistant",
		content: [{ type: "text", text: "calling a tool" }],
		api: "openai-completions",
		provider: "p",
		model: "m",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
		stopReason: "toolUse",
		timestamp,
	}) as unknown as AgentMessage;
function result(step: number, extraBytes = 64): ToolResultMessage {
	const line = `observation ${step} line\n`;
	const text = `result ${step}\n${line.repeat(Math.ceil((THRESHOLD_BYTES + extraBytes) / line.length))}`;
	return { role: "toolResult", toolCallId: `call-${step}`, toolName: "read", content: [{ type: "text", text }], isError: false, timestamp: step };
}
const isPlaceholder = (message: AgentMessage | undefined) =>
	message?.role === "toolResult" &&
	message.content.some((block) => block.type === "text" && block.text.startsWith("[large tool result replaced"));

async function harness(options: ObservationPackOptions) {
	const dir = await mkdtemp(join(tmpdir(), "observationpack-verifier-"));
	roots.push(dir);
	const clock = { now: 0 };
	const pi = new FakePi();
	createObservationPackExtension({ now: () => clock.now, ...options })(pi.asExtensionApi());
	return {
		clock,
		send: (messages: AgentMessage[]) => {
			const last = messages.findLast((message) => message.role === "assistant") as { timestamp?: number } | undefined;
			clock.now = Math.max(clock.now, (last?.timestamp ?? 0) + 1_000);
			return pi.emitContext(messages, fakeContext(dir, { model: { provider: "p", id: "m" } as never }));
		},
		ledger: async () =>
			(await readFile(join(dir, "sol-pi", "session-a", "observation-pack", "ledger.jsonl"), "utf8").catch(() => ""))
				.split("\n")
				.filter(Boolean)
				.map((line) => JSON.parse(line) as Record<string, unknown>),
	};
}

describe("observation pack batching: verifier edge cases", () => {
	// Session tree navigation (/tree, /branch) or an upstream hook that drops assistant turns
	// can present a swapped result with fewer than FULL_SENDS assistants after it.
	// B2: once sent as a placeholder for this root, it must stay a placeholder.
	it.each([
		["batched (T=20000)", { batchThresholdTokens: 20_000 }],
		["legacy (T=0)", { batchThresholdTokens: 0 }],
	] as const)("B2: a swapped observation never flips back when history is re-branched [%s]", async (_label, options) => {
		const h = await harness(options);
		const base = [user("start"), assistant(1_000), result(1), assistant(2_000), assistant(3_000)];
		const first = await h.send(base); // process-start flush in batched mode, legacy swap otherwise
		expect(isPlaceholder(first[2])).toBe(true);
		// Branch back to right after the tool result, then continue with one new assistant turn.
		const branched = [user("start"), assistant(1_000), result(1), assistant(4_000)];
		const second = await h.send(branched);
		expect(isPlaceholder(second[2])).toBe(true);
	});

	// Round 2 (replay of real omp sessions against 890ecc3): shapes that showed up in the replay.
	const stripTimestamps = (rows: Record<string, unknown>[]) => rows.map(({ timestamp: _t, ...row }) => JSON.stringify(row));
	const toolTurn = (step: number, timestamp: number, model = "m") =>
		[{ ...(assistant(timestamp) as object), model } as AgentMessage, result(step)] as AgentMessage[];

	it("omp live steering: zero-assistant context calls before every request leave outputs and ledger unchanged", async () => {
		const plain = await harness({ batchThresholdTokens: 30_000, prefixDiagnostics: true });
		const steered = await harness({ batchThresholdTokens: 30_000, prefixDiagnostics: true });
		// Timeline with a threshold flush, a cold gap and a model change.
		const times = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 400_000, 401_000, 402_000, 403_000, 404_000, 405_000];
		let history: AgentMessage[] = [user("start")];
		for (let step = 0; step < times.length; step += 1) {
			const model = step >= 9 ? "m2" : "m";
			history = [...history, ...toolTurn(step + 1, times[step] as number, model)];
			const steer = [user(`steer ${step}`)];
			const unchanged = await steered.send(steer);
			expect(JSON.stringify(unchanged)).toBe(JSON.stringify(steer));
			const a = await plain.send(history);
			const b = await steered.send(history);
			expect(JSON.stringify(b)).toBe(JSON.stringify(a));
		}
		const rowsA = await plain.ledger();
		expect(rowsA.some((row) => row.event === "placeholder")).toBe(true);
		expect(stripTimestamps(await steered.ledger())).toEqual(stripTimestamps(rowsA));
	});

	it("compaction repeats request numbers: no duplicate (event, id, request) rows and swapped results stay placeholders", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		let history: AgentMessage[] = [user("start")];
		for (let step = 1; step <= 6; step += 1) {
			history = [...history, ...toolTurn(step, step * 1_000)];
			await h.send(history);
		}
		// Compaction: summary + the last kept turns, so assistant counts (request numbers) drop and then rise again.
		const summary = { role: "compactionSummary", summary: "earlier work", tokensBefore: 1, timestamp: 6_500 } as unknown as AgentMessage;
		let compacted: AgentMessage[] = [summary, ...history.slice(-6)];
		const firstKept = compacted.findIndex((message) => message.role === "toolResult");
		const before = await h.send(compacted);
		expect(isPlaceholder(before[firstKept])).toBe(true);
		for (let step = 7; step <= 10; step += 1) {
			compacted = [...compacted, ...toolTurn(step, step * 1_000)];
			const out = await h.send(compacted);
			expect(isPlaceholder(out[firstKept])).toBe(true);
		}
		const rows = await h.ledger();
		const keys = rows.map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(new Set(keys).size).toBe(keys.length);
		const flushRows = rows.filter((row) => row.flushReason !== undefined).map((row) => row.id);
		expect(new Set(flushRows).size).toBe(flushRows.length);
	});
});
