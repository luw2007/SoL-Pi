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

});
