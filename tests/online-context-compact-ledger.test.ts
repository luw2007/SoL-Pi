/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { afterEach, describe, expect, it } from "vitest";
import { createOnlineContextCompactExtension } from "../src/sol-pi/extensions/online-context-compact/extension.ts";
import { FakePi, FakeSessionManager, fakeContext } from "./helpers.ts";

const roots: string[] = [];
const SESSION_ID = "ledger-probe-session";

afterEach(async () => {
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

function assistant(text: string): AgentMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "test",
		provider: "test",
		model: "test",
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

async function sessionRoot(): Promise<string> {
	const value = await mkdtemp(join(tmpdir(), "online-compact-ledger-test-"));
	roots.push(value);
	return value;
}

async function readLedgerRows(sessionDir: string): Promise<Array<Record<string, unknown>>> {
	const content = await readFile(
		join(sessionDir, "sol-pi", SESSION_ID, "online-context-compact", "turn-end-ledger.jsonl"),
		"utf8",
	);
	return content
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function assistantMessage(overrides: Record<string, unknown> = {}): Record<string, unknown> {
	return {
		role: "assistant" as const,
		content: [],
		attribution: "agent",
		timestamp: Date.now(),
		stopReason: "stop",
		...overrides,
	};
}

describe("online context compact turn_end ledger", () => {
	it("records no_boundary when update_plan was never called, without ever evaluating decideCompaction", async () => {
		const sessionDir = await sessionRoot();
		const manager = new FakeSessionManager([], SESSION_ID, sessionDir);
		const pi = new FakePi();
		createOnlineContextCompactExtension()(pi.asExtensionApi());
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 1_000, contextWindow: 200_000, percent: 0.5 }),
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext([], context);
		await pi.emit(
			"turn_end",
			{ type: "turn_end", turnIndex: 1, message: assistantMessage(), toolResults: [] },
			context,
		);

		const rows = await readLedgerRows(sessionDir);
		expect(rows).toHaveLength(1);
		expect(rows[0]).toMatchObject({ bucket: "no_boundary", decisionEvaluated: false, willCompact: false });
	});

	it("records no_boundary on a repeat turn_end even while a prior decision is still pending settlement, because pendingBoundary was already consumed", async () => {
		const sessionDir = await sessionRoot();
		const manager = new FakeSessionManager([], SESSION_ID, sessionDir);
		const pi = new FakePi();
		createOnlineContextCompactExtension()(pi.asExtensionApi());
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 1_000, contextWindow: 200_000, percent: 0.5 }),
			// Never settles, so `selected` stays populated across the next turn_end.
			compact: () => new Promise<never>(() => {}),
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext([], context);

		const plan = pi.tool("update_plan");
		await plan.execute!(
			"plan-open",
			{ steps: [{ id: "s1", goal: "do it", status: "in_progress" }] },
			undefined,
			() => {},
			context,
		);
		await plan.execute!(
			"plan-done",
			{ steps: [{ id: "s1", goal: "do it", status: "completed" }] },
			undefined,
			() => {},
			context,
		);

		// First turn_end: boundary pending, nothing selected yet -> evaluates.
		await pi.emit(
			"turn_end",
			{
				type: "turn_end",
				turnIndex: 1,
				message: assistantMessage(),
				toolResults: [
					{ role: "toolResult", toolCallId: "plan-done", toolName: "update_plan", content: [], isError: false, timestamp: Date.now() },
				],
			},
			context,
		);

		const rowsAfterFirst = await readLedgerRows(sessionDir);
		expect(rowsAfterFirst).toHaveLength(1);
		expect(rowsAfterFirst[0]?.decisionEvaluated).toBe(true);

		// Second turn_end fires while the first compaction is still pending
		// (`selected` populated) but `pendingBoundary` was already consumed by
		// the first turn -> the `!boundary` guard wins, matching its precedence
		// over the `selected` check in the original `if (!boundary || selected)`.
		// This deliberately does NOT exercise the `already_selected` bucket —
		// see the next test for that, which requires a pending boundary AND a
		// pending selection simultaneously.
		await pi.emit(
			"turn_end",
			{ type: "turn_end", turnIndex: 2, message: assistantMessage(), toolResults: [] },
			context,
		);
		const rowsAfterSecond = await readLedgerRows(sessionDir);
		expect(rowsAfterSecond).toHaveLength(2);
		expect(rowsAfterSecond[1]?.bucket).toBe("no_boundary");
	});

	it("records already_selected when a new boundary becomes pending while a prior decision is still pending settlement", async () => {
		const sessionDir = await sessionRoot();
		const manager = new FakeSessionManager([], SESSION_ID, sessionDir);
		// Native compaction prices the provider-visible messages corresponding
		// to the real branch. Two large old turns exceed the replacement memo
		// cost while the final turn remains inside the native retained suffix.
		const messages: AgentMessage[] = Array.from({ length: 3 }, (_, i) => ({
			role: "user",
			content: `turn ${i}: ${"x".repeat(20_000)}`,
			timestamp: Date.now(),
		}));
		for (const message of messages) manager.appendMessage(message);
		const pi = new FakePi();
		// The small window triggers window protection once native history
		// yields positive savings, independently of the request horizon.
		createOnlineContextCompactExtension({ keepRecentTokens: 10 })(pi.asExtensionApi());
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 5_000, contextWindow: 20_000, percent: 0.25 }),
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext(messages, context);

		const plan = pi.tool("update_plan");
		await plan.execute!(
			"plan-open",
			{ steps: [{ id: "s1", goal: "do it", status: "in_progress" }] },
			undefined,
			() => {},
			context,
		);
		await plan.execute!(
			"plan-done-1",
			{ steps: [{ id: "s1", goal: "do it", status: "completed" }] },
			undefined,
			() => {},
			context,
		);

		// First turn_end: boundary pending, nothing selected yet -> evaluates,
		// decides to compact via `window_protection` (see fixture comment
		// above), and since this fake harness has no `ompHost`, the handler
		// sets `selected = { decision }` and returns without ever clearing it.
		await pi.emit(
			"turn_end",
			{
				type: "turn_end",
				turnIndex: 1,
				message: assistantMessage(),
				toolResults: [
					{ role: "toolResult", toolCallId: "plan-done-1", toolName: "update_plan", content: [], isError: false, timestamp: Date.now() },
				],
			},
			context,
		);
		const rowsAfterFirst = await readLedgerRows(sessionDir);
		expect(rowsAfterFirst).toHaveLength(1);
		expect(rowsAfterFirst[0]).toMatchObject({ decisionEvaluated: true, willCompact: true, reason: "window_protection" });

		// Complete a SECOND plan step before the second turn_end, so a fresh
		// `pendingBoundary` is set. This time `!boundary` is false but
		// `selected` is still truthy from the first turn, so the
		// `already_selected` bucket must win.
		await plan.execute!(
			"plan-step-2",
			{ steps: [{ id: "s2", goal: "do more", status: "in_progress" }] },
			undefined,
			() => {},
			context,
		);
		await plan.execute!(
			"plan-done-2",
			{ steps: [{ id: "s2", goal: "do more", status: "completed" }] },
			undefined,
			() => {},
			context,
		);
		await pi.emit(
			"turn_end",
			{
				type: "turn_end",
				turnIndex: 2,
				message: assistantMessage(),
				toolResults: [
					{ role: "toolResult", toolCallId: "plan-done-2", toolName: "update_plan", content: [], isError: false, timestamp: Date.now() },
				],
			},
			context,
		);
		const rowsAfterSecond = await readLedgerRows(sessionDir);
		expect(rowsAfterSecond).toHaveLength(2);
		expect(rowsAfterSecond[1]).toMatchObject({ bucket: "already_selected", decisionEvaluated: false });
	});

	it("performs the selected/abort handoff synchronously, before the ledger write's own I/O has a chance to let anything interleave", async () => {
		// Regression test for the actual bug: the ledger write used to be the
		// handler's FIRST await, sitting BEFORE `selected = { decision }` /
		// `context.abort()`. That let the promise machinery yield control
		// mid-handler while the decision to compact was already made but not
		// yet acted on — on pi (non-omp), `context.abort()` synchronously
		// triggers `agent_settled` -> compact's consumption of `selected` in
		// the SAME tick it is called, so anything that could observe or act on
		// state between "decided" and "handed off" was a live race. Proving
		// this requires observing state WHILE the handler's own promise is
		// still pending, not after `await`ing it to completion (every other
		// test in this suite awaits fully and so cannot detect this class of
		// bug — see the `extension.ts` call-site comment for why the fix
		// fires the ledger write without awaiting it first).
		const sessionDir = await sessionRoot();
		const manager = new FakeSessionManager([], SESSION_ID, sessionDir);
		const messages: AgentMessage[] = Array.from({ length: 3 }, (_, i) => ({
			role: "user",
			content: `turn ${i}: ${"x".repeat(20_000)}`,
			timestamp: Date.now(),
		}));
		for (const message of messages) manager.appendMessage(message);
		const pi = new FakePi();
		createOnlineContextCompactExtension({ keepRecentTokens: 10 })(pi.asExtensionApi());
		let aborted = false;
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 5_000, contextWindow: 20_000, percent: 0.25 }),
			abort: () => {
				aborted = true;
			},
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext(messages, context);

		const plan = pi.tool("update_plan");
		await plan.execute!(
			"plan-open",
			{ steps: [{ id: "s1", goal: "do it", status: "in_progress" }] },
			undefined,
			() => {},
			context,
		);
		await plan.execute!(
			"plan-done",
			{ steps: [{ id: "s1", goal: "do it", status: "completed" }] },
			undefined,
			() => {},
			context,
		);

		// Deliberately NOT awaited yet: a JS function call runs synchronously
		// up to its own first `await`/return, and that holds transitively
		// through every layer here (`FakePi.emit`'s own async body, calling
		// the turn_end handler, calling `compactionLedgerFor(context)(entry)`
		// which itself only suspends at `await mkdir` INSIDE the ledger, never
		// back out to this call site). So if the handoff is truly synchronous
		// and ahead of the ledger write, `aborted` must already be `true` the
		// instant this call returns — before any `await` in this test file
		// runs at all.
		const pending = pi.emit(
			"turn_end",
			{
				type: "turn_end",
				turnIndex: 1,
				message: assistantMessage(),
				toolResults: [
					{ role: "toolResult", toolCallId: "plan-done", toolName: "update_plan", content: [], isError: false, timestamp: Date.now() },
				],
			},
			context,
		);
		expect(aborted).toBe(true);

		await pending;
		const rows = await readLedgerRows(sessionDir);
		expect(rows).toHaveLength(1);
		expect(rows[0]).toMatchObject({ decisionEvaluated: true, willCompact: true, reason: "window_protection" });
	});

	it("records boundary_tool_result_error and never evaluates decideCompaction when the plan tool call itself errored", async () => {
		const sessionDir = await sessionRoot();
		const manager = new FakeSessionManager([], SESSION_ID, sessionDir);
		const pi = new FakePi();
		createOnlineContextCompactExtension()(pi.asExtensionApi());
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 1_000, contextWindow: 200_000, percent: 0.5 }),
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext([], context);

		const plan = pi.tool("update_plan");
		await plan.execute!(
			"plan-open",
			{ steps: [{ id: "s1", goal: "do it", status: "in_progress" }] },
			undefined,
			() => {},
			context,
		);
		await plan.execute!(
			"plan-done",
			{ steps: [{ id: "s1", goal: "do it", status: "completed" }] },
			undefined,
			() => {},
			context,
		);

		await pi.emit(
			"turn_end",
			{
				type: "turn_end",
				turnIndex: 1,
				message: assistantMessage(),
				toolResults: [
					{ role: "toolResult", toolCallId: "plan-done", toolName: "update_plan", content: [], isError: true, timestamp: Date.now() },
				],
			},
			context,
		);

		const rows = await readLedgerRows(sessionDir);
		expect(rows).toHaveLength(1);
		expect(rows[0]).toMatchObject({ bucket: "boundary_tool_result_error", decisionEvaluated: false });
	});

	it("fails open when the session has no persistent directory, without breaking the turn's own compaction decision", async () => {
		const manager = new FakeSessionManager([], SESSION_ID, "");
		const pi = new FakePi();
		createOnlineContextCompactExtension()(pi.asExtensionApi());
		const context = fakeContext(manager, {
			getSystemPrompt: () => "test prompt",
			getContextUsage: () => ({ tokens: 1_000, contextWindow: 200_000, percent: 0.5 }),
		});

		await pi.emit("session_start", { type: "session_start" }, context);
		await pi.emitContext([], context);

		// getSessionDir() returning "" makes runtimeRoot() throw inside the ledger
		// resolution path; the turn_end handler must still complete normally.
		await expect(
			pi.emit(
				"turn_end",
				{ type: "turn_end", turnIndex: 1, message: assistantMessage(), toolResults: [] },
				context,
			),
		).resolves.not.toThrow();
	});
});
