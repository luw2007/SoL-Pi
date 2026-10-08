/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * Robustness and omp-compat verification for ObservationPack cache-aware batching.
 *
 * Defects found by earlier rounds are fixed and their tests are plain `it`.
 * Round 1 additions are in the "round 1" describe blocks at the end.
 * Round 2 additions (against 7251819) are in the "round 2" describe blocks; R9
 * (main request misclassified as an omp side turn) is fixed too.
 */
import { spawnSync } from "node:child_process";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG, loadSolPiConfig } from "../src/sol-pi/config.ts";
import {
	createObservationPackExtension,
	type ObservationPackOptions,
	THRESHOLD_BYTES,
} from "../src/sol-pi/extensions/observation-pack/index.ts";
import { FakePi, fakeContext } from "./helpers.ts";

const SESSION_ID = "session-a";
const PROVIDER = "provider-a";
const MODEL = "model-a";
const roots: string[] = [];

afterEach(async () => {
	for (const root of roots) await chmod(join(root, "sol-pi"), 0o700).catch(() => undefined);
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
	vi.restoreAllMocks();
});

async function sessionRoot(): Promise<string> {
	const value = await mkdtemp(join(tmpdir(), "op-robust-verify-"));
	roots.push(value);
	return value;
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

/** [user, a1, r1, a2, r2, ...]; assistant k is stamped k seconds. */
function history(steps: number): AgentMessage[] {
	const messages: AgentMessage[] = [user("start")];
	for (let step = 1; step <= steps; step += 1) messages.push(assistant(step * 1_000), result(step));
	return messages;
}

const resultIndex = (step: number): number => step * 2;

function textOf(message: AgentMessage | undefined): string {
	if (!message || message.role !== "toolResult") throw new Error("expected tool result");
	return message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

const isPlaceholder = (message: AgentMessage | undefined): boolean =>
	textOf(message).startsWith("[large tool result replaced");

interface Harness {
	readonly pi: FakePi;
	readonly clock: { now: number };
	readonly sessionDir: string;
	readonly packDir: string;
	context(overrides?: Partial<ExtensionContext>): ExtensionContext;
	send(messages: readonly AgentMessage[], overrides?: Partial<ExtensionContext>): Promise<AgentMessage[]>;
	ledger(): Promise<Record<string, unknown>[]>;
	rawLedger(): Promise<string>;
}

async function harness(options: ObservationPackOptions = {}, sessionDir?: string): Promise<Harness> {
	const dir = sessionDir ?? (await sessionRoot());
	const clock = { now: 0 };
	const pi = new FakePi();
	createObservationPackExtension({ batchThresholdTokens: 5_000, now: () => clock.now, ...options })(pi.asExtensionApi());
	const packDir = join(dir, "sol-pi", SESSION_ID, "observation-pack");
	const rawLedger = () => readFile(join(packDir, "ledger.jsonl"), "utf8").catch(() => "");
	const context = (overrides: Partial<ExtensionContext> = {}) =>
		fakeContext(dir, { model: { provider: PROVIDER, id: MODEL } as never, ...overrides });
	return {
		pi,
		clock,
		sessionDir: dir,
		packDir,
		context,
		async send(messages, overrides) {
			const last = messages.findLast((message) => message.role === "assistant");
			const stamp = (last as { timestamp?: unknown } | undefined)?.timestamp;
			clock.now = Math.max(clock.now, (typeof stamp === "number" && Number.isFinite(stamp) ? stamp : 0) + 1_000);
			return pi.emitContext(messages, context(overrides));
		},
		async ledger() {
			return (await rawLedger())
				.split("\n")
				.filter(Boolean)
				.map((line) => JSON.parse(line) as Record<string, unknown>);
		},
		rawLedger,
	};
}

async function sendSteps(h: Harness, from: number, to: number): Promise<AgentMessage[]> {
	let projected: AgentMessage[] = [];
	for (let steps = from; steps <= to; steps += 1) projected = await h.send(history(steps));
	return projected;
}

function silenceFailOpen(): string[] {
	const errors: string[] = [];
	vi.spyOn(console, "error").mockImplementation((...values: unknown[]) => {
		errors.push(values.map(String).join(" "));
	});
	return errors;
}

describe("omp host call patterns", () => {
	// omp (providers.openaiLiveSteering, default true) runs the context hook on
	// ONLY the steering messages typed while a Codex response streams
	// (packages/agent/src/live-steering.ts: toProvider -> transformContext(steeringMessages)).
	// That is not a provider request of the conversation and must not become the
	// "previous request" used for prefix-change detection.
	it("a live-steering context call (steering messages only) does not trigger a prefix-changed flush", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 4); // results 1 and 2 are pending (eligible, below threshold)
		await h.send([user("steer: also check the tests")]);
		const next = [...history(4), user("steer: also check the tests"), assistant(5_000), result(5)];
		const projected = await h.send(next);
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(false);
		expect(isPlaceholder(projected[resultIndex(2)])).toBe(false);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it("a side request (history + extra prompt, e.g. handoff / ephemeral turn) does not trigger a flush", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 4);
		await h.send([...history(4), user("side question")]);
		const projected = await h.send(history(5));
		expect([1, 2, 3].map((step) => isPlaceholder(projected[resultIndex(step)]))).toEqual([false, false, false]);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it("ignores omp symbol-keyed per-call metadata when fingerprinting", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const historyIndex = Symbol("agent.message.contextHistoryIndex");
		const perCall = Symbol("agent.message.perCallContext");
		const tag = (messages: AgentMessage[], salt: number) =>
			messages.map((message, index) => Object.assign({ ...message }, { [historyIndex]: index + salt, [perCall]: salt % 2 === 0 }));
		for (let steps = 1; steps <= 5; steps += 1) {
			// emitContext in FakePi structuredClones (drops symbols), so tag after cloning by calling the handler directly.
			const handler = h.pi.handlers.get("context")?.[0];
			h.clock.now = steps * 1_000 + 1_000;
			const out = (await handler?.({ type: "context", messages: tag(history(steps), steps * 7) }, h.context())) as {
				messages: AgentMessage[];
			};
			for (let step = 1; step <= steps; step += 1) expect(isPlaceholder(out.messages[resultIndex(step)])).toBe(false);
		}
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it("tolerates omp message roles and shapes (compaction summary, custom, developer, fileMention, completedAt)", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		const extra = [
			{ role: "compactionSummary", summary: "earlier work", tokensBefore: 123, timestamp: 1 },
			{ role: "branchSummary", summary: "branch", fromId: "x", timestamp: 1 },
			{ role: "custom", customType: "advisor", content: "note", display: false, timestamp: 1 },
			{ role: "developer", content: [{ type: "text", text: "dev" }], timestamp: 1 },
			{ role: "fileMention", files: [{ path: "/a" }], timestamp: 1 },
		] as unknown as AgentMessage[];
		const withShapes = (steps: number) => {
			const messages = [...extra, ...history(steps)];
			return messages.map((message) =>
				message.role === "assistant" ? ({ ...message, completedAt: 999_999, duration: 5 } as AgentMessage) : message,
			);
		};
		let projected: AgentMessage[] = [];
		for (let steps = 1; steps <= 4; steps += 1) projected = await h.send(withShapes(steps));
		expect(isPlaceholder(projected[extra.length + resultIndex(1)])).toBe(true);
		expect(errors).toEqual([]);
	});
});

describe("state transitions", () => {
	// Spec B2: "once an observation has been sent as a placeholder for a session
	// root, every later request of that root sends the placeholder. Never flip back."
	// The original implementation kept this via sentCounts; batched mode derives
	// the count from history only, so a rewind (omp checkpoint/rewind tool,
	// /tree navigation, dropped assistant turn) that leaves fewer than FULL_SENDS
	// assistants after the result sends it in full again.
	it("a swapped observation stays a placeholder after history is rewound", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		const swapped = await sendSteps(h, 1, 4);
		expect(isPlaceholder(swapped[resultIndex(1)])).toBe(true);
		// Rewind to just after assistant 2: result 1 now has only one assistant after it.
		const rewound = [...history(1), assistant(2_000, "retry from checkpoint")];
		h.clock.now = 10_000;
		const projected = await h.pi.emitContext(rewound, h.context());
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(true);
	});

	it("legacy mode (T=0) keeps the swapped placeholder after the same rewind (reference behaviour)", async () => {
		const h = await harness({ batchThresholdTokens: 0 });
		await sendSteps(h, 1, 4);
		const rewound = [...history(1), assistant(2_000, "retry from checkpoint")];
		const projected = await h.pi.emitContext(rewound, h.context());
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(true);
	});

	it("retries of an aborted request (same messages) neither advance nor duplicate", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 3);
		for (let attempt = 0; attempt < 5; attempt += 1) await h.send(history(4));
		const rows = await h.ledger();
		const keys = rows.map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(new Set(keys).size).toBe(keys.length);
		expect(rows.filter((row) => row.flushReason)).toEqual([]);
	});

	// Spec B6: "no duplicate row for the same (root, observation id, request, event)
	// within a process" and the toast fires once per observation.
	it("concurrent context calls for the same request do not duplicate ledger rows or toasts", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		const notify = vi.fn();
		const tui = { mode: "tui", hasUI: true, ui: { notify, setStatus: vi.fn() } as never } as Partial<ExtensionContext>;
		await sendSteps(h, 1, 2);
		h.clock.now = 4_000;
		await Promise.all([
			h.pi.emitContext(history(3), h.context(tui)),
			h.pi.emitContext(history(3), h.context(tui)),
		]);
		const rows = await h.ledger();
		const keys = rows.map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(new Set(keys).size).toBe(keys.length);
		expect(notify).toHaveBeenCalledTimes(1);
	});

	it("resume in a fresh process toasts every eligible old observation at once (documents toast volume)", async () => {
		const dir = await sessionRoot();
		const notify = vi.fn();
		const tui = { mode: "tui", hasUI: true, ui: { notify, setStatus: vi.fn() } as never } as Partial<ExtensionContext>;
		const restarted = await harness({}, dir);
		await restarted.send(history(12), tui);
		// 10 of the 12 results are eligible (>= FULL_SENDS assistants after them).
		expect(notify).toHaveBeenCalledTimes(10);
		const legacyDir = await sessionRoot();
		const legacyNotify = vi.fn();
		const legacy = await harness({ batchThresholdTokens: 0 }, legacyDir);
		await legacy.send(history(12), { mode: "tui", hasUI: true, ui: { notify: legacyNotify, setStatus: vi.fn() } as never });
		expect(legacyNotify).toHaveBeenCalledTimes(1);
	});
});

describe("fail-open", () => {
	it("sends full text and does not throw when the archive directory cannot be created", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		await mkdir(h.packDir, { recursive: true });
		await writeFile(join(h.packDir, "objects"), "not a directory");
		const projected = await sendSteps(h, 1, 4);
		for (let step = 1; step <= 4; step += 1) expect(textOf(projected[resultIndex(step)])).toBe(textOf(result(step)));
		expect(errors.length).toBeGreaterThan(0);
	});

	it("sends full text when the ledger is unwritable and swaps later once it is writable again", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 2);
		await rm(join(h.packDir, "ledger.jsonl"));
		await mkdir(join(h.packDir, "ledger.jsonl"));
		const blocked = await h.send(history(3));
		expect(textOf(blocked[resultIndex(1)])).toBe(textOf(result(1)));
		expect(errors.some((error) => error.includes("fail-open"))).toBe(true);
		await rm(join(h.packDir, "ledger.jsonl"), { recursive: true });
		const recovered = await h.send(history(4));
		expect(isPlaceholder(recovered[resultIndex(1)])).toBe(true);
		const flush = (await h.ledger()).filter((row) => row.flushReason);
		expect(flush.length).toBeGreaterThan(0);
	});

	it("an already-swapped observation falls back to full text (fail-open) if its ledger write fails", async () => {
		silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		const swapped = await sendSteps(h, 1, 3);
		expect(isPlaceholder(swapped[resultIndex(1)])).toBe(true);
		await rm(join(h.packDir, "ledger.jsonl"));
		await mkdir(join(h.packDir, "ledger.jsonl"));
		const projected = await h.send(history(4));
		// Fail-open wins over monotonicity here; recorded so a change is noticed.
		expect(textOf(projected[resultIndex(1)])).toBe(textOf(result(1)));
	});

	it.each([
		["NaN", Number.NaN],
		["Infinity", Number.POSITIVE_INFINITY],
		["string", "2026-09-28T00:00:00.000Z"],
		["missing", undefined],
		["null", null],
		["future", 10 ** 15],
	])("does not throw or flush on an assistant timestamp that is %s", async (_label, stamp) => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 3);
		const messages = history(4);
		messages[resultIndex(4) - 1] = assistant(stamp);
		h.clock.now = 5_000;
		const projected = await h.pi.emitContext(messages, h.context());
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(false);
		expect(errors).toEqual([]);
	});

	it("does not throw when assistant provider/model are missing or non-strings", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 3);
		const messages = history(4);
		messages[resultIndex(4) - 1] = assistant(4_000, "x", undefined, 42);
		const projected = await h.send(messages);
		expect(isPlaceholder(projected[resultIndex(1)])).toBe(false);
		expect(errors).toEqual([]);
	});

	it("does not throw when ctx.model has a non-string id or a throwing getter", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 3);
		await expect(h.send(history(4), { model: { provider: 1, id: null } as never })).resolves.toBeDefined();
		const ctx = h.context();
		Object.defineProperty(ctx, "model", {
			get() {
				throw new Error("boom");
			},
		});
		await expect(h.pi.emitContext(history(5), ctx)).resolves.toBeDefined();
	});

	it("an unserializable message (BigInt) fails open: full text, no throw", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		const handler = h.pi.handlers.get("context")?.[0];
		for (let steps = 1; steps <= 4; steps += 1) {
			const messages = history(steps);
			(messages[0] as unknown as { details: unknown }).details = { big: 10n };
			h.clock.now = steps * 1_000 + 1_000;
			const out = (await handler?.({ type: "context", messages }, h.context())) as { messages: AgentMessage[] };
			expect(out.messages).toHaveLength(messages.length);
		}
		expect(errors.some((error) => error.includes("observation batching"))).toBe(true);
	});

	// Pre-existing (the original hook has the same guard outside try/catch), but
	// spec B9 says the hook must never throw.
	it("a toolResult without a content array does not throw out of the hook", async () => {
		silenceFailOpen();
		const h = await harness();
		const broken = [...history(1), { role: "toolResult", toolCallId: "x", toolName: "read", isError: false } as unknown as AgentMessage];
		await expect(h.send(broken)).resolves.toBeDefined();
	});

	it("a context call without a persistent session directory does not throw", async () => {
		const h = await harness();
		const manager = { getSessionDir: () => undefined, getSessionId: () => SESSION_ID };
		await expect(h.pi.emitContext(history(1), fakeContext(h.sessionDir, { sessionManager: manager as never }))).resolves.toBeDefined();
	});
});

describe("ledger backward compatibility", () => {
	const LEGACY_FULL = ["contentHash", "event", "id", "originalBytes", "originalLines", "originalTokens", "request", "timestamp", "tool"];
	const LEGACY_PLACEHOLDER = [
		"event",
		"id",
		"originalBytes",
		"originalLines",
		"originalTokens",
		"placeholderBytes",
		"placeholderTokens",
		"removedTokens",
		"request",
		"sendNumber",
		"timestamp",
		"tool",
	];

	it("every row is one valid JSON object that keeps all fields existing readers use", async () => {
		const h = await harness();
		await sendSteps(h, 1, 8);
		const dir = await sessionRoot();
		const restarted = await harness({}, dir);
		await restarted.send(history(8));
		for (const raw of [await h.rawLedger(), await restarted.rawLedger()]) {
			expect(raw.endsWith("\n")).toBe(true);
			const lines = raw.split("\n").filter(Boolean);
			expect(lines.length).toBeGreaterThan(0);
			for (const line of lines) {
				const row = JSON.parse(line) as Record<string, unknown>;
				const required = row.event === "full" ? LEGACY_FULL : LEGACY_PLACEHOLDER;
				expect(Object.keys(row)).toEqual(expect.arrayContaining(required));
				expect(typeof row.request).toBe("number");
				expect(Number.isFinite(Date.parse(String(row.timestamp)))).toBe(true);
				if (row.event === "placeholder") expect(row.sendNumber as number).toBeGreaterThan(2);
				if ("deferred" in row) expect(row).toMatchObject({ event: "full", deferred: true });
				if ("flushReason" in row) {
					expect(["threshold", "cold-gap", "model-change", "prefix-changed", "process-start", "legacy"]).toContain(row.flushReason);
				}
			}
		}
	});

	it("each observation gets exactly one flushReason row per process, on its first placeholder", async () => {
		const h = await harness();
		await sendSteps(h, 1, 12);
		const rows = await h.ledger();
		const byId = new Map<string, Record<string, unknown>[]>();
		for (const row of rows.filter((candidate) => candidate.event === "placeholder")) {
			byId.set(String(row.id), [...(byId.get(String(row.id)) ?? []), row]);
		}
		expect(byId.size).toBeGreaterThan(0);
		for (const placeholderRows of byId.values()) {
			expect(placeholderRows.filter((row) => row.flushReason)).toHaveLength(1);
			expect(placeholderRows[0]?.flushReason).toBeDefined();
		}
	});
});

describe("configuration compatibility", () => {
	const configRoots: string[] = [];
	afterEach(() => {
		for (const root of configRoots.splice(0)) rmSync(root, { recursive: true, force: true });
	});
	function fixture(value: unknown): { cwd: string; agentDir: string; path: string } {
		const root = mkdtempSync(join(tmpdir(), "op-robust-config-"));
		configRoots.push(root);
		const cwd = join(root, "project");
		const agentDir = join(root, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
		const path = join(agentDir, "sol-pi.json");
		writeFileSync(path, typeof value === "string" ? value : JSON.stringify(value));
		return { cwd, agentDir, path };
	}
	const preflight = (path: string) =>
		spawnSync(process.execPath, [join(process.cwd(), "scripts/check-sol-pi-config.mjs"), "--config", path], {
			encoding: "utf8",
		});

	it("an existing user config without the new keys loads with the new defaults (config.ts and preflight)", () => {
		const legacyUserConfig = {
			version: 1,
			actionFusion: false,
			observationPack: true,
			evidencePreservingReducer: true,
			evidencePreservingReducerProvider: "coco",
			evidencePreservingReducerModel: "deepseek-v4-flash",
			onlineContextCompact: true,
			cacheWriteReadRatio: 12.5,
		};
		const { cwd, agentDir, path } = fixture(legacyUserConfig);
		const loaded = loadSolPiConfig(cwd, agentDir, true);
		expect(loaded).toMatchObject({
			observationPackBatchThresholdTokens: DEFAULT_CONFIG.observationPackBatchThresholdTokens,
			observationPackColdGapMs: DEFAULT_CONFIG.observationPackColdGapMs,
			observationPackPrefixDiagnostics: false,
		});
		const result = preflight(path);
		expect(result.status, result.stderr).toBe(0);
		expect(JSON.parse(result.stdout).effective_config).toMatchObject({
			observationPackBatchThresholdTokens: 20_000,
			observationPackColdGapMs: 300_000,
			observationPackPrefixDiagnostics: false,
		});
	});

	it.each([
		["observationPackBatchThresholdTokens", 2 ** 53],
		["observationPackBatchThresholdTokens", true],
		["observationPackBatchThresholdTokens", [1]],
		["observationPackBatchThresholdTokens", "1e3"],
		["observationPackColdGapMs", -1],
		["observationPackColdGapMs", 2 ** 53],
		["observationPackColdGapMs", "300000"],
		["observationPackColdGapMs", false],
		["observationPackPrefixDiagnostics", 1],
		["observationPackPrefixDiagnostics", "false"],
		["observationPackPrefixDiagnostics", {}],
	])("both validators reject %s = %j with a direct error", (key, value) => {
		const { cwd, agentDir, path } = fixture({ version: 1, observationPack: true, [key]: value });
		expect(() => loadSolPiConfig(cwd, agentDir, true)).toThrow(key);
		const result = preflight(path);
		expect(result.status).not.toBe(0);
		expect(result.stderr + result.stdout).toContain(key);
	});

	it.each([
		["observationPackBatchThresholdTokens", Number.MAX_SAFE_INTEGER],
		["observationPackBatchThresholdTokens", 0],
		["observationPackColdGapMs", 1],
		["observationPackColdGapMs", Number.MAX_SAFE_INTEGER],
	])("both validators accept boundary %s = %j", (key, value) => {
		const { cwd, agentDir, path } = fixture({ version: 1, observationPack: true, [key]: value });
		expect(loadSolPiConfig(cwd, agentDir, true)).toMatchObject({ [key]: value });
		const result = preflight(path);
		expect(result.status, result.stderr).toBe(0);
	});

	it("both validators reject an unknown near-miss key", () => {
		const { cwd, agentDir, path } = fixture({ version: 1, observationPackBatchThreshold: 1 });
		expect(() => loadSolPiConfig(cwd, agentDir, true)).toThrow();
		expect(preflight(path).status).not.toBe(0);
	});
});

describe("scale", () => {
	it("handles a large context (2,000 messages, 400 large results) within a bounded time per call", async () => {
		const h = await harness({ prefixDiagnostics: true });
		const big = history(400);
		for (let index = 0; index < 1_200; index += 1) big.splice(1, 0, user(`filler ${index}`));
		const started = performance.now();
		await h.send(big);
		const first = performance.now() - started;
		const again = performance.now();
		await h.send(big);
		const repeat = performance.now() - again;
		console.info(`[verify] 2000-msg context: first ${first.toFixed(0)} ms, repeat ${repeat.toFixed(0)} ms`);
		expect(repeat).toBeLessThan(5_000);
	}, 60_000);
});

/** omp runEphemeralTurn (/btw) context: main history + [developer, (user q, assistant a)*k, user q]. */
function btwContext(main: readonly AgentMessage[], previousAnswers: number, stamp: number): AgentMessage[] {
	const side: AgentMessage[] = [
		...main,
		{ role: "developer", content: [{ type: "text", text: "side question rules" }], attribution: "agent", timestamp: stamp } as unknown as AgentMessage,
	];
	for (let index = 0; index < previousAnswers; index += 1) {
		side.push(user(`btw question ${index}`), assistant(stamp, `btw answer ${index}`));
	}
	side.push(user("btw question now"));
	return side;
}

describe("round 1: omp side turns on the main root (/btw, runEphemeralTurn)", () => {
	// omp session.runEphemeralTurn -> convertMessagesToLlm -> transformContext runs the
	// context hook of the MAIN session root with [...messages, developer, ...btw history, user].
	// /btw follow-ups carry earlier side answers as synthetic assistant messages, so every
	// main observation appears with k extra assistants after it. The history-derived count
	// then makes a result eligible (and flushable) in the side call although the main stream
	// has sent it fewer than FULL_SENDS times; the side call's swap is monotonic, so the next
	// MAIN request sends that result as a placeholder at its 2nd main send (B1) and edits the
	// main prompt on a warm request. Pre-existing class: 0bff376 advanced sentCounts on every
	// side call, which is worse (k = 0 already triggered it).
	// Fixed in round 2 (R7): batched mode recognises the side-turn suffix, counts only main
	// assistants and neither flushes nor updates the previous request on it.
	it("a /btw follow-up with two earlier answers does not swap a result the main stream sent only once", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 2); // r1 sent twice, r2 sent once on the main stream
		await h.send(btwContext(history(2), 2, 3_000));
		const main = await h.send(history(3)); // r2's second main send
		expect(isPlaceholder(main[resultIndex(2)])).toBe(false);
	});

	it("a first /btw question (no earlier answers) does not swap anything early on the main stream", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 2);
		await h.send(btwContext(history(2), 0, 3_000));
		const main = await h.send(history(3));
		expect(isPlaceholder(main[resultIndex(1)])).toBe(true); // eligible on main anyway (T=1)
		expect(isPlaceholder(main[resultIndex(2)])).toBe(false);
		expect(isPlaceholder(main[resultIndex(3)])).toBe(false);
	});

	it("a /btw during streaming (partial assistant appended) does not cause a prefix flush on the next main request", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		await sendSteps(h, 1, 4);
		const partial = assistant(5_000, "partial answer so far");
		await h.send([...btwContext([...history(4), partial], 0, 5_500)]);
		const projected = await h.send(history(5));
		expect([1, 2, 3].map((step) => isPlaceholder(projected[resultIndex(step)]))).toEqual([false, false, false]);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});
});

describe("round 1: concurrency and repeats after the fix", () => {
	it("interleaved concurrent calls for different requests neither throw, duplicate rows, nor toast twice", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		const notify = vi.fn();
		const tui = { mode: "tui", hasUI: true, ui: { notify, setStatus: vi.fn() } as never } as Partial<ExtensionContext>;
		await sendSteps(h, 1, 2);
		h.clock.now = 10_000;
		const [three, four, threeAgain] = await Promise.all([
			h.pi.emitContext(history(3), h.context(tui)),
			h.pi.emitContext(history(4), h.context(tui)),
			h.pi.emitContext(history(3), h.context(tui)),
		]);
		expect(isPlaceholder(three[resultIndex(1)])).toBe(true);
		expect(isPlaceholder(threeAgain[resultIndex(1)])).toBe(true);
		expect([1, 2].map((step) => isPlaceholder(four[resultIndex(step)]))).toEqual([true, true]);
		const rows = await h.ledger();
		const keys = rows.map((row) => `${row.event}:${row.id}:${row.request}`);
		expect(new Set(keys).size).toBe(keys.length);
		const flushIds = rows.filter((row) => row.flushReason).map((row) => row.id);
		expect(new Set(flushIds).size).toBe(flushIds.length);
		expect(notify).toHaveBeenCalledTimes(flushIds.length);
		// Later requests keep both placeholders (monotonic).
		const later = await h.send(history(5));
		expect([1, 2].map((step) => isPlaceholder(later[resultIndex(step)]))).toEqual([true, true]);
	});

	it("many live-steering calls between two requests never flush and never grow the prefix state", async () => {
		const h = await harness({ batchThresholdTokens: 100_000, prefixDiagnostics: true });
		await sendSteps(h, 1, 4);
		for (let index = 0; index < 20; index += 1) await h.send([user(`steer ${index}`)]);
		const projected = await h.send([...history(4), user("steer 19"), assistant(5_000), result(5)]);
		expect([1, 2, 3].map((step) => isPlaceholder(projected[resultIndex(step)]))).toEqual([false, false, false]);
		const diagnostics = (await readFile(join(h.packDir, "prefix-ledger.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line) as Record<string, unknown>);
		const steering = diagnostics.filter((row) => row.continuation === false);
		expect(steering).toHaveLength(20);
		for (const row of steering) expect(row).toMatchObject({ flush: false, prevMessageCount: null, firstChangedIndex: null });
		expect(diagnostics.at(-1)).toMatchObject({ continuation: true, firstChangedIndex: null, flush: false });
	});

	it("a zero-assistant call with a large tool result (odd host shape) sends it in full and does not throw", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		const projected = await h.send([user("start"), result(1), result(2)]);
		expect(textOf(projected[1])).toBe(textOf(result(1)));
		expect(errors).toEqual([]);
	});
});

describe("round 1: ledger after compaction and rewind", () => {
	// B6 dedup is process-wide on (event, id, request). The request number is the
	// assistant count + 1, so after compaction it repeats; a real re-send of the same
	// observation under a request number it already used gets no ledger row.
	it("documents: after compaction a placeholder re-sent under an already-used request number gets no new ledger row", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 8);
		const before = (await h.ledger()).length;
		// Compaction keeps steps 3..8 behind a summary: 6 assistants -> request 7 again.
		const compacted = [
			{ role: "compactionSummary", summary: "steps 1-2", tokensBefore: 1, timestamp: 1 } as unknown as AgentMessage,
			...history(8).slice(resultIndex(2) + 1),
		];
		const projected = await h.send(compacted);
		const placeholders = projected.filter((message) => message.role === "toolResult" && isPlaceholder(message)).length;
		const newRows = (await h.ledger()).slice(before).filter((row) => row.event === "placeholder").length;
		expect(placeholders).toBeGreaterThan(0);
		expect(newRows).toBeLessThan(placeholders);
	});

	it("documents: a placeholder re-sent after a rewind is logged with sendNumber <= FULL_SENDS", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 4);
		h.clock.now = 20_000;
		await h.pi.emitContext([...history(1), assistant(2_000, "retry from checkpoint")], h.context());
		const rewoundRows = (await h.ledger()).filter((row) => row.event === "placeholder" && row.request === 3 && (row.sendNumber as number) <= 2);
		expect(rewoundRows.length).toBe(1);
	});
});

describe("round 1: scale and state growth", () => {
	it("per-call latency over a long growing session stays bounded and close to legacy", async () => {
		const run = async (batchThresholdTokens: number) => {
			const h = await harness({ batchThresholdTokens });
			const samples: number[] = [];
			for (let steps = 1; steps <= 120; steps += 1) {
				const messages = history(steps);
				const started = performance.now();
				await h.send(messages);
				samples.push(performance.now() - started);
			}
			const tail = samples.slice(-40).sort((a, b) => a - b);
			return { p50: tail[20] ?? 0, p95: tail[37] ?? 0, rows: (await h.ledger()).length };
		};
		const legacy = await run(0);
		const batched = await run(20_000);
		console.info(`[verify] 120-step session: legacy p50 ${legacy.p50.toFixed(1)} p95 ${legacy.p95.toFixed(1)} ms rows ${legacy.rows}; batched p50 ${batched.p50.toFixed(1)} p95 ${batched.p95.toFixed(1)} ms rows ${batched.rows}`);
		expect(batched.p95).toBeLessThan(Math.max(250, legacy.p95 * 3));
		// One ledger row (and one de-dup key) per (observation, request): quadratic in session length, capped at 100k keys per root.
		expect(batched.rows).toBeLessThan(120 * 121);
	}, 120_000);
});

// ---------------------------------------------------------------------------
// Round 2 (against 7251819): warming-aware cold gap and omp side turns.
// ---------------------------------------------------------------------------

type OmpEntry = Record<string, unknown> & { id: string; parentId: string | null; type: string; timestamp: string };

/**
 * Session manager shaped like omp 18.3.5 `SessionManager` for the parts ObservationPack reads:
 * Map-backed getEntry, getLeafEntry, and appendModelUsage that (like omp) moves the leaf onto the
 * usage entry when parentId is the current leaf, and keeps the leaf otherwise.
 */
class OmpSessionManager {
	readonly entries = new Map<string, OmpEntry>();
	leafId: string | null = null;
	private next = 1;
	constructor(
		readonly sessionDir: string,
		readonly sessionId = SESSION_ID,
	) {}
	getSessionId(): string {
		return this.sessionId;
	}
	getSessionDir(): string {
		return this.sessionDir;
	}
	getSessionFile(): string {
		return join(this.sessionDir, `${this.sessionId}.jsonl`);
	}
	getLeafId(): string | null {
		return this.leafId;
	}
	getLeafEntry(): OmpEntry | undefined {
		return this.leafId === null ? undefined : this.entries.get(this.leafId);
	}
	getEntry(id: string): OmpEntry | undefined {
		return this.entries.get(id);
	}
	getEntries(): OmpEntry[] {
		return [...this.entries.values()];
	}
	getBranch(): OmpEntry[] {
		const branch: OmpEntry[] = [];
		for (let entry = this.getLeafEntry(); entry; entry = entry.parentId ? this.entries.get(entry.parentId) : undefined) branch.unshift(entry);
		return branch;
	}
	private push(entry: Omit<OmpEntry, "id" | "parentId">, parentId = this.leafId, at = Date.now()): string {
		const id = `e${this.next++}`;
		this.entries.set(id, { ...entry, id, parentId, timestamp: new Date(at).toISOString() } as OmpEntry);
		this.leafId = id;
		return id;
	}
	appendMessage(message: AgentMessage): string {
		return this.push({ type: "message", message } as never);
	}
	/** omp `appendModelUsage({purpose, api, provider, model, usage, stopReason}, {sessionId, parentId})`. */
	appendModelUsage(at: number, usage: Record<string, unknown>, purpose = "cache-warm"): string {
		const leaf = this.leafId;
		const id = this.push(
			{ type: "model_usage", purpose, api: "anthropic-messages", provider: PROVIDER, model: MODEL, usage, stopReason: "length" } as never,
			leaf,
			at,
		);
		return id;
	}
	/** /tree navigation: move the leaf to an earlier entry. */
	navigate(id: string): void {
		this.leafId = id;
	}
}

const warmUsage = (cacheRead: unknown, cacheWrite: unknown = 0) => ({
	input: 3,
	output: 1,
	cacheRead,
	cacheWrite,
	totalTokens: 4,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

const MINUTE = 60_000;

describe("round 2: warming-aware cold gap through the real hook (omp entry shapes)", () => {
	/** Main stream up to history(2); returns the manager with the leaf on the third assistant. */
	async function warmedSession(h: Harness): Promise<OmpSessionManager> {
		const manager = new OmpSessionManager(h.sessionDir);
		for (let steps = 1; steps <= 2; steps += 1) {
			manager.appendMessage(assistant(steps * 1_000));
			manager.appendMessage(result(steps) as AgentMessage);
			await h.send(history(steps), { sessionManager: manager as never });
		}
		manager.appendMessage(assistant(3_000));
		return manager;
	}

	it("an omp 'cache-warm' refresh 2 min before a 20 min-idle request prevents the cold-gap flush", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		manager.appendModelUsage(3_000 + 18 * MINUTE, warmUsage(40_000));
		manager.appendMessage(user("next"));
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(isPlaceholder(out[resultIndex(1)])).toBe(false);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it("control: without the refresh the same request flushes with cold-gap", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		manager.appendMessage(user("next"));
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
		expect((await h.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []))).toEqual(["cold-gap"]);
	});

	it("a refresh left on an abandoned branch (after /tree navigation) does not count", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		const branchPoint = manager.leafId as string;
		manager.appendMessage(user("abandoned"));
		manager.appendModelUsage(3_000 + 18 * MINUTE, warmUsage(40_000));
		manager.navigate(branchPoint);
		manager.appendMessage(user("other branch"));
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
	});

	it("a refresh older than the last assistant in event.messages cannot lengthen the gap (session persistence lag)", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		// The branch still ends at an old refresh, but the in-memory context already has a newer assistant.
		manager.appendModelUsage(3_500, warmUsage(40_000));
		// 299.8 s after the last assistant (no cold gap), 300.3 s after the refresh (would be cold if used).
		h.clock.now = 4_000 + 300_000 - 200;
		const out = await h.pi.emitContext(history(4), h.context({ sessionManager: manager as never }));
		expect([1, 2].map((step) => isPlaceholder(out[resultIndex(step)]))).toEqual([false, false]);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it.each([
		["usage cacheRead NaN", { usage: warmUsage(Number.NaN) }],
		["usage cacheRead string", { usage: warmUsage("40000") }],
		["usage missing", { usage: undefined }],
		["usage null", { usage: null }],
		["timestamp not a date", { timestamp: "yesterday" }],
		["timestamp as number", { timestamp: 3_000 + 18 * MINUTE }],
		["purpose not a string", { purpose: 7 }],
	])("malformed warm entry (%s) never throws and never suppresses a cold-gap flush wrongly", async (_label, patch) => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		const id = manager.appendModelUsage(3_000 + 18 * MINUTE, warmUsage(40_000));
		Object.assign(manager.entries.get(id) as OmpEntry, patch);
		manager.appendMessage(user("next"));
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		// A string "40000" is coerced by Number(); only that variant legitimately counts as a warm refresh.
		const counted = _label === "usage cacheRead string";
		expect(isPlaceholder(out[resultIndex(1)])).toBe(!counted);
		expect(errors).toEqual([]);
	});

	it.each([
		["getLeafEntry returns a string", (m: OmpSessionManager) => Object.assign(m, { getLeafEntry: () => "leaf" })],
		["getLeafEntry returns a number", (m: OmpSessionManager) => Object.assign(m, { getLeafEntry: () => 42 })],
		[
			"getEntry throws",
			(m: OmpSessionManager) =>
				Object.assign(m, {
					getEntry: () => {
						throw new Error("entry store closed");
					},
				}),
		],
		[
			"cyclic parent chain",
			(m: OmpSessionManager) => {
				const leaf = m.getLeafEntry() as OmpEntry;
				leaf.parentId = leaf.id;
				return m;
			},
		],
		["getEntry is not a function", (m: OmpSessionManager) => Object.assign(m, { getEntry: 1 })],
		[
			"getLeafEntry is a throwing getter",
			(m: OmpSessionManager) =>
				Object.defineProperty(m, "getLeafEntry", {
					get() {
						throw new Error("disposed");
					},
				}),
		],
	])("hostile session manager (%s): no throw, falls back to the assistant-based gap", async (_label, mutate) => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		manager.appendMessage(user("next"));
		mutate(manager);
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
	});

	it("a long run of non-assistant entries after the last assistant is scanned in bounded time", async () => {
		const h = await harness({ batchThresholdTokens: 100_000 });
		const manager = await warmedSession(h);
		manager.appendModelUsage(3_000 + 18 * MINUTE, warmUsage(40_000));
		for (let index = 0; index < 20_000; index += 1) manager.appendModelUsage(3_000 + 18 * MINUTE, warmUsage(0), "auto-thinking");
		h.clock.now = 3_000 + 20 * MINUTE;
		const started = performance.now();
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(performance.now() - started).toBeLessThan(2_000);
		// Beyond the 256-entry scan window the refresh is not found: the assistant gap applies (documents the bound).
		expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
	});

	it("documents: a refresh stamped in the future (clock skew) yields a negative gap and no cold-gap flush", async () => {
		const h = await harness({ batchThresholdTokens: 100_000, prefixDiagnostics: true });
		const manager = await warmedSession(h);
		manager.appendModelUsage(3_000 + 60 * MINUTE, warmUsage(40_000));
		h.clock.now = 3_000 + 20 * MINUTE;
		const out = await h.pi.emitContext(history(3), h.context({ sessionManager: manager as never }));
		expect(isPlaceholder(out[resultIndex(1)])).toBe(false);
		const rows = (await readFile(join(h.packDir, "prefix-ledger.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
		expect(rows.at(-1).gapMs).toBeLessThan(0);
	});
});

/** An assistant as the host records it after an abort or error: text-only (or empty) content and zero usage. */
function interruptedAssistant(stopReason: "aborted" | "error", content: unknown[], timestamp: number): AgentMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: PROVIDER,
		model: MODEL,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: {} },
		stopReason,
		timestamp,
	} as unknown as AgentMessage;
}

function developer(text: string, timestamp: number): AgentMessage {
	return { role: "developer", content: [{ type: "text", text }], attribution: "agent", timestamp } as unknown as AgentMessage;
}

describe("round 2: omp side-turn detection false positives on the main stream", () => {
	// omp appends developer messages to the MAIN stream (todo reminder, empty/unexpected-stop retry,
	// checkpoint reminder, plan-mode reminder, synthetic prompts). If the next provider response is
	// aborted (Esc) before any usage arrives (OpenAI Responses reports usage only at completion) or
	// fails with an error, the host keeps a text-only/empty, zero-usage assistant. A user prompt then
	// yields [..., developer, assistant(zero usage), user], which sideTurnStart classifies as a /btw
	// side turn. Real omp data: 4 of 54,318 user prompts in ~/.omp/agent/sessions match this shape
	// (developer -> aborted x3 / error x1 -> user).
	it.each([
		["aborted with partial text", interruptedAssistant("aborted", [{ type: "text", text: "Let me" }], 2_500)],
		["error with empty content", interruptedAssistant("error", [], 2_500)],
	])(
		"a main request after developer + %s is still treated as a request (counted send, threshold flush)",
		async (_label, interrupted) => {
			const h = await harness({ batchThresholdTokens: 1, prefixDiagnostics: true });
			await sendSteps(h, 1, 2); // r1 sent twice (a2 + ...), r2 once
			// Main stream: todo reminder -> response interrupted (r1's 3rd provider request) -> user types.
			const messages = [...history(2), developer("<system-reminder>You stopped with 1 incomplete todo</system-reminder>", 2_400), interrupted, user("never mind, do X")];
			const out = await h.send(messages);
			// r1 has been part of a2's and the interrupted request: 3rd send, eligible, threshold 1 -> placeholder.
			expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
			const rows = (await readFile(join(h.packDir, "prefix-ledger.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
			expect(rows.at(-1)).toMatchObject({ continuation: true, request: 4 });
		},
	);

	it("legacy T=0 (no side-turn detection) swaps r1 on that same request (reference)", async () => {
		const h = await harness({ batchThresholdTokens: 0 });
		await sendSteps(h, 1, 2);
		const out = await h.send([...history(2), developer("reminder", 2_400), interruptedAssistant("aborted", [{ type: "text", text: "Let me" }], 2_500), user("x")]);
		expect(isPlaceholder(out[resultIndex(1)])).toBe(true);
	});

	it("the misclassification lasts one request: the next main request swaps r1 (bounded impact)", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 2);
		const base = [...history(2), developer("reminder", 2_400), interruptedAssistant("aborted", [{ type: "text", text: "Let me" }], 2_500), user("x")];
		await h.send(base);
		const next = await h.send([...base, assistant(3_000), result(3)]);
		expect(isPlaceholder(next[resultIndex(1)])).toBe(true);
	});
});

describe("round 2: side turns, robustness", () => {
	it("repeated /btw follow-ups (k = 1..5 answers) between main requests never swap early and never cause a prefix flush", async () => {
		const h = await harness({ batchThresholdTokens: 100_000, prefixDiagnostics: true });
		await sendSteps(h, 1, 2);
		for (let steps = 2; steps <= 6; steps += 1) {
			for (let k = 1; k <= 5; k += 1) await h.send(btwContext(history(steps), k, steps * 1_000 + 500));
			const main = await h.send(history(steps + 1));
			// Nothing swaps: below threshold, no cold gap, no model change, no prefix change.
			expect(main.filter((message) => message.role === "toolResult" && isPlaceholder(message))).toEqual([]);
		}
		const rows = (await readFile(join(h.packDir, "prefix-ledger.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
		expect(rows.filter((row) => row.continuation === true && row.firstChangedIndex !== null)).toEqual([]);
		expect(rows.filter((row) => row.continuation === false).length).toBe(25);
		expect((await h.ledger()).filter((row) => row.flushReason)).toEqual([]);
	});

	it("a /btw with answers as the FIRST call in a fresh process neither flushes nor toasts; the next main request flushes process-start", async () => {
		const dir = await sessionRoot();
		const first = await harness({ batchThresholdTokens: 100_000 }, dir);
		await sendSteps(first, 1, 3);
		const h = await harness({ batchThresholdTokens: 100_000 }, dir);
		const toasts = vi.fn();
		const ui = { ui: { notify: toasts, setStatus: toasts, setWidget: toasts } as never, hasUI: true };
		const side = await h.send(btwContext(history(4), 2, 4_500), ui);
		expect(side.filter((message) => message.role === "toolResult" && isPlaceholder(message))).toEqual([]);
		const main = await h.send(history(5), ui);
		expect([1, 2, 3].map((step) => isPlaceholder(main[resultIndex(step)]))).toEqual([true, true, true]);
		const reasons = (await h.ledger()).flatMap((row) => (row.flushReason ? [row.flushReason] : []));
		expect(reasons.filter((reason) => reason === "process-start").length).toBe(3);
	});

	it("side-turn detection over hostile message arrays (null entries, non-array content, null usage) does not throw", async () => {
		const errors = silenceFailOpen();
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 3);
		const shapes: unknown[][] = [
			[...history(3), null, developer("d", 1), user("q"), assistant(1), user("q2")],
			[...history(3), developer("d", 1), user("q"), { role: "assistant", content: "plain string", usage: null }, user("q2")],
			[...history(3), developer("d", 1), user("q"), { role: "assistant", content: [null], usage: { input: "0" } }, user("q2")],
			[...history(3), developer("d", 1), user("q"), { role: "assistant", content: [], usage: { input: Number.NaN } }, user("q2")],
			[developer("d", 1), user("q"), assistant(1), user("q2")],
		];
		for (const messages of shapes) {
			const out = await h.pi.emitContext(messages as AgentMessage[], h.context());
			expect(out.length).toBe(messages.length);
		}
		expect(errors.every((line) => line.includes("[observationpack] fail-open"))).toBe(true);
	});

	it("documents: a /btw while idle logs a deferred full row under the NEXT main request number, which then also gets the placeholder row", async () => {
		const h = await harness({ batchThresholdTokens: 1 });
		await sendSteps(h, 1, 2); // requests 2 and 3; request 3's answer is a final assistant
		const idle = [...history(2), assistant(2_500, "final answer")];
		await h.send(btwContext(idle, 1, 2_600)); // side turn: request number 4 (3 main assistants + 1)
		const main = await h.send([...idle, user("next task")]); // main request 4
		expect(isPlaceholder(main[resultIndex(1)])).toBe(true);
		const rows = await h.ledger();
		const r1Id = rows.find((row) => row.event === "full")?.id;
		const r1 = rows.filter((row) => row.request === 4 && row.id === r1Id);
		// The same (id, request 4) carries a deferred full row (side turn) and the placeholder row (main request).
		expect(r1.map((row) => [row.event, row.deferred ?? false])).toEqual([
			["full", true],
			["placeholder", false],
		]);
	});
});
