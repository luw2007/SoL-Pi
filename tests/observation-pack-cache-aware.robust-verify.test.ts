/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * Robustness and omp-compat verification for ObservationPack cache-aware batching.
 *
 * Tests marked `it.fails` document confirmed defects: they assert the behaviour
 * the spec requires and currently fail. Once a defect is fixed, vitest reports
 * the `it.fails` as failing, which is the cue to turn it into a plain `it`.
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
