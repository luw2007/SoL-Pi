/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { mkdtemp, rm, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { BashOperations, ExtensionAPI, ExtensionToolContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import {
	assertUnchangedBeforeCommand,
	createActionFusionExtension,
	fusedApproval,
	fusedApprovalHooks,
	liveToolSchema,
	mutationTargets,
	THEN_RUN_SKIPPED,
} from "../src/sol-pi/extensions/action-fusion/index.ts";
import {
	isHostSchemeTarget,
	resolveToolPath,
	withFusedFileQueue,
	withFusedFileQueues,
} from "../src/sol-pi/extensions/action-fusion/file-queue.ts";

type ToolResult = { content: Array<{ type: string; text?: string }>; details: unknown; isError?: boolean };

function text(result: ToolResult): string {
	return result.content
		.filter((block) => block.type === "text")
		.map((block) => block.text ?? "")
		.join("\n");
}

function delay(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve!: () => void;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

function loadEdit(bashOperations?: BashOperations): ToolDefinition {
	const registered = new Map<string, ToolDefinition>();
	const pi = { registerTool: (tool: ToolDefinition) => registered.set(tool.name, tool) } as unknown as ExtensionAPI;
	createActionFusionExtension(bashOperations ? { bashOptions: { operations: bashOperations } } : {})(pi);
	const edit = registered.get("edit");
	if (!edit) throw new Error("action fusion did not register edit");
	return edit;
}

function hostContext(cwd: string, host: Record<string, unknown>): ExtensionToolContext {
	return {
		mode: "json",
		hasUI: false,
		cwd,
		model: undefined,
		sessionManager: { getSessionFile: () => undefined, getSessionId: () => "host-guards-test" },
		ui: {},
		...host,
	} as unknown as ExtensionToolContext;
}

function countingBash(): { operations: BashOperations; calls: () => number } {
	let calls = 0;
	return {
		operations: {
			exec: async () => {
				calls++;
				return { exitCode: 0 };
			},
		},
		calls: () => calls,
	};
}

const tempDirs: string[] = [];

async function createTempDir(): Promise<string> {
	const dir = await mkdtemp(join(tmpdir(), "pi-host-guards-"));
	tempDirs.push(dir);
	return dir;
}

afterEach(async () => {
	await Promise.all(tempDirs.splice(0, tempDirs.length).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("host-owned mutation", () => {
	it("runs the edit through the host's own tool, without then_run, when the host offers delegation", async () => {
		const dir = await createTempDir();
		await writeFile(join(dir, "target.txt"), "before\n");
		const bash = countingBash();
		const invoked: Record<string, unknown>[] = [];
		const ctx = hostContext(dir, {
			invokeTool: async (params: Record<string, unknown>) => {
				invoked.push(params);
				return { content: [{ type: "text", text: "native edit applied" }], details: undefined };
			},
		});

		const result = (await loadEdit(bash.operations).execute(
			"edit-native",
			{ path: "target.txt", edits: [{ oldText: "before", newText: "after" }], then_run: { command: "true" } },
			undefined,
			undefined,
			ctx,
		)) as ToolResult;

		expect(invoked).toEqual([{ path: "target.txt", edits: [{ oldText: "before", newText: "after" }] }]);
		expect(text(result)).toContain("native edit applied");
		expect(text(result)).toContain("[then_run:succeeded]");
		expect(bash.calls()).toBe(1);
	});

	it("does not run then_run when the mutation returns an error instead of throwing", async () => {
		const dir = await createTempDir();
		const bash = countingBash();
		const ctx = hostContext(dir, {
			invokeTool: async () => ({
				content: [{ type: "text", text: "hash mismatch for target.txt" }],
				details: undefined,
				isError: true,
			}),
		});

		const result = (await loadEdit(bash.operations).execute(
			"edit-returned-error",
			{ path: "target.txt", edits: [{ oldText: "a", newText: "b" }], then_run: { command: "must not run" } },
			undefined,
			undefined,
			ctx,
		)) as ToolResult;

		expect(bash.calls()).toBe(0);
		expect(result.isError).toBe(true);
		expect(text(result)).toContain("hash mismatch for target.txt");
		expect(text(result)).toContain(THEN_RUN_SKIPPED);
	});

	it("runs the command as a nested host call and reports a blocked call as a failed then_run", async () => {
		const dir = await createTempDir();
		await writeFile(join(dir, "target.txt"), "x\n");
		const nested: Array<{ name: string; args: unknown }> = [];
		const ctx = hostContext(dir, {
			invokeTool: async () => ({ content: [{ type: "text", text: "native edit applied" }], details: undefined }),
			executeTool: async (name: string, args: unknown) => {
				nested.push({ name, args });
				return { isError: true, result: { content: [{ type: "text", text: "blocked by host policy" }], details: undefined } };
			},
		});

		await expect(
			loadEdit().execute(
				"edit-nested-bash",
				{ path: "target.txt", edits: [{ oldText: "x", newText: "y" }], then_run: { command: "rm -rf build" } },
				undefined,
				undefined,
				ctx,
			),
		).rejects.toThrow(/\[then_run:failed\][\s\S]*blocked by host policy/);
		expect(nested).toEqual([{ name: "bash", args: { command: "rm -rf build" } }]);
	});
});

describe("approval of a fused call", () => {
	const native = (args: unknown) => ((args as { path?: string }).path === "locked.txt" ? { tier: "write", policy: "deny" } : "write");

	it("keeps the built-in decision for a call without then_run", () => {
		expect(fusedApproval(native)?.({ path: "a.txt" })).toBe("write");
	});

	it("classifies a call with then_run as command execution under the host's bash policy", () => {
		expect(fusedApproval(native)?.({ path: "a.txt", then_run: { command: "npm test" } })).toEqual({
			tier: "exec",
			policyKey: "bash",
			reason: "then_run: npm test",
		});
	});

	it("never relaxes a built-in deny, and adds nothing for a host without approvals", () => {
		expect(fusedApproval(native)?.({ path: "locked.txt", then_run: { command: "true" } })).toEqual({
			tier: "write",
			policy: "deny",
		});
		expect(fusedApproval(undefined)).toBeUndefined();
	});

	it("keeps a must-prompt signal from the built-in tool", () => {
		const mustPrompt = () => ({ tier: "write", override: true });
		expect(fusedApproval(mustPrompt)?.({ path: "a.txt", then_run: { command: "true" } })).toEqual({
			tier: "exec",
			policyKey: "bash",
			reason: "then_run: true",
			override: true,
		});
	});

	const template = {
		approval: () => "write",
		formatApprovalDetails: (args: unknown) => [`File: ${(args as { path: string }).path}`],
		approvalCacheTargets: (args: unknown) => [(args as { path: string }).path],
	};

	it("shows the command in the approval prompt", () => {
		const hooks = fusedApprovalHooks(template);
		expect(hooks.formatApprovalDetails?.({ path: "a.txt" })).toEqual(["File: a.txt"]);
		expect(hooks.formatApprovalDetails?.({ path: "a.txt", then_run: { command: "rm -rf build" } })).toEqual([
			"File: a.txt",
			"Command: rm -rf build",
		]);
	});

	it("passes the host's cache targets through, so restricted folders keep their subject", () => {
		const hooks = fusedApprovalHooks(template);
		expect(hooks.approvalCacheTargets?.({ path: "a.txt", then_run: { command: "npm test" } })).toEqual(["a.txt"]);
		expect(fusedApprovalHooks({ approval: () => "write" })).not.toHaveProperty("approvalCacheTargets");
	});
});

describe("patch-string targets", () => {
	it("unquotes header paths, accepts untagged headers, and adds move destinations", () => {
		const payload = [
			'["dir/a b.ts"#1A2B]',
			"PUT 1.=1:",
			"+x",
			"[new-file.ts]",
			"PUT <1:",
			"+y",
			"[src/old.ts#0F0F]",
			'MV "src/new name.ts"',
			"[gone.ts#ABCD]",
			"REM",
		].join("\n");
		expect(mutationTargets({ input: payload })).toEqual([
			"dir/a b.ts",
			"new-file.ts",
			"src/old.ts",
			"src/new name.ts",
			"gone.ts",
		]);
	});
});

describe("pre-command check", () => {
	it("accepts a deleted target that stays absent and rejects one that reappears", async () => {
		const dir = await createTempDir();
		const deleted = join(dir, "deleted.txt");

		await expect(assertUnchangedBeforeCommand(deleted, async () => {})).resolves.toBeUndefined();
		await expect(
			assertUnchangedBeforeCommand(deleted, async () => {
				await writeFile(deleted, "recreated\n");
			}),
		).rejects.toThrow("target content changed after the fused mutation");
		await unlink(deleted);
	});

	it("leaves a host-scheme target to the host instead of reading it from disk", async () => {
		await expect(assertUnchangedBeforeCommand(["local://plan.md"], async () => {})).resolves.toBeUndefined();
	});
});

describe("locking every target", () => {
	it("holds a later target too, so a fused mutation of it waits", async () => {
		const dir = await createTempDir();
		const first = join(dir, "first.txt");
		const second = join(dir, "second.txt");
		const events: string[] = [];
		const started = deferred();
		const release = deferred();

		const both = withFusedFileQueues([first, second], async () => {
			events.push("both:start");
			started.resolve();
			await release.promise;
			events.push("both:end");
		});
		await started.promise;
		const later = withFusedFileQueue(second, async () => {
			events.push("second");
		});
		// Asserting that `later` has NOT run needs a bounded wait; there is no signal for an absence.
		await delay(20);
		expect(events).toEqual(["both:start"]);
		release.resolve();
		await Promise.all([both, later]);
		expect(events).toEqual(["both:start", "both:end", "second"]);
	});

	it("takes shared targets in one order, so opposite argument orders cannot deadlock", async () => {
		const dir = await createTempDir();
		const a = join(dir, "a.txt");
		const b = join(dir, "b.txt");
		const done: string[] = [];
		const gate = deferred();

		// Opposite argument orders: a lock taken in argument order would deadlock and time the test out.
		const ab = withFusedFileQueues([a, b], async () => {
			await gate.promise;
			done.push("ab");
		});
		const ba = withFusedFileQueues([b, a], async () => {
			done.push("ba");
		});
		gate.resolve();
		await Promise.all([ab, ba]);
		expect(done.sort()).toEqual(["ab", "ba"]);
	});
});

describe("live host schema", () => {
	it("reads the session's current edit schema and drops a then_run the extension added earlier", () => {
		const pi = {
			getAllTools: () => [
				{
					name: "edit",
					parameters: {
						toJsonSchema: () => ({
							type: "object",
							properties: { path: {}, old_string: {}, new_string: {}, replace_all: {}, then_run: {} },
							required: ["path", "old_string", "new_string"],
						}),
					},
				},
			],
		} as unknown as ExtensionAPI;

		expect(liveToolSchema(pi, "edit")).toEqual({
			properties: { path: {}, old_string: {}, new_string: {}, replace_all: {} },
			required: ["path", "old_string", "new_string"],
		});
	});

	it("reports nothing when the host lists no tools or throws while listing", () => {
		expect(liveToolSchema({} as ExtensionAPI, "edit")).toBeUndefined();
		const throwing = {
			getAllTools: () => {
				throw new Error("runtime not initialized");
			},
		} as unknown as ExtensionAPI;
		expect(liveToolSchema(throwing, "edit")).toBeUndefined();
	});
});

describe("target resolution matches the host", () => {
	it("keeps an @ that is part of the file name but strips the host's path prefix", () => {
		// The host strips `@` only before a path or a URL; `@config.json` is a real
		// file name, so the guard must hash that same file.
		expect(isHostSchemeTarget("@config.json")).toBe(false);
		expect(resolveToolPath("/w", "@config.json")).toBe(resolve("/w", "@config.json"));
		expect(resolveToolPath("/w", "@/notes.md")).toBe(resolve("/notes.md"));
	});

	it("treats a drive-letter path as a file and another user's home as the host's", () => {
		expect(isHostSchemeTarget("C://x/y.txt")).toBe(false);
		expect(isHostSchemeTarget("local://plan.md")).toBe(true);
		expect(isHostSchemeTarget("~alice/notes.md")).toBe(true);
	});
});
