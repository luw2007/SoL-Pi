/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createActionFusionExtension } from "../src/sol-pi/extensions/action-fusion/index.ts";
import { fakeToolContext } from "./helpers.ts";

const execution = vi.hoisted(() => ({ edit: vi.fn(), write: vi.fn() }));
vi.mock("@earendil-works/pi-coding-agent", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@earendil-works/pi-coding-agent")>();
	return {
		...actual,
		createEditToolDefinition: (...args: Parameters<typeof actual.createEditToolDefinition>) => {
			const tool = actual.createEditToolDefinition(...args);
			return { ...tool, execute: (...input: Parameters<typeof tool.execute>) => {
				execution.edit(...input);
				return tool.execute(...input);
			} };
		},
		createWriteToolDefinition: (...args: Parameters<typeof actual.createWriteToolDefinition>) => {
			const tool = actual.createWriteToolDefinition(...args);
			return { ...tool, execute: (...input: Parameters<typeof tool.execute>) => {
				execution.write(...input);
				return tool.execute(...input);
			} };
		},
	};
});

const roots: string[] = [];
afterEach(async () => {
	vi.clearAllMocks();
	await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture(name: "edit" | "write") {
	const cwd = await mkdtemp(join(tmpdir(), "fusion-delegation-"));
	roots.push(cwd);
	await writeFile(join(cwd, "target.txt"), "before\n");
	const tools = new Map<string, ToolDefinition>();
	createActionFusionExtension()({ registerTool: (tool: ToolDefinition) => tools.set(tool.name, tool) } as unknown as ExtensionAPI);
	const input = name === "edit"
		? { path: "target.txt", edits: [{ oldText: "before", newText: "after" }] }
		: { path: "target.txt", content: "after\n" };
	return { cwd, tool: tools.get(name)!, input };
}

describe.each(["edit", "write"] as const)("Action Fusion %s delegation", (name) => {
	it("forwards signal and streaming updates to omp without calling the Pi fallback", async () => {
		const { cwd, tool, input } = await fixture(name);
		const controller = new AbortController();
		const update = { content: [{ type: "text" as const, text: "progress" }], details: undefined };
		const onUpdate = vi.fn();
		const invokeTool = vi.fn(async (_input, options) => {
			expect(options.signal).toBe(controller.signal);
			options.onUpdate(update);
			return update;
		});
		const ctx = fakeToolContext(cwd, { cwd, invokeTool });
		const result = await tool.execute("delegated", input, controller.signal, onUpdate, ctx);
		expect(invokeTool).toHaveBeenCalledWith(input, { signal: controller.signal, onUpdate });
		expect(onUpdate).toHaveBeenCalledWith(update);
		expect(result).toBe(update);
		expect(execution[name]).not.toHaveBeenCalled();
	});

	it("propagates cancellation during omp mutation and skips then_run", async () => {
		const { cwd, tool, input } = await fixture(name);
		const controller = new AbortController();
		const invokeTool = vi.fn(async (_input, options) => {
			return await new Promise<never>((_resolve, reject) => {
				options.signal.addEventListener("abort", () => reject(new Error("mutation aborted")), { once: true });
				controller.abort();
			});
		});
		await expect(tool.execute("cancelled", { ...input, then_run: { command: "touch should-not-run" } },
			controller.signal, undefined, fakeToolContext(cwd, { cwd, invokeTool }))).rejects.toThrow("mutation aborted");
		await expect(access(join(cwd, "should-not-run"))).rejects.toMatchObject({ code: "ENOENT" });
		expect(invokeTool.mock.calls[0]?.[0]).toEqual(input);
		expect(execution[name]).not.toHaveBeenCalled();
	});

	it("honors a cancelled signal on Pi without running the follow-up command", async () => {
		const { cwd, tool, input } = await fixture(name);
		const controller = new AbortController();
		controller.abort();
		await expect(tool.execute("pi-cancelled", { ...input, then_run: { command: "touch should-not-run" } },
			controller.signal, undefined, fakeToolContext(cwd, { cwd }))).rejects.toThrow(/abort/i);
		await expect(access(join(cwd, "should-not-run"))).rejects.toMatchObject({ code: "ENOENT" });
	});

	it("keeps Pi's built-in execution with the original signal, callback, id and context", async () => {
		const { cwd, tool, input } = await fixture(name);
		const signal = new AbortController().signal;
		const onUpdate = vi.fn();
		const ctx = fakeToolContext(cwd, { cwd });
		expect("invokeTool" in ctx).toBe(false);
		await tool.execute("pi-call", input, signal, onUpdate, ctx);
		expect(execution[name]).toHaveBeenCalledWith("pi-call", input, signal, onUpdate, ctx);
	});
});
