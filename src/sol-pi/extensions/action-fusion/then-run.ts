/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import {
	type BashToolOptions,
	createBashToolDefinition,
	type ExtensionToolContext,
} from "@earendil-works/pi-coding-agent";
import { isHostSchemeTarget, isMissingPathError, withFusedFileQueues } from "./file-queue.ts";

export const THEN_RUN_SUCCEEDED = "[then_run:succeeded]";
export const THEN_RUN_FAILED = "[then_run:failed]";
export const THEN_RUN_SKIPPED = "[then_run:skipped]";

export interface ThenRunInput {
	command: string;
	timeout?: number;
}

/**
 * The same `then_run` parameter as plain JSON Schema, for hosts whose tool
 * schemas are not TypeBox. Kept literal on purpose: the host compiles it with
 * its own TypeBox, which rejects a TypeBox value built by this package.
 */
export function thenRunJsonSchema(description: string): Record<string, unknown> {
	return {
		type: "object",
		description,
		properties: {
			command: { type: "string", description: "Bash command to run" },
			timeout: { type: "number", description: "Timeout in seconds (optional, no default timeout)" },
		},
		required: ["command"],
	};
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function resultText(result: AgentToolResult<unknown>): string {
	return result.content
		.filter((block) => block.type === "text")
		.map((block) => block.text)
		.join("\n");
}

/** Hosts report a failed tool call either by throwing or by returning `isError: true`. */
function isErrorResult(result: AgentToolResult<unknown>): boolean {
	return (result as { isError?: unknown }).isError === true;
}

function thenRunSkippedError(error: unknown): Error {
	return new Error(
		`${errorText(error)}\n\n${THEN_RUN_SKIPPED} The file mutation did not complete successfully; the command was not run.`,
	);
}

const ABSENT = "absent";

/** Content digest of a target, or `absent` once a delete or move has removed it. */
async function targetState(path: string): Promise<string> {
	try {
		return createHash("sha256").update(await readFile(path)).digest("hex");
	} catch (error) {
		if (isMissingPathError(error)) return ABSENT;
		throw error;
	}
}

/**
 * Prove nothing else rewrote the mutation's targets before the command runs.
 * A deleted or moved-away target must stay absent; a target the host resolves
 * through its own URL scheme is not on the filesystem and is left to the host.
 */
export async function assertUnchangedBeforeCommand(
	path: string | readonly string[],
	yieldForInterference: () => Promise<void> = () => new Promise<void>((resolve) => setImmediate(resolve)),
): Promise<void> {
	const paths = (typeof path === "string" ? [path] : path).filter((target) => !isHostSchemeTarget(target));
	try {
		const mutationStates = await Promise.all(paths.map(targetState));
		await yieldForInterference();
		const commandStates = await Promise.all(paths.map(targetState));
		if (mutationStates.some((state, index) => state !== commandStates[index])) {
			throw new Error("target content changed after the fused mutation");
		}
	} catch (error) {
		throw new Error(`${THEN_RUN_SKIPPED} ${errorText(error)}; the command was not run.`);
	}
}

/**
 * Run the follow-up command. A host that exposes `executeTool` runs it as a
 * nested `bash` call, so its own blocking hooks apply; explicit `bashOptions`
 * (tests, embedded runtimes) keep the direct definition.
 */
async function runCommand(
	toolCallId: string,
	thenRun: ThenRunInput,
	bashOptions: BashToolOptions | undefined,
	signal: AbortSignal | undefined,
	ctx: ExtensionToolContext,
): Promise<AgentToolResult<unknown>> {
	const executeTool = (ctx as Partial<ExtensionToolContext>).executeTool;
	if (bashOptions === undefined && typeof executeTool === "function") {
		const outcome = await ctx.executeTool("bash", thenRun, { signal });
		if (outcome.isError) throw new Error(resultText(outcome.result));
		return outcome.result;
	}
	const bash = createBashToolDefinition(ctx.cwd, bashOptions);
	return bash.execute(`${toolCallId}:then_run`, thenRun, signal, undefined, ctx);
}

/**
 * Apply a file mutation and, when the model asked for one, run its follow-up
 * command before returning a single observation.
 *
 * Both steps hold the SoL-Pi queue slot of every target, so another fused
 * mutation of any of them cannot interleave. Pi's built-in mutation tool keeps
 * its own queue; the two queues are not nested.
 */
export async function executeMutationThenRun<TDetails>({
	toolCallId,
	absolutePath,
	thenRun,
	mutate,
	bashOptions,
	signal,
	ctx,
}: {
	toolCallId: string;
	absolutePath: string | readonly string[];
	thenRun: ThenRunInput | undefined;
	mutate: () => Promise<AgentToolResult<TDetails>>;
	bashOptions: BashToolOptions | undefined;
	signal: AbortSignal | undefined;
	ctx: ExtensionToolContext;
}): Promise<AgentToolResult<TDetails>> {
	const targets = typeof absolutePath === "string" ? [absolutePath] : absolutePath;
	if (targets.length === 0) throw new Error("action-fusion: no mutation target");
	return withFusedFileQueues(targets, async () => {
		let mutationResult: AgentToolResult<TDetails>;
		try {
			mutationResult = await mutate();
		} catch (error) {
			if (thenRun !== undefined) {
				throw thenRunSkippedError(error);
			}
			throw error;
		}

		if (thenRun === undefined) {
			return mutationResult;
		}

		if (isErrorResult(mutationResult)) {
			return {
				...mutationResult,
				content: [
					...mutationResult.content,
					{
						type: "text",
						text: `${THEN_RUN_SKIPPED} The file mutation returned an error; the command was not run.`,
					},
				],
			};
		}

		try {
			await assertUnchangedBeforeCommand(targets);
		} catch (error) {
			throw new Error([resultText(mutationResult), errorText(error)].filter(Boolean).join("\n\n"));
		}
		try {
			const bashResult = await runCommand(toolCallId, thenRun, bashOptions, signal, ctx);
			const output = resultText(bashResult);
			return {
				...mutationResult,
				content: [
					...mutationResult.content,
					{ type: "text", text: output ? `${THEN_RUN_SUCCEEDED}\n${output}` : THEN_RUN_SUCCEEDED },
				],
			};
		} catch (error) {
			const mutationOutput = resultText(mutationResult);
			throw new Error([mutationOutput, THEN_RUN_FAILED, errorText(error)].filter(Boolean).join("\n\n"));
		}
	});
}
