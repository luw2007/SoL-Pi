/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
/**
 * Action Fusion - fuse a file mutation and its follow-up command into one turn.
 *
 * Base pi rollouts repeatedly showed the same pair of turns: edit or write a
 * file, then run a command to test, build, or start it. This extension replaces
 * the built-in `edit` and `write` tools with versions that take an optional
 * `then_run` object, apply the mutation, run the command, and return one
 * combined observation. The model decision between the two turns disappears.
 *
 * Everything else about `edit` and `write` is inherited from the built-in
 * definitions: their schemas, prompt text, argument shims, and renderers.
 *
 * This standalone version composes only Pi's public tool definitions.
 */

import {
	type BashToolOptions,
	createEditToolDefinition,
	createWriteToolDefinition,
	type EditToolDetails,
	type EditToolOptions,
	type ExtensionAPI,
	type ExtensionContext,
	type ExtensionFactory,
	type WriteToolOptions,
} from "@earendil-works/pi-coding-agent";
import { Type, type TSchema } from "typebox";
import { resolveToolPath } from "./file-queue.ts";
import { isNestedToolCallId, renderSolPiTool, showSolPiSavings } from "../../tui.ts";
import { createThenRunSchema, runFusedMutation, type ThenRunInput, type ThenRunStatus } from "./then-run.ts";

const EDIT_THEN_RUN_DESCRIPTION =
	"Command to run next on this file after the edit succeeds — e.g. run, build, start/restart, install, or check it; optional timeout in seconds. Skipped if the edit fails; a non-zero exit is reported but keeps the edit.";
const WRITE_THEN_RUN_DESCRIPTION =
	"Command to run next on this file after the write succeeds — e.g. run, build, start/restart, install, or check it; optional timeout in seconds. Skipped if the write fails; a non-zero exit is reported but keeps the write.";

export interface ActionFusionOptions {
	/** Optional programmatic bash overrides, primarily for tests and embedded runtimes. */
	readonly bashOptions?: BashToolOptions;
	/** Overrides for the underlying built-in `edit` tool. */
	readonly editOptions?: EditToolOptions;
	/** Overrides for the underlying built-in `write` tool. */
	readonly writeOptions?: WriteToolOptions;
}

/**
 * Built-in tool definitions capture their cwd in closures, so keep one per
 * working directory instead of rebuilding them on every call and every redraw.
 */
function memoizeByCwd<T>(create: (cwd: string) => T): (cwd: string) => T {
	const cache = new Map<string, T>();
	return (cwd) => {
		const cached = cache.get(cwd);
		if (cached) return cached;
		const created = create(cwd);
		cache.set(cwd, created);
		return created;
	};
}

/**
 * Announce the avoided round-trip only when the command actually succeeded,
 * as reported by the fusion itself, never by matching command output. Calls
 * from a codemode script (`<parent id>/<n>`) save no model round-trip of their
 * own: the script already chains tools inside one model call.
 */
function announceSavings(status: ThenRunStatus, toolCallId: string, ctx: ExtensionContext): void {
	if (status !== "succeeded" || isNestedToolCallId(toolCallId)) return;
	showSolPiSavings(ctx, "Action Fusion", "1 model round-trip avoided");
}

/** The fused argument a renderer receives, when the host typed it as a bare schema. */
function passedThenRun(args: unknown): unknown {
	return typeof args === "object" && args !== null && "then_run" in args ? args.then_run : undefined;
}

/**
 * Show the savings banner on a fused call unless it settled as an error: a
 * failed or skipped command avoided no round-trip. Pi redraws the call row
 * with the settled `isError` as well, so both renderers drop it together.
 */
function showsSavingsBanner(thenRun: unknown, context: { readonly isError?: boolean }): boolean {
	return Boolean(thenRun) && !context.isError;
}

/**
 * The built-in tool's own parameter properties, which the fused tool re-declares
 * alongside `then_run`.
 *
 * Pi publishes them as a TypeBox `Type.Object`, so `parameters.properties` is
 * the whole contract. A Pi-compatible host may describe the same tool with its
 * own schema value instead — potentially callable rather than a plain object,
 * and exposing the contract only through `toJsonSchema()`. Reading `.properties`
 * alone yields `{}` there, which would publish a fused tool carrying `then_run`
 * and nothing else: the model can no longer send `path`, and the fused execute
 * path resolves an undefined target. Rebuild from `toJsonSchema()` in that case.
 */
export function hostToolProperties(parameters: unknown): Record<string, unknown> {
	if (!parameters || (typeof parameters !== "object" && typeof parameters !== "function")) return {};
	if ("properties" in parameters) {
		const properties = parameters.properties as Record<string, unknown>;
		if (properties && Object.keys(properties).length > 0) return properties;
	}
	if (!("toJsonSchema" in parameters) || typeof parameters.toJsonSchema !== "function") return {};
	const json = parameters.toJsonSchema() as {
		properties?: Record<string, unknown>;
		required?: readonly string[];
	};
	const required = new Set(json.required ?? []);
	return Object.fromEntries(
		Object.entries(json.properties ?? {}).map(([key, schema]) => [
			key,
			required.has(key) ? schema : Type.Optional(schema as never),
		]),
	);
}

/**
 * The host's declared contract when it publishes parameters as a callable schema
 * without readable `properties` (omp's Pi shim, for example).
 */
function hostJsonContract(
	parameters: unknown,
): { readonly properties: Record<string, unknown>; readonly required: readonly string[] } | undefined {
	if (!parameters || (typeof parameters !== "object" && typeof parameters !== "function")) return undefined;
	const readable = "properties" in parameters ? parameters.properties : undefined;
	if (readable && typeof readable === "object" && Object.keys(readable).length > 0) return undefined;
	if (!("toJsonSchema" in parameters) || typeof parameters.toJsonSchema !== "function") return undefined;
	// Unchecked cast: the function is the host's own JSON Schema exporter.
	const json = parameters.toJsonSchema() as {
		properties?: Record<string, unknown>;
		required?: readonly string[];
	};
	const properties = json.properties ?? {};
	if (Object.keys(properties).length === 0) return undefined;
	return { properties, required: json.required ?? [] };
}

/** The host's declared parameter names, read without materializing a schema. */
function hostParameterNames(parameters: unknown): ReadonlySet<string> {
	const contract = hostJsonContract(parameters);
	if (contract) return new Set(Object.keys(contract.properties));
	const readable =
		parameters && typeof parameters === "object" && "properties" in parameters ? parameters.properties : undefined;
	return new Set(readable && typeof readable === "object" ? Object.keys(readable) : []);
}

function thenRunJsonSchema(description: string): Record<string, unknown> {
	return {
		type: "object",
		properties: {
			command: { type: "string", description: "Bash command to run" },
			timeout: { type: "number", description: "Timeout in seconds (optional, no default timeout)" },
		},
		required: ["command"],
	};
}

/**
 * Fuse `then_run` into the host's own parameter contract. A reduced host that
 * only exports JSON Schema keeps that shape: its converter rejects a TypeBox
 * `Type.Optional` wrapped around a plain schema fragment.
 */
function fusedToolParameters(parameters: unknown, description: string): TSchema {
	const contract = hostJsonContract(parameters);
	if (contract) {
		// Unchecked cast: the host consumes its own JSON Schema shape, which TypeBox
		// cannot express; the fields follow the built-in tool's declared contract.
		return {
			type: "object",
			properties: { ...contract.properties, then_run: thenRunJsonSchema(description) },
			required: contract.required,
		} as unknown as TSchema;
	}
	return Type.Object({
		...hostToolProperties(parameters),
		then_run: createThenRunSchema(description),
	});
}

export function createActionFusionExtension(options: ActionFusionOptions = {}): ExtensionFactory {
	const baseEdit = memoizeByCwd((cwd: string) => createEditToolDefinition(cwd, options.editOptions));
	const baseWrite = memoizeByCwd((cwd: string) => createWriteToolDefinition(cwd, options.writeOptions));

	return (pi: ExtensionAPI) => {
		const editTemplate = baseEdit(process.cwd());
		const writeTemplate = baseWrite(process.cwd());

		// The host's declared parameter names, without materializing a TypeBox
		// schema: a reduced host rejects `Type.Optional` over its JSON fragments.
		const editDeclaresPath = hostParameterNames(editTemplate.parameters).has("path");
		const writeDeclaresPath = hostParameterNames(writeTemplate.parameters).has("path");

		// Unchecked cast: fused parameters keep the host's own declared shape; the
		// builder only appends the optional `then_run` field.
		const editParameters = fusedToolParameters(editTemplate.parameters, EDIT_THEN_RUN_DESCRIPTION) as typeof editTemplate.parameters;
		const writeParameters = fusedToolParameters(writeTemplate.parameters, WRITE_THEN_RUN_DESCRIPTION) as typeof writeTemplate.parameters;

		// The fused queue and the pre-command hash check are keyed on one target
		// file, so only replace a mutation tool that names its target with `path`.
		// A host whose edit tool takes a different shape (a multi-file patch, say)
		// keeps its built-in tool rather than getting a fused one that cannot
		// resolve a target.
		if (editDeclaresPath) {
			pi.registerTool<typeof editParameters, EditToolDetails | undefined>({
				...editTemplate,
				parameters: editParameters,
				async execute(toolCallId, input, signal, onUpdate, ctx) {
					const { then_run, ...editInput } = input as typeof input & { then_run?: ThenRunInput };
					const { result, status } = await runFusedMutation({
						toolCallId,
						absolutePath: resolveToolPath(ctx.cwd, input.path),
						thenRun: then_run,
						bashOptions: options.bashOptions,
						signal,
						ctx,
						mutate: () => baseEdit(ctx.cwd).execute(toolCallId, editInput, signal, onUpdate, ctx),
					});
					announceSavings(status, toolCallId, ctx);
					return result;
				},
				renderCall: (args, theme, context) => {
					const base = baseEdit(context.cwd).renderCall!(args, theme, context);
					return showsSavingsBanner(passedThenRun(args), context)
						? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
						: base;
				},
				renderResult: (result, resultOptions, theme, context) => {
					const base = baseEdit(context.cwd).renderResult!(result, resultOptions, theme, context);
					return showsSavingsBanner(passedThenRun(context.args), context)
						? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
						: base;
				},
			});
		}

		if (!writeDeclaresPath) return;

		pi.registerTool<typeof writeParameters, undefined>({
			...writeTemplate,
			parameters: writeParameters,
			async execute(toolCallId, input, signal, onUpdate, ctx) {
				const { then_run, ...writeInput } = input as typeof input & { then_run?: ThenRunInput };
				const { result, status } = await runFusedMutation({
					toolCallId,
					absolutePath: resolveToolPath(ctx.cwd, input.path),
					thenRun: then_run,
					bashOptions: options.bashOptions,
					signal,
					ctx,
					mutate: () => baseWrite(ctx.cwd).execute(toolCallId, writeInput, signal, onUpdate, ctx),
				});
				announceSavings(status, toolCallId, ctx);
				return result;
			},
			renderCall: (args, theme, context) => {
				const base = baseWrite(context.cwd).renderCall!(args, theme, context);
				return showsSavingsBanner(passedThenRun(args), context)
					? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
					: base;
			},
			renderResult: (result, resultOptions, theme, context) => {
				const base = baseWrite(context.cwd).renderResult!(result, resultOptions, theme, context);
				return showsSavingsBanner(passedThenRun(context.args), context)
					? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
					: base;
			},
		});
	};
}

export type { ThenRunInput } from "./then-run.ts";
export {
	assertUnchangedBeforeCommand,
	executeMutationThenRun,
	THEN_RUN_FAILED,
	THEN_RUN_SKIPPED,
	THEN_RUN_SUCCEEDED,
} from "./then-run.ts";

export function registerActionFusion(pi: ExtensionAPI): void {
	createActionFusionExtension()(pi);
}

export default registerActionFusion;
