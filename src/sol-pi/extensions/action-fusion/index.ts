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
 * Host guarantees are preserved, not reimplemented:
 * - the mutation runs through the host's own tool (`ctx.invokeTool`) when the
 *   host offers it, so its session state, plan restrictions and active edit
 *   schema apply;
 * - a call carrying `then_run` is classified as command execution scoped to the
 *   host's `bash` policy, so the host's approval gate decides it;
 * - the command runs as a nested `bash` call where the host exposes
 *   `ctx.executeTool`, so its blocking hooks apply.
 *
 * This standalone version composes Pi's public tool definitions. Host extras
 * (`ctx.invokeTool`, `ctx.executeTool`, the `approval` declaration) are used
 * only when detected, so the same build runs on a host that has none of them.
 *
 * Known limits, stated so a caller is not surprised:
 * - the host consults one approval policy per call, so a set `bash` policy
 *   replaces the mutation tool's own policy for calls carrying `then_run`;
 * - where the host offers no nested tool call (OMP 18.13 has none), the command
 *   runs through the public bash definition, so host-only command rules such as
 *   pattern deny lists do not see it;
 * - the fused `edit` schema is read once at `session_start`, so a mode change
 *   later in the session needs a new session to take effect;
 * - on a host without same-name delegation, the mutation runs from the
 *   load-time definition, so that host's session state cannot apply.
 */

import {
	type AgentToolResult,
	type AgentToolUpdateCallback,
	type BashToolOptions,
	createEditToolDefinition,
	createWriteToolDefinition,
	type EditToolDetails,
	type EditToolOptions,
	type ExtensionAPI,
	type ExtensionFactory,
	type ExtensionToolContext,
	type WriteToolOptions,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { isHostSchemeTarget, resolveToolPath } from "./file-queue.ts";
import { renderSolPiTool, showSolPiSavings } from "../../tui.ts";
import {
	executeMutationThenRun,
	THEN_RUN_SKIPPED,
	THEN_RUN_SUCCEEDED,
	type ThenRunInput,
	thenRunJsonSchema,
} from "./then-run.ts";

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

interface DeclaredSchema {
	readonly properties: Record<string, unknown>;
	readonly required: readonly string[];
}

/**
 * A tool's declared parameters as plain JSON Schema.
 *
 * Pi publishes them as a TypeBox `Type.Object`; compatible hosts may expose
 * either a plain JSON Schema object or a callable `toJsonSchema()` value.
 * Undefined when the shape cannot be read, which leaves the built-in tool in
 * place.
 */
function declaredSchema(parameters: unknown): DeclaredSchema | undefined {
	try {
		if (!parameters || (typeof parameters !== "object" && typeof parameters !== "function")) return undefined;
		let json = parameters as { properties?: unknown; required?: unknown; toJsonSchema?: () => unknown };
		if ((!json.properties || (typeof json.properties === "object" && Object.keys(json.properties).length === 0)) && typeof json.toJsonSchema === "function") {
			json = json.toJsonSchema() as typeof json;
		}
		if (!json || typeof json !== "object" || !json.properties || typeof json.properties !== "object" || Array.isArray(json.properties)) return undefined;
		const entries = Object.entries(json.properties);
		if (entries.some(([, schema]) => !schema || typeof schema !== "object" || Array.isArray(schema))) return undefined;
		const required = Array.isArray(json.required)
			? json.required.filter((key): key is string => typeof key === "string")
			: [];
		return { properties: Object.fromEntries(entries), required };
	} catch {
		return undefined;
	}
}

/**
 * The fused tool's parameters: the host's own shape plus `then_run`, as plain
 * JSON. A host that publishes its schema through `toJsonSchema()` hands back
 * plain objects, and wrapping one in `Type.Optional` throws under the TypeBox
 * build OMP ships, so optionality is expressed the JSON Schema way: only
 * `required` names a mandatory parameter.
 */
export function fusedToolParameters(parameters: unknown, thenRunDescription: string): DeclaredSchema | undefined {
	const declared = declaredSchema(parameters);
	if (!declared) return undefined;
	return {
		properties: { ...declared.properties, then_run: thenRunJsonSchema(thenRunDescription) },
		required: [...declared.required],
	};
}

/** A declared shape as the TypeBox value `registerTool` expects. */
function asParameters(schema: DeclaredSchema) {
	return Type.Unsafe({ type: "object", properties: schema.properties, required: [...schema.required] });
}

/**
 * The schema the host currently publishes for a built-in tool. Reflects the
 * session's active settings (e.g. OMP's `edit.mode`), unlike a definition built
 * at load time. Undefined when the host offers no tool listing or the schema
 * cannot be read.
 */
export function liveToolSchema(pi: ExtensionAPI, name: string): DeclaredSchema | undefined {
	try {
		const listing = (pi as { getAllTools?: () => readonly { name: string; parameters: unknown }[] }).getAllTools;
		if (typeof listing !== "function") return undefined;
		const tool = listing.call(pi).find((candidate) => candidate.name === name);
		const declared = declaredSchema(tool?.parameters);
		if (!declared) return undefined;
		const { then_run: _ownParameter, ...properties } = declared.properties;
		return { properties, required: declared.required.filter((key) => key !== "then_run") };
	} catch {
		return undefined;
	}
}

function unquote(raw: string): string {
	const trimmed = raw.trim();
	const first = trimmed[0];
	if (trimmed.length >= 2 && (first === '"' || first === "'") && trimmed.endsWith(first)) return trimmed.slice(1, -1);
	return trimmed;
}

/** The path of a `[PATH#TAG]` (or untagged `[PATH]`) section header, quotes removed. */
function sectionPath(line: string): string | undefined {
	const trimmed = line.trimEnd();
	if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) return undefined;
	let inner = trimmed.slice(1, -1).trim();
	const tag = /#[0-9A-Fa-f]{4}$/.exec(inner);
	if (tag) inner = inner.slice(0, tag.index);
	const path = unquote(inner);
	return path.length > 0 ? path : undefined;
}

/**
 * Paths a mutation touches. Pi's replace-mode tools name one `path`; a host whose
 * edit tool takes a single patch string names them in `[PATH#TAG]` section
 * headers, and a `MV DEST` operation adds its destination. A section that only
 * deletes (`REM`) keeps its own path: the pre-command check then proves it
 * stays absent.
 */
export function mutationTargets(input: unknown): string[] {
	const record = (input ?? {}) as { path?: unknown; input?: unknown };
	if (typeof record.path === "string") return [record.path];
	if (typeof record.input !== "string") return [];
	const targets: string[] = [];
	let inSection = false;
	for (const rawLine of record.input.replace(/^\uFEFF/, "").split("\n")) {
		const line = rawLine.replace(/\r$/, "");
		const path = sectionPath(line);
		if (path !== undefined) {
			targets.push(path);
			inSection = true;
			continue;
		}
		const trimmed = line.trim();
		if (inSection && trimmed.startsWith("MV ")) {
			const destination = unquote(trimmed.slice(3));
			if (destination) targets.push(destination);
		}
	}
	return [...new Set(targets)];
}

/** Where a target is locked and verified: its filesystem path, or its literal host-scheme name. */
function targetKey(cwd: string, target: string): string {
	return isHostSchemeTarget(target) ? target : resolveToolPath(cwd, target);
}

type ApprovalDecision =
	| string
	| { tier: string; policy?: string; reason?: string; override?: boolean; policyKey?: string };

function splitThenRun(args: unknown): { rest: unknown; thenRun: ThenRunInput | undefined } {
	if (!args || typeof args !== "object") return { rest: args, thenRun: undefined };
	const { then_run: thenRun, ...rest } = args as { then_run?: ThenRunInput };
	return { rest, thenRun };
}

/**
 * The approval declaration of a fused tool, for hosts that classify tool calls
 * (OMP's `approval`). Without `then_run` the built-in decision is unchanged.
 * With it, the call also executes a command, so it is classified as `exec` and
 * scoped to the host's `bash` policy (`policyKey`): a `bash: deny` blocks it, and
 * an unset `bash` policy falls back to the built-in tool's own. A built-in
 * `deny` is never relaxed. Undefined when the host declares no approval.
 *
 * Limit: the host consults one user policy per call, so a set `bash` policy
 * takes precedence over the user's `edit`/`write` policy for calls that carry
 * `then_run`.
 */
export function fusedApproval(native: unknown): ((args: unknown) => ApprovalDecision) | undefined {
	if (native === undefined || native === null) return undefined;
	return (args) => {
		const { rest, thenRun } = splitThenRun(args);
		const decision = (
			typeof native === "function" ? (native as (value: unknown) => ApprovalDecision)(rest) : native
		) as ApprovalDecision;
		if (thenRun === undefined) return decision;
		if (typeof decision === "object" && decision.policy === "deny") return decision;
		// A must-prompt signal from the built-in tool survives the reclassification.
		const override = typeof decision === "object" && decision.override === true;
		return {
			tier: "exec",
			policyKey: "bash",
			reason: `then_run: ${String(thenRun.command ?? "")}`,
			...(override ? { override } : {}),
		};
	};
}

type ArgsHook<T> = (args: unknown) => T;

/**
 * Approval hooks of a fused tool: the decision above, plus the prompt lines the
 * host shows. The built-in tool names its files there, so a call carrying
 * `then_run` also names the command.
 */
export function fusedApprovalHooks(template: unknown): {
	approval?: ArgsHook<ApprovalDecision>;
	formatApprovalDetails?: ArgsHook<string | string[]>;
	approvalCacheTargets?: ArgsHook<readonly string[] | undefined>;
} {
	const native = template as {
		approval?: unknown;
		formatApprovalDetails?: ArgsHook<string | string[] | undefined>;
		approvalCacheTargets?: ArgsHook<readonly string[] | undefined>;
	};
	const approval = fusedApproval(native.approval);
	if (!approval) return {};
	return {
		approval,
		formatApprovalDetails: (args) => {
			const { rest, thenRun } = splitThenRun(args);
			const details = native.formatApprovalDetails?.(rest) ?? [];
			const lines = typeof details === "string" ? [details] : [...details];
			return thenRun === undefined ? lines : [...lines, `Command: ${String(thenRun.command ?? "")}`];
		},
		// The host uses these targets for `tools.restrictedFolders`, so they pass
		// through untouched rather than being replaced or dropped here.
		...(native.approvalCacheTargets ? { approvalCacheTargets: native.approvalCacheTargets } : {}),
	};
}

type NativeInvoker = <T>(
	params: Record<string, unknown>,
	options?: { signal?: AbortSignal; onUpdate?: AgentToolUpdateCallback<T> },
) => Promise<AgentToolResult<T>>;

/**
 * Run the mutation through the host's own tool when it offers same-name
 * delegation, so the session's state and restrictions apply; otherwise through
 * the public definition.
 */
function runMutation<T>(
	ctx: ExtensionToolContext,
	params: Record<string, unknown>,
	signal: AbortSignal | undefined,
	onUpdate: AgentToolUpdateCallback<T> | undefined,
	fallback: () => Promise<AgentToolResult<T>>,
): Promise<AgentToolResult<T>> {
	const host = ctx as ExtensionToolContext & { invokeTool?: NativeInvoker };
	return typeof host.invokeTool === "function" ? host.invokeTool<T>(params, { signal, onUpdate }) : fallback();
}

export function createActionFusionExtension(options: ActionFusionOptions = {}): ExtensionFactory {
	const baseEdit = memoizeByCwd((cwd: string) => createEditToolDefinition(cwd, options.editOptions));
	const baseWrite = memoizeByCwd((cwd: string) => createWriteToolDefinition(cwd, options.writeOptions));

	return (pi: ExtensionAPI) => {
		const editTemplate = baseEdit(process.cwd());
		const writeTemplate = baseWrite(process.cwd());
		const editDeclared = fusedToolParameters(editTemplate.parameters, EDIT_THEN_RUN_DESCRIPTION);
		const writeDeclared = fusedToolParameters(writeTemplate.parameters, WRITE_THEN_RUN_DESCRIPTION);

		// The fused queue and the pre-command check are keyed on the mutation's
		// targets, so replace a mutation tool only when its schema names them: a
		// `path` field, or the patch-string shape whose `[PATH#TAG]` sections carry
		// the paths. A host with any other shape keeps its built-in tool.
		const registerEdit = (declared: DeclaredSchema): void => {
			if (!("path" in declared.properties || "input" in declared.properties)) return;
			const parameters = asParameters(declared) as unknown as typeof editTemplate.parameters;
			pi.registerTool<typeof editTemplate.parameters, EditToolDetails | undefined>({
				...editTemplate,
				...fusedApprovalHooks(editTemplate),
				parameters,
				async execute(toolCallId, input, signal, onUpdate, ctx) {
					const { then_run, ...editInput } = input as typeof input & { then_run?: ThenRunInput };
					const mutate = () =>
						runMutation(ctx, editInput, signal, onUpdate, () =>
							baseEdit(ctx.cwd).execute(toolCallId, editInput, signal, onUpdate, ctx),
						);
					const targets = mutationTargets(input);
					if (targets.length === 0) {
						// Nothing to lock or verify: apply the edit, never the command.
						const result = await mutate();
						if (then_run === undefined) return result;
						return {
							...result,
							content: [
								...result.content,
								{
									type: "text",
									text: `${THEN_RUN_SKIPPED} The edit names no target that can be locked and verified; the command was not run.`,
								},
							],
						};
					}
					const result = await executeMutationThenRun({
						toolCallId,
						absolutePath: targets.map((target) => targetKey(ctx.cwd, target)),
						thenRun: then_run,
						bashOptions: options.bashOptions,
						signal,
						ctx,
						mutate,
					});
					if (
						then_run &&
						result.content.some((block) => block.type === "text" && block.text.includes(THEN_RUN_SUCCEEDED))
					) {
						showSolPiSavings(ctx, "Action Fusion", "1 model round-trip avoided");
					}
					return result;
				},
				renderCall: (args, theme, context) => {
					const base = baseEdit(context.cwd).renderCall!(args, theme, context);
					return (args as { then_run?: ThenRunInput }).then_run
						? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
						: base;
				},
				renderResult: (result, resultOptions, theme, context) => {
					const base = baseEdit(context.cwd).renderResult!(result, resultOptions, theme, context);
					return (context.args as { then_run?: ThenRunInput }).then_run
						? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
						: base;
				},
			});
		};

		// A host that classifies its built-in tools (an `approval` declaration)
		// may also publish a mode-dependent edit schema. A definition built at
		// load time only knows the default mode, so register once the session has
		// started and its live schema is listed; fall back to the load-time schema.
		const hostClassifiesTools = (editTemplate as { approval?: unknown }).approval !== undefined && typeof pi.on === "function";
		if (hostClassifiesTools) {
			pi.on("session_start", () => {
				const live = liveToolSchema(pi, "edit");
				const declared = live
					? fusedToolParameters({ properties: live.properties, required: live.required }, EDIT_THEN_RUN_DESCRIPTION)
					: undefined;
				registerEdit(declared ?? editDeclared ?? { properties: {}, required: [] });
			});
		} else {
			registerEdit(editDeclared ?? { properties: {}, required: [] });
		}

		if (!writeDeclared || !("path" in writeDeclared.properties)) return;
		const writeParameters = asParameters(writeDeclared) as unknown as typeof writeTemplate.parameters;

		pi.registerTool<typeof writeParameters, undefined>({
			...writeTemplate,
			...fusedApprovalHooks(writeTemplate),
			parameters: writeParameters,
			async execute(toolCallId, input, signal, onUpdate, ctx) {
				const { then_run, ...writeInput } = input as typeof input & { then_run?: ThenRunInput };
				const result = await executeMutationThenRun({
					toolCallId,
					absolutePath: targetKey(ctx.cwd, input.path),
					thenRun: then_run,
					bashOptions: options.bashOptions,
					signal,
					ctx,
					mutate: () =>
						runMutation(ctx, writeInput, signal, onUpdate, () =>
							baseWrite(ctx.cwd).execute(toolCallId, writeInput, signal, onUpdate, ctx),
						),
				});
				if (
					then_run &&
					result.content.some((block) => block.type === "text" && block.text.includes(THEN_RUN_SUCCEEDED))
				) {
					showSolPiSavings(ctx, "Action Fusion", "1 model round-trip avoided");
				}
				return result;
			},
			renderCall: (args, theme, context) => {
				const base = baseWrite(context.cwd).renderCall!(args, theme, context);
				return (args as { then_run?: ThenRunInput }).then_run
					? renderSolPiTool(theme, "Action Fusion", "1 model round-trip avoided", base)
					: base;
			},
			renderResult: (result, resultOptions, theme, context) => {
				const base = baseWrite(context.cwd).renderResult!(result, resultOptions, theme, context);
				return (context.args as { then_run?: ThenRunInput }).then_run
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
