/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

/**
 * Boundary/decision classification for a single `turn_end` invocation.
 *
 * Mirrors the exact guard order in `extension.ts`'s `turn_end` handler:
 *
 *   const boundary = pendingBoundary; pendingBoundary = undefined;
 *   if (!boundary || selected) return;
 *   const toolResult = event.toolResults.find(...);
 *   if (
 *     event.message.role !== "assistant" ||
 *     event.message.stopReason === "error" ||
 *     event.message.stopReason === "aborted" ||
 *     context.signal?.aborted ||
 *     !toolResult ||
 *     toolResult.isError
 *   ) return;
 *   // -> decideCompaction() is evaluated
 *
 * `!boundary` and `selected` are independent booleans and can both be true;
 * this classifier picks the first bucket in the same left-to-right order the
 * guards already evaluate in, so it reports the same case the original code
 * short-circuits on. `raw` retains every underlying boolean so a later
 * consumer can tell overlapping conditions apart instead of trusting only
 * the single reported bucket.
 */
export type TurnEndBucket =
	| "no_boundary"
	| "protection_candidate"
	| "already_selected"
	| "turn_not_assistant"
	| "turn_error"
	| "turn_aborted"
	| "context_aborted"
	| "boundary_tool_result_missing"
	| "boundary_tool_result_error"
	| "decision_evaluated";

export type TurnEndClassificationInput = {
	readonly hasPendingBoundary: boolean;
	/** A budget/window trigger permits an otherwise boundary-less safe turn to be evaluated. */
	readonly hasProtectionCandidate?: boolean;
	readonly hasAlreadySelected: boolean;
	readonly messageRole: string | undefined;
	readonly stopReason: string | undefined;
	readonly contextAborted: boolean;
	readonly toolResultFound: boolean;
	readonly toolResultIsError: boolean;
};

export type TurnEndClassification = {
	readonly bucket: TurnEndBucket;
	readonly decisionEvaluated: boolean;
	readonly raw: TurnEndClassificationInput;
};

export function classifyTurnEnd(input: TurnEndClassificationInput): TurnEndClassification {
	const candidate = input.hasPendingBoundary || input.hasProtectionCandidate === true;
	const bucket: TurnEndBucket = !candidate
		? "no_boundary"
		: input.hasAlreadySelected
			? "already_selected"
			: input.messageRole !== "assistant"
				? "turn_not_assistant"
				: input.stopReason === "error"
					? "turn_error"
					: input.stopReason === "aborted"
						? "turn_aborted"
						: input.contextAborted
							? "context_aborted"
							: input.hasPendingBoundary && !input.toolResultFound
								? "boundary_tool_result_missing"
								: input.hasPendingBoundary && input.toolResultIsError
									? "boundary_tool_result_error"
									: input.hasPendingBoundary
										? "decision_evaluated"
										: "protection_candidate";
	return {
		bucket,
		decisionEvaluated: bucket === "decision_evaluated" || bucket === "protection_candidate",
		raw: input,
	};
}
