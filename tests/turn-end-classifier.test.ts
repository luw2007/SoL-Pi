/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { describe, expect, it } from "vitest";
import {
	classifyTurnEnd,
	type TurnEndBucket,
	type TurnEndClassificationInput,
} from "../src/sol-pi/extensions/online-context-compact/turn-end-classifier.ts";

const BASE: TurnEndClassificationInput = {
	hasPendingBoundary: true,
	hasAlreadySelected: false,
	messageRole: "assistant",
	stopReason: "toolUse",
	contextAborted: false,
	toolResultFound: true,
	toolResultIsError: false,
};

describe("classifyTurnEnd — every bucket, in the same left-to-right order the original guard evaluates", () => {
	const cases: ReadonlyArray<{
		readonly name: string;
		readonly patch: Partial<TurnEndClassificationInput>;
		readonly bucket: TurnEndBucket;
		readonly decisionEvaluated: boolean;
	}> = [
		{ name: "no pending boundary", patch: { hasPendingBoundary: false }, bucket: "no_boundary", decisionEvaluated: false },
		{ name: "already selected", patch: { hasAlreadySelected: true }, bucket: "already_selected", decisionEvaluated: false },
		{ name: "turn message is not assistant", patch: { messageRole: "user" }, bucket: "turn_not_assistant", decisionEvaluated: false },
		{ name: "turn stopped on error", patch: { stopReason: "error" }, bucket: "turn_error", decisionEvaluated: false },
		{ name: "turn was aborted", patch: { stopReason: "aborted" }, bucket: "turn_aborted", decisionEvaluated: false },
		{ name: "context signal aborted", patch: { contextAborted: true }, bucket: "context_aborted", decisionEvaluated: false },
		{ name: "boundary tool result missing", patch: { toolResultFound: false }, bucket: "boundary_tool_result_missing", decisionEvaluated: false },
		{ name: "boundary tool result errored", patch: { toolResultIsError: true }, bucket: "boundary_tool_result_error", decisionEvaluated: false },
		{ name: "every guard clears", patch: {}, bucket: "decision_evaluated", decisionEvaluated: true },
	];

	for (const { name, patch, bucket, decisionEvaluated } of cases) {
		it(`classifies "${name}" as ${bucket}`, () => {
			const result = classifyTurnEnd({ ...BASE, ...patch });
			expect(result.bucket).toBe(bucket);
			expect(result.decisionEvaluated).toBe(decisionEvaluated);
			expect(result.raw).toEqual({ ...BASE, ...patch });
		});
	}
});

describe("classifyTurnEnd — precedence when multiple conditions overlap", () => {
	it("no_boundary wins over already_selected, matching `if (!boundary || selected) return` short-circuiting on the first operand", () => {
		const result = classifyTurnEnd({ ...BASE, hasPendingBoundary: false, hasAlreadySelected: true });
		expect(result.bucket).toBe("no_boundary");
	});

	it("no_boundary wins over every later guard, including a genuinely errored turn", () => {
		const result = classifyTurnEnd({
			...BASE,
			hasPendingBoundary: false,
			messageRole: "user",
			stopReason: "error",
			contextAborted: true,
			toolResultFound: false,
			toolResultIsError: true,
		});
		expect(result.bucket).toBe("no_boundary");
	});

	it("already_selected wins over turn_not_assistant when both hold", () => {
		const result = classifyTurnEnd({ ...BASE, hasAlreadySelected: true, messageRole: "user" });
		expect(result.bucket).toBe("already_selected");
	});

	it("turn_error wins over turn_aborted and context_aborted, matching the original `||` chain's left-to-right order", () => {
		const result = classifyTurnEnd({ ...BASE, stopReason: "error", contextAborted: true });
		expect(result.bucket).toBe("turn_error");
	});

	it("turn_aborted wins over context_aborted", () => {
		const result = classifyTurnEnd({ ...BASE, stopReason: "aborted", contextAborted: true });
		expect(result.bucket).toBe("turn_aborted");
	});

	it("context_aborted wins over a missing tool result", () => {
		const result = classifyTurnEnd({ ...BASE, contextAborted: true, toolResultFound: false });
		expect(result.bucket).toBe("context_aborted");
	});

	it("boundary_tool_result_missing wins over toolResultIsError (isError is meaningless without a found result)", () => {
		const result = classifyTurnEnd({ ...BASE, toolResultFound: false, toolResultIsError: true });
		expect(result.bucket).toBe("boundary_tool_result_missing");
	});
});
